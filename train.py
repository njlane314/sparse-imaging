from pathlib import Path
from typing import Dict, Optional, Tuple

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn

try:
    import config as cfg
    from dataset import BalancedBatchSampler, ShardDataset, collate_sparse_planes
    from model import MultiPlaneSparseEvidenceUResNet
except ImportError:
    from . import config as cfg
    from .dataset import BalancedBatchSampler, ShardDataset, collate_sparse_planes
    from .model import MultiPlaneSparseEvidenceUResNet


def poly_lr(step, max_steps, lr0, power):
    if max_steps <= 0:
        return lr0
    frac = max(0.0, 1.0 - float(step) / float(max_steps))
    return float(lr0) * (frac**power)


def _capture_random_state():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _checkpoint_path_for_step(base_path: Path, step: int) -> Path:
    return base_path.with_name(f"{base_path.stem}_step{step:07d}{base_path.suffix}")


def _split_indices(
    labels: np.ndarray,
    keep_mask: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(keep_mask)
    if indices.size < 2:
        raise ValueError("need at least two valid events after filtering")
    rng = np.random.default_rng(seed)
    indices = indices[rng.permutation(indices.size)]
    n_val = 0
    if val_fraction > 0:
        n_val = min(max(int(val_fraction * indices.size), 1), indices.size - 1)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    for cls in (0, 1):
        if train_idx.size == 0 or not np.any(labels[train_idx] == cls):
            candidates = val_idx[labels[val_idx] == cls]
            if candidates.size == 0:
                raise ValueError("training split requires at least one event from each class")
            take = int(candidates[0])
            val_idx = val_idx[val_idx != take]
            train_idx = np.concatenate([train_idx, [take]])
    return train_idx.astype(np.int64, copy=False), val_idx.astype(np.int64, copy=False)


def _take_batches(loader_iter, loader, n):
    batches = []
    for _ in range(n):
        try:
            batches.append(next(loader_iter))
        except StopIteration:
            loader_iter = iter(loader)
            batches.append(next(loader_iter))
    return batches, loader_iter


def _make_inputs(planes, coords_by_plane, feats_by_plane, device):
    inputs: Dict[str, ME.SparseTensor] = {}
    for name in planes:
        inputs[name] = ME.SparseTensor(
            features=feats_by_plane[name].to(device, non_blocking=True),
            coordinates=coords_by_plane[name],
            device=device,
        )
    return inputs


def train_llr():
    if cfg.MAX_STEPS <= 0:
        raise ValueError("MAX_STEPS must be > 0")
    if not 0 <= cfg.VAL_FRACTION < 1:
        raise ValueError("VAL_FRACTION must be in [0, 1)")
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = _load(Path(cfg.SHARDS_DIR) / "index.pt")
    n_events = int(meta["n_events"])
    labels_all = np.asarray(meta["labels"], dtype=np.uint8).reshape(-1)
    if labels_all.shape[0] != n_events:
        raise ValueError(f"index.pt labels has len={labels_all.shape[0]} but n_events={n_events}")
    keep_mask = np.ones(n_events, dtype=bool)
    nnz_all: Optional[np.ndarray] = None
    if "nnz" in meta and meta["nnz"] is not None:
        if isinstance(meta["nnz"], torch.Tensor):
            nnz_all = meta["nnz"].to(dtype=torch.int64).cpu().numpy().reshape(-1)
        else:
            nnz_all = np.asarray(meta["nnz"], dtype=np.int64).reshape(-1)
    if nnz_all is not None:
        if nnz_all.shape[0] != n_events:
            raise ValueError(f"index.pt nnz has len={nnz_all.shape[0]} but n_events={n_events}")
        keep_mask = nnz_all > 0
        dropped = int(n_events - keep_mask.sum())
        print(f"[data] keeping nnz>0 events: {int(keep_mask.sum())}/{n_events} (dropped {dropped})")
        if not keep_mask.any():
            raise ValueError("all events have nnz==0 after sparsification")
    train_idx, val_idx = _split_indices(labels_all, keep_mask, cfg.VAL_FRACTION, cfg.SEED)
    print(f"[data] train={train_idx.size} val={val_idx.size}")
    ds_train = ShardDataset(cfg.SHARDS_DIR, train_idx, cache_size=2)
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_sampler=BalancedBatchSampler(ds_train, batch_size=cfg.BATCH_SIZE, seed=cfg.SEED),
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_sparse_planes,
        pin_memory=True,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )
    dl_val = None
    if val_idx.size > 0:
        ds_val = ShardDataset(cfg.SHARDS_DIR, val_idx, cache_size=2)
        dl_val = torch.utils.data.DataLoader(
            ds_val,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            collate_fn=collate_sparse_planes,
            pin_memory=True,
            persistent_workers=cfg.NUM_WORKERS > 0,
        )
    val_every = int(cfg.VAL_EVERY)
    val_num_batches = int(cfg.VAL_BATCHES)
    val_cache_batches = int(cfg.VAL_CACHE_BATCHES)
    if val_every < 0:
        raise ValueError("VAL_EVERY must be >= 0")
    if val_num_batches <= 0:
        raise ValueError("VAL_BATCHES must be >= 1")
    if val_cache_batches < 0:
        raise ValueError("VAL_CACHE_BATCHES must be >= 0")
    cached_val_batches = []
    val_it = None
    if dl_val is None:
        val_every = 0
    else:
        val_it = iter(dl_val)
        if val_every > 0 and val_cache_batches > 0:
            cached_val_batches, val_it = _take_batches(val_it, dl_val, val_cache_batches)
            print(f"[val] cached {len(cached_val_batches)} val batches in RAM")
    planes = ("u", "v", "w")
    model = MultiPlaneSparseEvidenceUResNet(
        in_ch=2,
        plane_names=planes,
        preset=cfg.BACKBONE,
        embed_dim=cfg.EMBED_DIM,
    ).to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.LR0,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    train_it = iter(dl_train)
    log_path = Path(cfg.LOSS_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_flush_every = int(cfg.LOG_FLUSH_EVERY)
    train_diagnostics_every = int(cfg.TRAIN_DIAGNOSTICS_EVERY)
    if log_flush_every < 0:
        raise ValueError("LOG_FLUSH_EVERY must be >= 0")
    if train_diagnostics_every < 0:
        raise ValueError("TRAIN_DIAGNOSTICS_EVERY must be >= 0")
    ckpt_base_path = Path(cfg.CHECKPOINT_PATH)
    ckpt_base_path.parent.mkdir(parents=True, exist_ok=True)
    initial_random_state = _capture_random_state()
    with log_path.open("w", buffering=64 * 1024) as log_f:
        log_f.write("#step\tis_val\tloss\n")
        if cfg.CHECKPOINT_EVERY > 0:
            ckpt_path = _checkpoint_path_for_step(ckpt_base_path, step=0)
            torch.save(
                {
                    "step": 0,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "initial_random_state": initial_random_state,
                    "random_state": initial_random_state,
                },
                ckpt_path,
            )
            print(f"[ckpt] saved initial state -> {ckpt_path}")
        for step in range(1, cfg.MAX_STEPS + 1):
            model.train()
            coords_by_plane, feats_by_plane, y, available_mask = next(train_it)
            y = y.to(device, non_blocking=True)
            logits = model(
                _make_inputs(planes, coords_by_plane, feats_by_plane, device),
                available_mask=available_mask.to(device, non_blocking=True),
            ).squeeze(1)
            if logits.shape != y.shape:
                raise RuntimeError(f"logits shape {tuple(logits.shape)} != y shape {tuple(y.shape)}")
            loss = loss_fn(logits, y)
            log_f.write(f"{step}\t0\t{loss.item():.8g}\n")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if train_diagnostics_every > 0 and step % train_diagnostics_every == 0:
                with torch.no_grad():
                    print("logits mean/std:", logits.mean().item(), logits.std().item())
                grad_norm = 0.0
                nonzero = 0
                for param in model.parameters():
                    if param.grad is not None:
                        grad = param.grad.detach()
                        grad_norm += grad.norm().item() ** 2
                        nonzero += int(grad.abs().sum().item() > 0)
                print("grad L2:", grad_norm**0.5, "nonzero_grad_params:", nonzero)
            opt.step()
            lr = poly_lr(step, cfg.MAX_STEPS, cfg.LR0, cfg.POLY_POWER)
            for group in opt.param_groups:
                group["lr"] = lr
            if step % 200 == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    acc = ((probs > 0.5) == (y > 0.5)).float().mean().item()
                print(f"step {step:7d}  loss {loss.item():.4f}  acc {acc:.3f}  lr {lr:.3e}")
            do_val = dl_val is not None and val_every > 0 and (step % val_every == 0 or step == cfg.MAX_STEPS)
            if do_val:
                model.eval()
                with torch.no_grad():
                    batches = cached_val_batches
                    if not batches:
                        batches, val_it = _take_batches(val_it, dl_val, val_num_batches)
                    vloss = 0.0
                    for val_coords, val_feats, y_val, val_available_mask in batches:
                        y_val = y_val.to(device, non_blocking=True)
                        val_logits = model(
                            _make_inputs(planes, val_coords, val_feats, device),
                            available_mask=val_available_mask.to(device, non_blocking=True),
                        ).squeeze(1)
                        if val_logits.shape != y_val.shape:
                            raise RuntimeError(f"[val] logits shape {tuple(val_logits.shape)} != y shape {tuple(y_val.shape)}")
                        vloss += loss_fn(val_logits, y_val).item()
                    val_loss = vloss / float(len(batches))
                if step % 200 == 0:
                    print(f"[val] step {step:7d}  loss {val_loss:.4f}")
                log_f.write(f"{step}\t1\t{val_loss:.8g}\n")
            if log_flush_every > 0 and step % log_flush_every == 0:
                log_f.flush()
            if cfg.CHECKPOINT_EVERY > 0 and step % cfg.CHECKPOINT_EVERY == 0:
                ckpt_path = _checkpoint_path_for_step(ckpt_base_path, step=step)
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "initial_random_state": initial_random_state,
                        "random_state": _capture_random_state(),
                    },
                    ckpt_path,
                )
                print(f"[ckpt] saved step {step:7d} -> {ckpt_path}")


if __name__ == "__main__":
    train_llr()
