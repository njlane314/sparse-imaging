from __future__ import annotations

import concurrent.futures as cf
import faulthandler
import gc
import os
import signal
import sys
import tempfile
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import uproot

try:
    import config as cfg
except ImportError:
    from . import config as cfg

_DUMMY_COORDS = np.array([[0, 0, 0]], dtype=np.int32)
_DUMMY_FEATS = np.array([[0.0, 0.0]], dtype=np.float16)


def _atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.tmp.", delete=False) as tf:
        tmp = Path(tf.name)
    torch.save(obj, tmp)
    tmp.replace(path)


def event_to_sparse(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    *,
    width: int,
    thresh: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    iu = np.flatnonzero(u > thresh)
    iv = np.flatnonzero(v > thresh)
    iw = np.flatnonzero(w > thresh)
    total = iu.size + iv.size + iw.size
    if total == 0:
        return _DUMMY_COORDS, _DUMMY_FEATS, 0
    coords = np.empty((total, 3), dtype=np.int32)
    feats = np.empty((total, 2), dtype=np.float16)
    offset = 0
    for plane, (arr, idx) in enumerate(((u, iu), (v, iv), (w, iw))):
        count = idx.size
        if count == 0:
            continue
        sl = slice(offset, offset + count)
        coords[sl, 0] = plane
        np.floor_divide(idx, width, out=coords[sl, 1], casting="unsafe")
        np.remainder(idx, width, out=coords[sl, 2], casting="unsafe")
        feats[sl, 0] = 1.0
        vals = arr[idx].astype(np.float16, copy=True)
        np.maximum(vals, 0.0, out=vals)
        np.log1p(vals, out=vals)
        feats[sl, 1] = vals
        offset += count
    return coords, feats, int(total)


def pack_events(coords_list: Iterable[np.ndarray], feats_list: Iterable[np.ndarray]):
    coords_list = list(coords_list)
    feats_list = list(feats_list)
    lengths = np.fromiter((coords.shape[0] for coords in coords_list), dtype=np.int64, count=len(coords_list))
    starts = np.empty(len(lengths) + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(lengths, out=starts[1:])
    total = int(starts[-1])
    coords = np.empty((total, 3), dtype=np.int32)
    feats = np.empty((total, 2), dtype=np.float16)
    offset = 0
    for event_coords, event_feats in zip(coords_list, feats_list):
        count = int(event_coords.shape[0])
        coords[offset:offset + count] = event_coords
        feats[offset:offset + count] = event_feats
        offset += count
    return torch.from_numpy(coords), torch.from_numpy(feats), torch.from_numpy(starts)


def _infer_hw_from_any(raw: np.ndarray, *, fallback_h: int, fallback_w: int) -> Tuple[int, int]:
    try:
        arr = np.asarray(raw)
    except Exception:
        return int(fallback_h), int(fallback_w)
    if arr.dtype != object:
        if arr.ndim >= 3:
            return int(arr.shape[-2]), int(arr.shape[-1])
        if arr.ndim == 2:
            if arr.shape[0] == fallback_h and arr.shape[1] == fallback_w:
                return int(fallback_h), int(fallback_w)
            hw = int(arr.shape[1])
            side = int(np.sqrt(hw))
            if side * side == hw:
                return side, side
            return int(fallback_h), int(fallback_w)
        if arr.ndim == 1:
            hw = int(arr.size)
            if hw == fallback_h * fallback_w:
                return int(fallback_h), int(fallback_w)
            side = int(np.sqrt(hw))
            if side * side == hw:
                return side, side
            return int(fallback_h), int(fallback_w)
    if arr.dtype == object and arr.ndim >= 1 and arr.shape[0] > 0:
        for item in arr:
            if item is None:
                continue
            try:
                value = np.asarray(item)
            except Exception:
                continue
            if value.ndim >= 2:
                return int(value.shape[-2]), int(value.shape[-1])
            hw = int(value.size)
            if hw == fallback_h * fallback_w:
                return int(fallback_h), int(fallback_w)
            side = int(np.sqrt(hw))
            if side * side == hw:
                return side, side
            break
    return int(fallback_h), int(fallback_w)


def _flatten_one(raw, hw: int, branch: str, event_index: int) -> np.ndarray:
    arr = np.asarray(raw).reshape(-1)
    if arr.size != hw:
        raise ValueError(f"{branch}[{event_index}] size {arr.size} != {hw} (H*W)")
    return arr


def write_shards_from_root():
    if cfg.SHARD_EVENTS <= 0:
        raise ValueError("SHARD_EVENTS must be > 0")
    if cfg.CHUNK_EVENTS <= 0:
        raise ValueError("CHUNK_EVENTS must be > 0")
    out_dir = Path(cfg.SHARDS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("shard_*.pt"):
        path.unlink(missing_ok=True)
    for path in out_dir.glob(".shard_*.pt.tmp.*"):
        path.unlink(missing_ok=True)
    index_path = out_dir / "index.pt"
    if index_path.exists():
        index_path.unlink()
    try:
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1)
    except Exception:
        pass
    start_time = time.monotonic()
    use_tty = sys.stdout.isatty()

    def format_eta(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return "--:--:--"
        minutes, sec = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"

    def render_progress(current: int, total: int, width: int = 40) -> None:
        if total <= 0:
            return
        frac = max(0.0, min(1.0, current / total))
        if not use_tty and current < total:
            return
        filled = int(round(frac * width))
        bar = "=" * filled + "-" * (width - filled)
        if current > 0:
            elapsed = time.monotonic() - start_time
            eta = format_eta((total - current) * (elapsed / current))
        else:
            eta = "--:--:--"
        print(f"\rProcessing events: |{bar}| {current}/{total} ETA {eta}", end="", flush=True)

    n_decomp = int(getattr(cfg, "UPROOT_DECOMP_WORKERS", 2))
    decomp_context = cf.ThreadPoolExecutor(max_workers=n_decomp) if n_decomp > 0 else nullcontext()
    with decomp_context as decomp:
        with uproot.open(
            cfg.ROOT_FILE,
            object_cache=None,
            array_cache=None,
            decompression_executor=decomp if n_decomp > 0 else None,
        ) as root_file:
            tree = root_file[cfg.TREE]
            labels_all = tree[cfg.BR_Y].array(library="np").astype(np.uint8).reshape(-1)
            weights_all = tree[cfg.BR_WGT].array(library="np").astype(np.float32).reshape(-1)
            n_events = int(min(labels_all.shape[0], weights_all.shape[0], int(tree.num_entries)))
            if labels_all.shape[0] != n_events or weights_all.shape[0] != n_events or int(tree.num_entries) != n_events:
                warnings.warn(
                    f"entry count mismatch: tree={int(tree.num_entries)} {cfg.BR_Y}={labels_all.shape[0]} {cfg.BR_WGT}={weights_all.shape[0]}; using n_events={n_events}"
                )
            labels = labels_all[:n_events].copy()
            weights = weights_all[:n_events].copy()
            nnz = np.zeros(n_events, dtype=np.int32)
            bad_events: list[int] = []
            bad_logged = 0
            shard_id = 0
            shard_start = 0
            coords_acc = []
            feats_acc = []
            n_acc = 0
            progress_every = max(1, n_events // 200)
            render_progress(0, n_events)
            h = int(cfg.H)
            w = int(cfg.W)
            hw = h * w
            thresh = float(cfg.THRESH)
            for start in range(0, n_events, cfg.CHUNK_EVENTS):
                stop = min(start + cfg.CHUNK_EVENTS, n_events)
                faulthandler.dump_traceback_later(int(getattr(cfg, "FAULTHANDLER_TIMEOUT", 120)), repeat=False)
                try:
                    arrays = tree.arrays(
                        [cfg.BR_U, cfg.BR_V, cfg.BR_W],
                        entry_start=start,
                        entry_stop=stop,
                        library="np",
                    )
                finally:
                    faulthandler.cancel_dump_traceback_later()
                uu_raw, vv_raw, ww_raw = arrays[cfg.BR_U], arrays[cfg.BR_V], arrays[cfg.BR_W]
                if start == 0:
                    hu, wu = _infer_hw_from_any(uu_raw, fallback_h=h, fallback_w=w)
                    hv, wv = _infer_hw_from_any(vv_raw, fallback_h=h, fallback_w=w)
                    hw_infer, ww_infer = _infer_hw_from_any(ww_raw, fallback_h=h, fallback_w=w)
                    if (hu, wu) == (hv, wv) == (hw_infer, ww_infer) and (hu, wu) != (h, w):
                        warnings.warn(f"cfg.H/cfg.W={h}x{w} do not match data={hu}x{wu}; using inferred H/W from file")
                        h, w = int(hu), int(wu)
                        hw = h * w
                for j in range(stop - start):
                    event_index = start + j
                    bad_event = False
                    try:
                        u = _flatten_one(uu_raw[j], hw, cfg.BR_U, event_index)
                    except Exception as exc:
                        if cfg.STRICT_SHAPES:
                            raise
                        bad_event = True
                        if bad_logged < cfg.MAX_BAD_EVENT_LOG:
                            warnings.warn(f"{cfg.BR_U}[{event_index}] malformed ({type(exc).__name__}: {exc}); zero-filling")
                            bad_logged += 1
                    try:
                        v = _flatten_one(vv_raw[j], hw, cfg.BR_V, event_index)
                    except Exception as exc:
                        if cfg.STRICT_SHAPES:
                            raise
                        bad_event = True
                        if bad_logged < cfg.MAX_BAD_EVENT_LOG:
                            warnings.warn(f"{cfg.BR_V}[{event_index}] malformed ({type(exc).__name__}: {exc}); zero-filling")
                            bad_logged += 1
                    try:
                        wplane = _flatten_one(ww_raw[j], hw, cfg.BR_W, event_index)
                    except Exception as exc:
                        if cfg.STRICT_SHAPES:
                            raise
                        bad_event = True
                        if bad_logged < cfg.MAX_BAD_EVENT_LOG:
                            warnings.warn(f"{cfg.BR_W}[{event_index}] malformed ({type(exc).__name__}: {exc}); zero-filling")
                            bad_logged += 1
                    if bad_event:
                        coords, feats, true_nnz = _DUMMY_COORDS, _DUMMY_FEATS, 0
                        bad_events.append(int(event_index))
                    else:
                        try:
                            coords, feats, true_nnz = event_to_sparse(u, v, wplane, width=w, thresh=thresh)
                        except Exception as exc:
                            if cfg.STRICT_SHAPES:
                                raise
                            coords, feats, true_nnz = _DUMMY_COORDS, _DUMMY_FEATS, 0
                            bad_events.append(int(event_index))
                            if bad_logged < cfg.MAX_BAD_EVENT_LOG:
                                warnings.warn(f"event {event_index}: event_to_sparse failed ({type(exc).__name__}: {exc}); zero-filling")
                                bad_logged += 1
                    nnz[event_index] = int(true_nnz)
                    coords_acc.append(coords)
                    feats_acc.append(feats)
                    n_acc += 1
                    if n_acc == cfg.SHARD_EVENTS:
                        coords_t, feats_t, starts_t = pack_events(coords_acc, feats_acc)
                        _atomic_torch_save(
                            {
                                "start_event": int(shard_start),
                                "n_events": int(n_acc),
                                "coords": coords_t,
                                "feats": feats_t,
                                "starts": starts_t,
                            },
                            out_dir / f"shard_{shard_id:05d}.pt",
                        )
                        shard_id += 1
                        shard_start = event_index + 1
                        coords_acc.clear()
                        feats_acc.clear()
                        n_acc = 0
                        gc.collect()
                    if (event_index + 1) % progress_every == 0 or event_index + 1 == n_events:
                        render_progress(event_index + 1, n_events)
            if n_acc:
                coords_t, feats_t, starts_t = pack_events(coords_acc, feats_acc)
                _atomic_torch_save(
                    {
                        "start_event": int(shard_start),
                        "n_events": int(n_acc),
                        "coords": coords_t,
                        "feats": feats_t,
                        "starts": starts_t,
                    },
                    out_dir / f"shard_{shard_id:05d}.pt",
                )
                shard_id += 1
            _atomic_torch_save(
                {
                    "H": int(h),
                    "W": int(w),
                    "thresh": float(thresh),
                    "n_events": int(n_events),
                    "shard_events": int(cfg.SHARD_EVENTS),
                    "labels": labels,
                    "weights": weights,
                    "nnz": nnz,
                    "bad_events": np.asarray(bad_events, dtype=np.int64),
                    "branches": {
                        "u": cfg.BR_U,
                        "v": cfg.BR_V,
                        "w": cfg.BR_W,
                        "y": cfg.BR_Y,
                        "wgt": cfg.BR_WGT,
                    },
                },
                index_path,
            )
    print()
    print(f"wrote {shard_id} shards to {out_dir} (events={n_events})")
    if bad_events:
        print(f"warning: {len(bad_events)} events had missing/malformed data and were zero-filled (see index.pt: bad_events)")


if __name__ == "__main__":
    write_shards_from_root()
