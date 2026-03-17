from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class ShardDataset(torch.utils.data.Dataset):
    def __init__(self, shards_dir: str, event_indices: np.ndarray, cache_size: int = 2):
        meta = _load(Path(shards_dir) / "index.pt")
        self.shards_dir = Path(shards_dir)
        self.shard_events = int(meta["shard_events"])
        self.labels_all = np.asarray(meta["labels"], dtype=np.uint8)
        self.weights_all = np.asarray(meta["weights"], dtype=np.float32)
        self.event_indices = np.asarray(event_indices, dtype=np.int64)
        self.labels = self.labels_all[self.event_indices].astype(np.uint8, copy=False)
        self.weights = self.weights_all[self.event_indices].astype(np.float32, copy=False)
        self.shard_ids = (self.event_indices // self.shard_events).astype(np.int64, copy=False)
        self.local_ids = (self.event_indices - self.shard_ids * self.shard_events).astype(np.int64, copy=False)
        self.cache_size = max(1, int(cache_size))
        self._cache = OrderedDict()

    def __len__(self):
        return int(self.event_indices.shape[0])

    def _load_shard(self, sid: int):
        sid = int(sid)
        if sid in self._cache:
            self._cache.move_to_end(sid)
            return self._cache[sid]
        shard = _load(self.shards_dir / f"shard_{sid:05d}.pt")
        self._cache[sid] = shard
        self._cache.move_to_end(sid)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return shard

    @staticmethod
    def _slice_one(shard, local: int):
        local = int(local)
        start = int(shard["starts"][local].item())
        stop = int(shard["starts"][local + 1].item())
        coords = shard["coords"][start:stop].to(dtype=torch.int32)
        feats = shard["feats"][start:stop].to(dtype=torch.float32)
        return coords, feats

    def __getitem__(self, i: int):
        event_index = int(self.event_indices[i])
        shard_id = int(event_index // self.shard_events)
        shard = self._load_shard(shard_id)
        local = event_index - int(shard["start_event"])
        coords, feats = self._slice_one(shard, local)
        return coords, feats, float(self.labels[i])


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: ShardDataset, batch_size: int, seed: int = 123):
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even")
        self.ds = dataset
        self.half = batch_size // 2
        self.rng = np.random.default_rng(int(seed))
        labels = np.asarray(self.ds.labels, dtype=np.uint8)
        weights = np.asarray(self.ds.weights, dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        self.signal = np.flatnonzero(labels == 1)
        self.background = np.flatnonzero(labels == 0)
        if self.signal.size == 0 or self.background.size == 0:
            raise ValueError("need both signal and background in the training split")
        signal_weights = weights[self.signal]
        background_weights = weights[self.background]
        if signal_weights.sum() <= 0 or background_weights.sum() <= 0:
            raise ValueError("weights must sum to >0 within each class")
        self.signal_prob = (signal_weights / signal_weights.sum()).astype(np.float64, copy=False)
        self.background_prob = (background_weights / background_weights.sum()).astype(np.float64, copy=False)

    def __iter__(self):
        while True:
            signal = self.rng.choice(self.signal, size=self.half, replace=True, p=self.signal_prob)
            background = self.rng.choice(
                self.background,
                size=self.half,
                replace=True,
                p=self.background_prob,
            )
            batch = np.concatenate([signal, background]).astype(np.int64, copy=False)
            order = np.lexsort((self.ds.local_ids[batch], self.ds.shard_ids[batch]))
            yield batch[order].tolist()


def collate_sparse_planes(batch, plane_names=("u", "v", "w")):
    coords_by_plane = {name: [] for name in plane_names}
    feats_by_plane = {name: [] for name in plane_names}
    labels = []
    available_mask = torch.zeros((len(batch), len(plane_names)), dtype=torch.float32)
    for batch_index, (coords, feats, label) in enumerate(batch):
        coords = torch.as_tensor(coords, dtype=torch.int32).contiguous()
        feats = torch.as_tensor(feats, dtype=torch.float32).contiguous()
        labels.append(label)
        for plane_index, name in enumerate(plane_names):
            mask = coords[:, 0] == plane_index
            if mask.any() and (feats[mask, 0] > 0).any():
                available_mask[batch_index, plane_index] = 1.0
                plane_coords = coords[mask][:, 1:3]
                plane_feats = feats[mask]
            else:
                plane_coords = torch.zeros((1, 2), dtype=torch.int32)
                plane_feats = torch.zeros((1, feats.shape[1]), dtype=torch.float32)
            batch_col = torch.full((plane_coords.shape[0], 1), batch_index, dtype=torch.int32)
            coords_by_plane[name].append(torch.cat([batch_col, plane_coords], dim=1))
            feats_by_plane[name].append(plane_feats)
    coords = {name: torch.cat(coords_by_plane[name], dim=0).contiguous() for name in plane_names}
    feats = {name: torch.cat(feats_by_plane[name], dim=0).contiguous() for name in plane_names}
    return coords, feats, torch.tensor(labels, dtype=torch.float32), available_mask


collate_me_fusion = collate_sparse_planes
