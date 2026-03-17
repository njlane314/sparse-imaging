from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import MinkowskiEngine as ME

class MultiPlaneSparseEvidenceUResNet(nn.Module):
    _PRESETS = {'tiny': {'base': 16, 'blocks': (1, 1, 1)}, 'small': {'base': 32, 'blocks': (1, 1, 1, 1)}, 'base': {'base': 32, 'blocks': (2, 2, 2, 2)}, 'wide': {'base': 48, 'blocks': (2, 2, 2, 2)}}

    @staticmethod
    def _replace_feature(x: ME.SparseTensor, new_F: torch.Tensor) -> ME.SparseTensor:
        if hasattr(x, 'replace_feature'):
            return x.replace_feature(new_F)
        cmk = getattr(x, 'coordinate_map_key', None)
        cm = getattr(x, 'coordinate_manager', None)
        ck = getattr(x, 'coords_key', None)
        cman = getattr(x, 'coords_man', None)
        if cman is None:
            cman = getattr(x, 'coords_manager', None)
        if cmk is not None and cm is not None:
            try:
                return ME.SparseTensor(features=new_F, coordinate_map_key=cmk, coordinate_manager=cm, device=new_F.device)
            except TypeError:
                pass
        if ck is not None and cman is not None:
            try:
                return ME.SparseTensor(features=new_F, coords_key=ck, coords_manager=cman, device=new_F.device)
            except TypeError:
                pass
        return ME.SparseTensor(features=new_F, coordinates=x.C, device=new_F.device)

    @staticmethod
    def _subm_conv(in_ch: int, out_ch: int, *, kernel_size: int, dimension: int, bias: bool=False) -> nn.Module:
        if hasattr(ME, 'MinkowskiSubmanifoldConvolution'):
            return ME.MinkowskiSubmanifoldConvolution(int(in_ch), int(out_ch), kernel_size=kernel_size, dimension=dimension, bias=bias)
        try:
            return ME.MinkowskiConvolution(int(in_ch), int(out_ch), kernel_size=kernel_size, stride=1, dimension=dimension, bias=bias, expand_coordinates=False)
        except TypeError:
            return ME.MinkowskiConvolution(int(in_ch), int(out_ch), kernel_size=kernel_size, stride=1, dimension=dimension, bias=bias)

    @staticmethod
    def _infer_batch_column(coords: torch.Tensor) -> int:
        if coords.ndim != 2 or coords.size(1) == 0:
            return 0
        if coords.size(1) == 1:
            return 0
        first = coords[:, 0]
        last = coords[:, -1]
        first_varies = bool((first != first[:1]).any().item())
        last_varies = bool((last != last[:1]).any().item())
        if first_varies and (not last_varies):
            return 0
        if last_varies and (not first_varies):
            return coords.size(1) - 1
        return 0

    @classmethod
    def _infer_num_batches_from_sparse(cls, x: ME.SparseTensor) -> int:
        if x.C.numel() == 0:
            return 0
        bcol = cls._infer_batch_column(x.C)
        return int(x.C[:, bcol].max().item()) + 1

    @classmethod
    def _global_sparse_to_dense(cls, x: ME.SparseTensor, batch_size: int) -> torch.Tensor:
        out = x.F.new_zeros((int(batch_size), x.F.size(1)))
        if x.F.numel() == 0 or batch_size == 0:
            return out
        if x.F.size(0) == batch_size:
            return x.F
        bcol = cls._infer_batch_column(x.C)
        batch_ids = x.C[:, bcol].long()
        out[batch_ids] = x.F
        return out

    class SparseLayerNorm(nn.Module):

        def __init__(self, c: int, eps: float=1e-06):
            super().__init__()
            self.ln = nn.LayerNorm(int(c), eps=eps)

        def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
            return MultiPlaneSparseEvidenceUResNet._replace_feature(x, self.ln(x.F))

    class ResBlock(nn.Module):

        def __init__(self, cin: int, cout: int, dimension: int=2):
            super().__init__()
            self.conv1 = MultiPlaneSparseEvidenceUResNet._subm_conv(cin, cout, kernel_size=3, dimension=dimension, bias=False)
            self.n1 = MultiPlaneSparseEvidenceUResNet.SparseLayerNorm(cout)
            self.conv2 = MultiPlaneSparseEvidenceUResNet._subm_conv(cout, cout, kernel_size=3, dimension=dimension, bias=False)
            self.n2 = MultiPlaneSparseEvidenceUResNet.SparseLayerNorm(cout)
            self.act = ME.MinkowskiReLU(inplace=False)
            self.proj = None
            if cin != cout:
                self.proj = nn.Sequential(ME.MinkowskiLinear(cin, cout, bias=False), MultiPlaneSparseEvidenceUResNet.SparseLayerNorm(cout))

        def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
            identity = x if self.proj is None else self.proj(x)
            out = self.conv1(x)
            out = self.n1(out)
            out = self.act(out)
            out = self.conv2(out)
            out = self.n2(out)
            return self.act(out + identity)

    class Down(nn.Sequential):

        def __init__(self, cin: int, cout: int, dimension: int=2):
            super().__init__(ME.MinkowskiConvolution(cin, cout, kernel_size=2, stride=2, dimension=dimension, bias=False), MultiPlaneSparseEvidenceUResNet.SparseLayerNorm(cout), ME.MinkowskiReLU(inplace=False))

    class Up(nn.Sequential):

        def __init__(self, cin: int, cout: int, dimension: int=2):
            super().__init__(ME.MinkowskiConvolutionTranspose(cin, cout, kernel_size=2, stride=2, dimension=dimension, bias=False), MultiPlaneSparseEvidenceUResNet.SparseLayerNorm(cout), ME.MinkowskiReLU(inplace=False))

    class ZeroInitCorrection(nn.Module):

        def __init__(self, d: int):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=False), nn.Linear(d, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z)

    def __init__(self, in_ch: int, plane_names: Sequence[str]=('u', 'v', 'w'), *, preset: Optional[str]='base', base: Optional[int]=None, blocks: Optional[Sequence[int]]=None, embed_dim: int=128, dimension: int=2, use_correction: bool=False, separate_plane_heads: bool=False):
        super().__init__()
        if preset is not None:
            if preset not in self._PRESETS:
                raise ValueError(f'unknown preset={preset!r}; choose from {sorted(self._PRESETS)}')
            p = self._PRESETS[preset]
            if base is None:
                base = int(p['base'])
            if blocks is None:
                blocks = tuple((int(v) for v in p['blocks']))
        if base is None or blocks is None:
            raise ValueError('provide either a preset or explicit base and blocks')
        blocks = tuple((int(v) for v in blocks))
        if len(blocks) < 2:
            raise ValueError('UResNet needs at least two resolution levels')
        self.plane_names = tuple(plane_names)
        self.num_views = len(self.plane_names)
        self.base = int(base)
        self.blocks_cfg = blocks
        self.embed_dim = int(embed_dim)
        self.dimension = int(dimension)
        self.separate_plane_heads = bool(separate_plane_heads)
        self.stem = nn.Sequential(self._subm_conv(in_ch, self.base, kernel_size=3, dimension=self.dimension, bias=False), self.SparseLayerNorm(self.base), ME.MinkowskiReLU(inplace=False))
        enc_ch = [self.base * 2 ** i for i in range(len(self.blocks_cfg))]
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = enc_ch[0]
        for i, nb in enumerate(self.blocks_cfg):
            self.enc_blocks.append(nn.Sequential(*[self.ResBlock(ch, ch, dimension=self.dimension) for _ in range(nb)]))
            if i < len(self.blocks_cfg) - 1:
                self.downs.append(self.Down(ch, enc_ch[i + 1], dimension=self.dimension))
                ch = enc_ch[i + 1]
        self.bottleneck = nn.Sequential(self.ResBlock(ch, ch, dimension=self.dimension), self.ResBlock(ch, ch, dimension=self.dimension))
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(len(self.blocks_cfg) - 1)):
            skip_ch = enc_ch[i]
            self.ups.append(self.Up(ch, skip_ch, dimension=self.dimension))
            self.dec_blocks.append(nn.Sequential(self.ResBlock(skip_ch + skip_ch, skip_ch, dimension=self.dimension), *[self.ResBlock(skip_ch, skip_ch, dimension=self.dimension) for _ in range(max(0, self.blocks_cfg[i] - 1))]))
            ch = skip_ch
        self.decoder_ch = ch
        if self.separate_plane_heads:
            self.local_heads = nn.ModuleList([ME.MinkowskiConvolution(self.decoder_ch, 1, kernel_size=1, stride=1, dimension=self.dimension, bias=False) for _ in range(self.num_views)])
        else:
            self.local_head = ME.MinkowskiConvolution(self.decoder_ch, 1, kernel_size=1, stride=1, dimension=self.dimension, bias=False)
        self.evidence_sum = ME.MinkowskiGlobalSumPooling()
        self.embed_pool = ME.MinkowskiGlobalAvgPooling()
        self.embed_proj = nn.Linear(self.decoder_ch, self.embed_dim)
        self.plane_scale = nn.Parameter(torch.ones(self.num_views))
        self.plane_bias = nn.Parameter(torch.zeros(self.num_views))
        self.correction = self.ZeroInitCorrection(self.embed_dim) if use_correction else None

    def _uresnet_features(self, x: ME.SparseTensor) -> ME.SparseTensor:
        x = self.stem(x)
        skips = []
        for i, blk in enumerate(self.enc_blocks):
            x = blk(x)
            skips.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.dec_blocks, reversed(skips[:-1])):
            x = up(x)
            x = ME.cat(x, skip)
            x = dec(x)
        return x

    def _local_head_for_plane(self, plane_idx: int) -> nn.Module:
        if self.separate_plane_heads:
            return self.local_heads[plane_idx]
        return self.local_head

    def _forward_one_plane(self, x: ME.SparseTensor, plane_idx: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, ME.SparseTensor]:
        feat = self._uresnet_features(x)
        a = self._local_head_for_plane(plane_idx)(feat)
        e_sparse = self.evidence_sum(a)
        e = self._global_sparse_to_dense(e_sparse, batch_size)
        e = self.plane_scale[plane_idx] * e + self.plane_bias[plane_idx]
        z_sparse = self.embed_pool(feat)
        z = self._global_sparse_to_dense(z_sparse, batch_size)
        z = self.embed_proj(z)
        return (z, e, a)

    def forward(self, inputs: Dict[str, ME.SparseTensor], available_mask: Optional[torch.Tensor]=None, return_parts: bool=False, return_maps: bool=False) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, ME.SparseTensor]]]]:
        if not self.plane_names:
            raise ValueError('plane_names must be non-empty')
        first_name = self.plane_names[0]
        if first_name not in inputs:
            raise KeyError(f'missing plane {first_name!r} in inputs')
        if available_mask is not None:
            batch_size = int(available_mask.size(0))
            mask = available_mask.to(dtype=torch.bool, device=next(self.parameters()).device)
            if mask.ndim != 2 or mask.size(1) != self.num_views:
                raise ValueError(f'available_mask must be [B, {self.num_views}]')
        else:
            batch_size = self._infer_num_batches_from_sparse(inputs[first_name])
            mask = None
        z_list = []
        e_list = []
        local_maps = {}
        for pid, name in enumerate(self.plane_names):
            if name not in inputs:
                raise KeyError(f'missing plane {name!r} in inputs; got keys={tuple(inputs.keys())}')
            z, e, a = self._forward_one_plane(inputs[name], plane_idx=pid, batch_size=batch_size)
            z_list.append(z)
            e_list.append(e)
            if return_maps:
                local_maps[name] = a
        Z = torch.stack(z_list, dim=1)
        E = torch.stack(e_list, dim=1).squeeze(-1)
        if mask is not None:
            mf = mask.to(dtype=E.dtype)
            event_logit = (E * mf).sum(dim=1, keepdim=True)
            denom = mf.sum(dim=1, keepdim=True).clamp_min(1.0)
            z_bar = (Z * mf.unsqueeze(-1)).sum(dim=1) / denom
        else:
            event_logit = E.sum(dim=1, keepdim=True)
            z_bar = Z.mean(dim=1)
        if self.correction is not None:
            event_logit = event_logit + self.correction(z_bar)
        if return_parts:
            out: Dict[str, Union[torch.Tensor, Dict[str, ME.SparseTensor]]] = {'event_logit': event_logit, 'plane_logits': E, 'embeddings': Z}
            if return_maps:
                out['local_evidence_maps'] = local_maps
            if available_mask is not None:
                out['available_mask'] = available_mask
            return out
        return event_logit
