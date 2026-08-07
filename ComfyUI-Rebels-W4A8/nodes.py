"""Rebels W4A8/W3A8 loader: run codebook-quantized safetensors in ComfyUI
without comfy-kitchen. Pure-torch on-the-fly dequant (GGUF-style ops).

Reads files produced by w4a8_convert.py:
  <n>            int8   packed codes (4-bit: 2/byte, 3-bit: 8 per 3 bytes)
  <n>_s_rel      f8e4m3 per-group scale [N, K/group]
  <n>_s_channel  f32    per-channel scale [N]
  <n>_codebook   f32    Lloyd-Max levels [2**bits]
  <n>.comfy_quant json  {bits, group_size, convrot_groupsize, orig_shape}
"""
import json
import math

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.sd
import comfy.utils
import folder_paths

_HAD = {}


def _hadamard(size, device, dtype):
    key = (size, device, dtype)
    if key not in _HAD:
        h = torch.ones(1, 1, device=device, dtype=torch.float32)
        while h.shape[0] < size:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HAD[key] = (h / math.sqrt(size)).to(dtype)
    return _HAD[key]


def _unpack(packed, k, bits):
    n = packed.shape[0]
    u = packed.to(torch.int32) & 0xFF
    if bits == 4:
        out = torch.empty(n, k, dtype=torch.int32, device=packed.device)
        out[:, 0::2] = u & 0xF
        out[:, 1::2] = (u >> 4) & 0xF
        return out
    if bits == 2:
        out = torch.empty(n, k, dtype=torch.int32, device=packed.device)
        out[:, 0::4] = u & 3
        out[:, 1::4] = (u >> 2) & 3
        out[:, 2::4] = (u >> 4) & 3
        out[:, 3::4] = (u >> 6) & 3
        return out
    g = u.view(n, k // 8, 3)
    acc = g[:, :, 0] | (g[:, :, 1] << 8) | (g[:, :, 2] << 16)
    return torch.stack([(acc >> (3 * i)) & 7 for i in range(8)],
                       -1).reshape(n, k)


def dequantize_w4a8(qdata, s_rel, s_channel, codebook, cfg, device, dtype,
                    chunk_elems=16_000_000):
    """Row-chunked dequant: peak VRAM = output fp16 + one chunk of temps."""
    n, k = cfg["orig_shape"]
    bits = cfg["bits"]
    gsz = cfg["group_size"]
    cr = cfg["convrot_groupsize"]
    out = torch.empty(n, k, device=device, dtype=dtype)
    rows_per = max(1, chunk_elems // k)
    cb = codebook.to(device).float()
    h = _hadamard(cr, device, torch.float32)
    s_ch_all = s_channel.to(device).float()
    for r0 in range(0, n, rows_per):
        r1 = min(r0 + rows_per, n)
        q = _unpack(qdata[r0:r1].to(device), k, bits)
        m = r1 - r0
        groups = k // gsz
        srel = s_rel[r0:r1].to(device).float()
        grid = (cb.view(1, 1, -1) * srel.unsqueeze(-1)).round().clamp(-127, 127)
        picked = torch.gather(grid, 2, q.view(m, groups, gsz).long())
        del q
        rot = (picked * s_ch_all[r0:r1].view(m, 1, 1)).view(m, k)
        del picked
        w = (rot.view(m, k // cr, cr) @ h.T).view(m, k)
        del rot
        out[r0:r1] = w.to(dtype)
        del w
    return out


class W4A8Tensor(torch.Tensor):
    """Packed weight that reports its ORIGINAL shape to ComfyUI."""

    @staticmethod
    def __new__(cls, qdata, s_rel, s_channel, codebook, cfg):
        t = torch.Tensor._make_subclass(cls, qdata, require_grad=False)
        t.s_rel = s_rel
        t.s_channel = s_channel
        t.codebook = codebook
        t.cfg = cfg
        return t

    def __init__(self, *a, **k):
        super().__init__()

    @property
    def shape(self):
        return torch.Size(self.cfg["orig_shape"])

    def size(self, dim=None):
        s = self.shape
        return s if dim is None else s[dim]

    def numel(self):
        n, k = self.cfg["orig_shape"]
        return n * k

    def element_size(self):
        return 1

    def new_empty(self, size, **kwargs):
        return torch.empty(size, **kwargs)

    def clone(self, *a, **k):
        return self

    def detach(self, *a, **k):
        return self

    def to(self, *args, **kwargs):
        device = None
        for a in args:
            if isinstance(a, (str, torch.device)):
                device = a
        device = kwargs.get("device", device)
        if device is None:
            return self
        moved = W4A8Tensor(
            torch.Tensor._make_subclass(torch.Tensor, self).to(device),
            self.s_rel.to(device), self.s_channel.to(device),
            self.codebook.to(device), self.cfg)
        return moved

    def __repr__(self):
        return (f"W4A8Tensor(orig_shape={tuple(self.cfg['orig_shape'])}, "
                f"bits={self.cfg['bits']})")


def _dequant_any(w, device, dtype):
    if isinstance(w, W4A8Tensor):
        return dequantize_w4a8(
            torch.Tensor._make_subclass(torch.Tensor, w),
            w.s_rel, w.s_channel, w.codebook, w.cfg, device, dtype)
    return w.to(device=device, dtype=dtype)


class W4A8Ops(comfy.ops.manual_cast):
    class Linear(comfy.ops.manual_cast.Linear):
        """Packed weight registered as a real Parameter so ComfyUI's memory
        manager counts it and handles lowvram offload (city96 pattern)."""

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                  strict, missing_keys, unexpected_keys,
                                  error_msgs):
            w = state_dict.get(prefix + "weight")
            if isinstance(w, W4A8Tensor):
                self.weight = torch.nn.Parameter(w, requires_grad=False)
                b = state_dict.get(prefix + "bias")
                if b is not None:
                    self.bias = torch.nn.Parameter(
                        b.detach().clone(), requires_grad=False)
                else:
                    self.bias = None
                return
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys,
                unexpected_keys, error_msgs)

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            if isinstance(self.weight, W4A8Tensor):
                destination[prefix + "weight"] = self.weight
                if self.bias is not None:
                    destination[prefix + "bias"] = self.bias
                return
            super()._save_to_state_dict(destination, prefix, keep_vars)

        def forward(self, x, *args, **kwargs):
            w = self.weight
            if isinstance(w, W4A8Tensor):
                weight = _dequant_any(w, x.device, x.dtype)
                bias = None
                if self.bias is not None:
                    bias = self.bias.to(device=x.device, dtype=x.dtype)
                out = torch.nn.functional.linear(x, weight, bias)
                del weight
                return out
            return super().forward(x, *args, **kwargs)


def load_w4a8_state_dict(path):
    """mmap-backed load: packed tensors are views onto the file, so the
    packed model never has to fit in RAM at once."""
    from safetensors.torch import load_file
    raw = load_file(path)          # zero-copy mmap on CPU
    cfgs = {}
    for k in list(raw):
        if k.endswith(".comfy_quant"):
            base = k[:-len(".comfy_quant")]
            cfgs[base] = json.loads(
                bytes(raw[k].numpy().tolist()).decode("utf-8"))
    sd = {}
    for k, t in raw.items():
        if k.endswith((".comfy_quant", "_s_rel", "_s_channel",
                       "_codebook", "_correction")):
            continue
        if k in cfgs and t.dtype == torch.int8:
            if (k + "_correction") in raw:
                raise RuntimeError(
                    f"{k}: asymmetric (correction) tensors not supported "
                    f"by this loader yet")
            sd[k] = W4A8Tensor(t, raw[k + "_s_rel"], raw[k + "_s_channel"],
                               raw[k + "_codebook"], cfgs[k])
        else:
            sd[k] = t
    return sd


class RebelsW4A8UnetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "RebelsW4A8"
    TITLE = "W4A8 Unet Loader (Rebels)"

    def load(self, unet_name):
        path = folder_paths.get_full_path("diffusion_models", unet_name)
        sd = load_w4a8_state_dict(path)
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": W4A8Ops,
                               "dtype": torch.float16})
        if model is None:
            raise RuntimeError(
                "ComfyUI could not detect the model architecture from this "
                "file. Check that the source model is supported by your "
                "ComfyUI version.")
        return (model,)


NODE_CLASS_MAPPINGS = {"RebelsW4A8UnetLoader": RebelsW4A8UnetLoader}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelsW4A8UnetLoader": "W4A8 Unet Loader (Rebels)"}
