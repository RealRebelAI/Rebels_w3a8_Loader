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
    """Packed weight that reports its ORIGINAL shape to ComfyUI and carries
    its quant metadata through torch ops (device moves, .data access, etc)."""

    _META = ("s_rel", "s_channel", "codebook", "cfg")

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

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        src = None
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, W4A8Tensor) and hasattr(a, "cfg"):
                src = a
                break
        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)

        def fix(o):
            if isinstance(o, W4A8Tensor) and not hasattr(o, "cfg") and src:
                for name in cls._META:
                    setattr(o, name, getattr(src, name))
            return o

        if isinstance(ret, torch.Tensor):
            fix(ret)
        elif isinstance(ret, (list, tuple)):
            for o in ret:
                fix(o)
        return ret

    @property
    def shape(self):
        return torch.Size(self.cfg["orig_shape"])

    def size(self, dim=None):
        s = self.shape
        return s if dim is None else s[dim]

    def numel(self):
        """TRUE packed byte count so ComfyUI's load AND unload accounting
        match reality (element_size is 1)."""
        with torch._C.DisableTorchFunctionSubclass():
            total = torch.Tensor.nelement(self)
        total += self.s_rel.nelement() * self.s_rel.element_size()
        total += self.s_channel.nelement() * self.s_channel.element_size()
        total += self.codebook.nelement() * self.codebook.element_size()
        return int(total)

    nelement = numel

    def element_size(self):
        return 1

    def new_empty(self, size, **kwargs):
        with torch._C.DisableTorchFunctionSubclass():
            return torch.empty(size, **kwargs)

    def clone(self, *a, **k):
        return self

    def detach(self, *a, **k):
        return self

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        for a in args:
            if isinstance(a, (str, torch.device)):
                device = a
        if device is None:
            return self
        with torch._C.DisableTorchFunctionSubclass():
            moved = torch.Tensor._make_subclass(torch.Tensor, self).to(device)
        return W4A8Tensor(moved, self.s_rel.to(device),
                          self.s_channel.to(device),
                          self.codebook.to(device), self.cfg)

    def __repr__(self):
        return (f"W4A8Tensor(shape={tuple(self.cfg['orig_shape'])}, "
                f"bits={self.cfg['bits']})")


def _dequant_any(w, device, dtype):
    if isinstance(w, W4A8Tensor):
        return dequantize_w4a8(
            torch.Tensor._make_subclass(torch.Tensor, w),
            w.s_rel, w.s_channel, w.codebook, w.cfg, device, dtype)
    return w.to(device=device, dtype=dtype)


def _fake_quant_act(x, bits):
    """Emulated activation quantization (per-token symmetric). No speed gain --
    this exists so activation-bit choices can be evaluated for QUALITY before
    any int kernel exists for them."""
    if bits is None or bits >= 16:
        return x
    qmax = (1 << (bits - 1)) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    return (x / scale).round().clamp(-qmax, qmax) * scale


def make_w4a8_ops(act_bits=None):
    class Ops(comfy.ops.manual_cast):
        class Linear(comfy.ops.manual_cast.Linear):
            # THE flag ComfyUI's dynamic-VRAM system looks for: with this set,
            # Comfy may keep our weights on CPU and call
            # forward_comfy_cast_weights() per forward, streaming them in.
            comfy_cast_weights = True

            def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs):
                w = state_dict.get(prefix + "weight")
                if isinstance(w, W4A8Tensor):
                    self.weight = torch.nn.Parameter(w, requires_grad=False)
                    b = state_dict.get(prefix + "bias")
                    self.bias = (torch.nn.Parameter(b.detach().clone(),
                                                    requires_grad=False)
                                 if b is not None else None)
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

            def forward_comfy_cast_weights(self, input, *args, **kwargs):
                w = self.weight
                if isinstance(w, W4A8Tensor):
                    x = _fake_quant_act(input, act_bits)
                    weight = _dequant_any(w, input.device, input.dtype)
                    bias = (self.bias.to(device=input.device,
                                         dtype=input.dtype)
                            if self.bias is not None else None)
                    out = torch.nn.functional.linear(x, weight, bias)
                    del weight
                    return out
                return super().forward_comfy_cast_weights(input, *args,
                                                          **kwargs)

    return Ops


W4A8Ops = make_w4a8_ops()


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
            "activation_bits": (["16 (none)", "8", "6", "4"],
                                {"default": "16 (none)",
                                 "tooltip": "Emulated activation quantization "
                                 "for quality testing (w4a8 / w4a6 / w4a4). "
                                 "No speed benefit -- measures quality only."}),
            "workspace_multiplier": ("FLOAT", {"default": 1.0, "min": 1.0,
                                               "max": 8.0, "step": 0.25,
                                               "tooltip": "Reserves extra VRAM "
                                               "for on-the-fly dequant. Raise "
                                               "if you OOM, lower if too much "
                                               "stays on CPU."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "RebelsW4A8"
    TITLE = "W4A8 Unet Loader (Rebels)"

    def load(self, unet_name, activation_bits="16 (none)",
             workspace_multiplier=1.0):
        ab = None if activation_bits.startswith("16") else int(activation_bits)
        path = folder_paths.get_full_path("diffusion_models", unet_name)
        sd = load_w4a8_state_dict(path)
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": make_w4a8_ops(ab),
                               "dtype": torch.float16})
        if model is not None and hasattr(model.model, "memory_usage_factor"):
            model.model.memory_usage_factor *= float(workspace_multiplier)
        if model is None:
            raise RuntimeError(
                "ComfyUI could not detect the model architecture from this "
                "file. Check that the source model is supported by your "
                "ComfyUI version.")
        return (model,)


NODE_CLASS_MAPPINGS = {"RebelsW4A8UnetLoader": RebelsW4A8UnetLoader}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelsW4A8UnetLoader": "W4A8 Unet Loader (Rebels)"}
