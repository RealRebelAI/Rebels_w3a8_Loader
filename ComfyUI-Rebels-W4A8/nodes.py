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
    """REGULAR (power-of-4 Kronecker) Hadamard, matching comfy-kitchen."""
    key = (size, str(device), dtype)
    if key not in _HAD:
        if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
            raise ValueError(f"ConvRot Hadamard must be a power of 4: {size}")
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                           [1, -1, 1, 1], [-1, 1, 1, 1]],
                          dtype=torch.float32, device=device)
        h, cur = h4, 4
        while cur < size:
            h = torch.kron(h, h4)
            cur *= 4
        _HAD[key] = (h / (size ** 0.5)).to(dtype)
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
        w = (rot.view(m, k // cr, cr) @ h).view(m, k)
        del rot
        out[r0:r1] = w.to(dtype)
        del w
    return out


def dequantize_w4a4(qdata, scale, cfg, device, dtype, chunk_elems=16_000_000):
    """comfy-kitchen TensorCoreConvRotW4A4Layout dequant (row-chunked)."""
    n, k = cfg["orig_shape"]
    cr = cfg.get("convrot_groupsize", 256)
    out = torch.empty(n, k, device=device, dtype=dtype)
    rows_per = max(1, chunk_elems // k)
    h = _hadamard_p4(cr, device)
    sc = scale.to(device).float()
    for r0 in range(0, n, rows_per):
        r1 = min(r0 + rows_per, n)
        x32 = qdata[r0:r1].to(device).to(torch.int32)
        lo, hi = x32 & 0x0F, (x32 >> 4) & 0x0F
        lo = torch.where(lo >= 8, lo - 16, lo)
        hi = torch.where(hi >= 8, hi - 16, hi)
        q = torch.stack([lo, hi], -1).reshape(r1 - r0, k).float()
        w = q * sc[r0:r1].reshape(-1, 1)
        m = r1 - r0
        out[r0:r1] = torch.matmul(w.reshape(m, k // cr, cr), h).reshape(
            m, k).to(dtype)
        del x32, lo, hi, q, w
    return out


_HAD_P4 = {}


def _hadamard_p4(size, device):
    key = (size, str(device))
    if key not in _HAD_P4:
        if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
            raise ValueError(f"W4A4 Hadamard needs a power of 4, got {size}")
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                           [1, -1, 1, 1], [-1, 1, 1, 1]],
                          dtype=torch.float32, device=device)
        h, cur = h4, 4
        while cur < size:
            h = torch.kron(h, h4)
            cur *= 4
        _HAD_P4[key] = h / (size ** 0.5)
    return _HAD_P4[key]


class W4A4Tensor(torch.Tensor):
    """Packed int4 weight, convrot_w4a4 layout (single per-row scale)."""

    _META = ("scale", "cfg")

    @staticmethod
    def __new__(cls, qdata, scale, cfg):
        t = torch.Tensor._make_subclass(cls, qdata, require_grad=False)
        t.scale = scale
        t.cfg = cfg
        return t

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        src = next((a for a in list(args) + list(kwargs.values())
                    if isinstance(a, W4A4Tensor) and hasattr(a, "cfg")), None)
        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)

        def fix(o):
            if isinstance(o, W4A4Tensor) and not hasattr(o, "cfg") and src:
                for nm in cls._META:
                    setattr(o, nm, getattr(src, nm))
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
        with torch._C.DisableTorchFunctionSubclass():
            total = torch.Tensor.nelement(self)
        return int(total + self.scale.nelement() * self.scale.element_size())

    nelement = numel

    def element_size(self):
        return 1

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
        return W4A4Tensor(moved, self.scale.to(device), self.cfg)

    def __repr__(self):
        return f"W4A4Tensor(shape={tuple(self.cfg['orig_shape'])}, bits=4)"


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
    if isinstance(w, W4A4Tensor):
        return dequantize_w4a4(
            torch.Tensor._make_subclass(torch.Tensor, w),
            w.scale, w.cfg, device, dtype)
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


def make_w4a8_ops(act_bits="auto"):
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
                if isinstance(w, (W4A8Tensor, W4A4Tensor)):
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
                if isinstance(self.weight, (W4A8Tensor, W4A4Tensor)):
                    destination[prefix + "weight"] = self.weight
                    if self.bias is not None:
                        destination[prefix + "bias"] = self.bias
                    return
                super()._save_to_state_dict(destination, prefix, keep_vars)

            def forward_comfy_cast_weights(self, input, *args, **kwargs):
                w = self.weight
                if isinstance(w, (W4A8Tensor, W4A4Tensor)):
                    ab = act_bits
                    if ab == "auto":
                        ab = w.cfg.get("act_bits",
                                       4 if isinstance(w, W4A4Tensor) else 8)
                        ab = None if int(ab) >= 16 else int(ab)
                    x = _fake_quant_act(input, ab)
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


_ST_DT = {"F64": ("f8", 8), "F32": ("f4", 4), "F16": ("f2", 2),
          "BF16": ("u2", 2), "F8_E4M3": ("u1", 1), "F8_E5M2": ("u1", 1),
          "I64": ("i8", 8), "I32": ("i4", 4), "I16": ("i2", 2),
          "I8": ("i1", 1), "U8": ("u1", 1), "BOOL": ("u1", 1)}
_TORCH_VIEW = {"BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn,
               "F8_E5M2": torch.float8_e5m2}


def load_w4a8_state_dict(path):
    """TRUE memory-mapped load. Tensors are views onto the file; the OS pages
    data in on demand, so an 18 GB checkpoint costs ~no RAM up front.
    (safetensors.load_file copies the whole file -- that will OOM/crash a
    16 GB machine before the model skeleton is even built.)"""
    import json as _json
    import struct as _struct

    import numpy as np

    with open(path, "rb") as f:
        hlen = _struct.unpack("<Q", f.read(8))[0]
        header = _json.loads(f.read(hlen).decode("utf-8"))
    header.pop("__metadata__", None)
    data_start = 8 + hlen

    mm = np.memmap(path, dtype=np.uint8, mode="r")

    def view(name):
        e = header[name]
        dt, esz = _ST_DT[e["dtype"]]
        a, b = e["data_offsets"]
        shape = tuple(e["shape"])
        n = 1
        for d in shape:
            n *= d
        arr = mm[data_start + a: data_start + a + n * esz].view(dt)
        t = torch.from_numpy(arr.reshape(shape) if shape else arr)
        if e["dtype"] in _TORCH_VIEW:
            t = t.view(_TORCH_VIEW[e["dtype"]])
        return t

    # accept BOTH key forms: "<mod>.weight.comfy_quant" (our original) and
    # "<mod>.comfy_quant" (ComfyUI native contract, what w4a8_fix_native.py
    # rewrites files to). Config is keyed by the WEIGHT tensor name.
    cfgs = {}
    for k in header:
        if not k.endswith(".comfy_quant"):
            continue
        blob = bytes(view(k).numpy().tolist())
        cfg = _json.loads(blob.decode("utf-8"))
        base = k[:-len(".comfy_quant")]
        wkey = base if base.endswith(".weight") else base + ".weight"
        if wkey in header:
            cfgs[wkey] = cfg
        elif base in header:
            cfgs[base] = cfg

    sd = {}
    for k, e in header.items():
        if k.endswith((".comfy_quant", "_s_rel", "_s_channel", "_codebook",
                       "_correction")):
            continue
        if k in cfgs and e["dtype"] == "I8":
            cfg = cfgs[k]
            if (k + "_s_rel") in header and (k + "_codebook") in header:
                sd[k] = W4A8Tensor(view(k), view(k + "_s_rel"),
                                   view(k + "_s_channel"),
                                   view(k + "_codebook"), cfg)
            elif (k + "_scale") in header:
                cfg.setdefault("convrot_groupsize", 256)
                sd[k] = W4A4Tensor(view(k), view(k + "_scale"), cfg)
            else:
                raise RuntimeError(
                    f"{k}: config present but no recognised scale tensors "
                    f"(need _s_rel+_codebook for w4a8, or _scale for w4a4)")
        else:
            sd[k] = view(k)
    sd["__mmap_keepalive__"] = mm      # stripped by the loader below
    return sd


class RebelsW4A8UnetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            "activation_bits": (["auto (from file)", "16", "8", "6", "4"],
                                {"default": "auto (from file)",
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
        ab = ("auto" if activation_bits.startswith("auto")
              else (None if activation_bits == "16"
                    else int(activation_bits)))
        path = folder_paths.get_full_path("diffusion_models", unet_name)
        sd = load_w4a8_state_dict(path)
        keepalive = sd.pop("__mmap_keepalive__", None)
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": make_w4a8_ops(ab),
                               "dtype": torch.float16})
        if model is not None:
            model._w4a8_mmap = keepalive
        if model is not None and hasattr(model.model, "memory_usage_factor"):
            model.model.memory_usage_factor *= float(workspace_multiplier)
        if model is None:
            raise RuntimeError(
                "ComfyUI could not detect the model architecture from this "
                "file. Check that the source model is supported by your "
                "ComfyUI version.")
        return (model,)




# ---------------------------------------------------------------- GGUF loader
def _load_gguf_module(name):
    """Import a module out of the installed ComfyUI-GGUF pack by path."""
    import importlib.util
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for folder in os.listdir(here):
        if folder.lower().replace("_", "-") in ("comfyui-gguf",):
            path = os.path.join(here, folder, name + ".py")
            if os.path.exists(path):
                spec = importlib.util.spec_from_file_location(
                    f"_rebels_gguf_{name}", path)
                mod = importlib.util.module_from_spec(spec)
                import sys as _s
                _s.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                return mod
    raise RuntimeError(
        "ComfyUI-GGUF not found in custom_nodes -- this node loads GGUF via "
        "that pack and cannot work without it.")


class RebelsGGUFUnetLoaderMeta:
    """UnetLoaderGGUF + a model_type override.

    Some architectures (Wan-Animate-2) are selected by the checkpoint's
    safetensors __metadata__ rather than by tensor names. GGUF has no
    equivalent field, so ComfyUI loads those models as the wrong variant.
    This node passes the metadata through explicitly.
    """

    FOLDERS = ("unet_gguf", "diffusion_models", "unet")

    @classmethod
    def _files(cls):
        seen, out = set(), []
        for key in cls.FOLDERS:
            try:
                names = folder_paths.get_filename_list(key)
            except Exception:
                continue
            for n in names:
                if n.lower().endswith(".gguf") and n not in seen:
                    seen.add(n)
                    out.append(n)
        return out or ["<no .gguf found>"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (cls._files(),),
            "model_type": ("STRING", {"default": "animate2",
                                      "tooltip": "Value for "
                                      "config.transformer.model_type, e.g. "
                                      "animate2. Leave blank for none."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "RebelsW4A8"
    TITLE = "GGUF Unet Loader + model_type (Rebels)"

    def load(self, unet_name, model_type="animate2"):
        loader = _load_gguf_module("loader")
        ops_mod = _load_gguf_module("ops")
        path = None
        for key in self.FOLDERS:
            try:
                path = folder_paths.get_full_path(key, unet_name)
            except Exception:
                path = None
            if path:
                break
        if not path:
            raise RuntimeError(
                f"could not resolve '{unet_name}' in any of "
                f"{self.FOLDERS}; put the .gguf in "
                f"ComfyUI/models/diffusion_models")
        sd = loader.gguf_sd_loader(path)
        metadata = None
        if model_type:
            metadata = {"config": json.dumps(
                {"transformer": {"model_type": model_type}})}
        ops = ops_mod.GGMLOps()
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}, metadata=metadata)
        if model is None:
            raise RuntimeError(
                "ComfyUI could not detect the architecture. If this is "
                "Wan-Animate-2, its block keys may still be nested "
                "(blocks.N.block.*) -- flatten them with gguf_rename_keys.py.")
        try:
            import logging
            cfg = getattr(model.model, "model_config", None)
            logging.info(
                "[Rebels GGUF] loaded %s | metadata model_type=%s | "
                "model class=%s",
                unet_name, model_type or "(none)",
                type(model.model).__name__)
            if cfg is not None:
                logging.info("[Rebels GGUF] unet_config=%s",
                             getattr(cfg, "unet_config", None))
        except Exception:
            pass
        return (model,)


NODE_CLASS_MAPPINGS = {
    "RebelsW4A8UnetLoader": RebelsW4A8UnetLoader,
    "RebelsGGUFUnetLoaderMeta": RebelsGGUFUnetLoaderMeta,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelsW4A8UnetLoader": "W4A8 Unet Loader (Rebels)",
    "RebelsGGUFUnetLoaderMeta": "GGUF Unet Loader + model_type (Rebels)",
}
