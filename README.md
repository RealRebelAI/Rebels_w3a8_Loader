# Rebels W3A8 Loader

ComfyUI loader nodes for quantized diffusion models that ComfyUI can't load on its own — sub-4-bit codebook formats, and GGUFs of architectures selected by file metadata.

Two nodes, both under the **RebelsW4A8** category.

---

## Why this exists

ComfyUI's native quantization registry covers `asym_w4a8_int8`, `convrot_w4a8`, `int8_tensorwise`, `nvfp4` and `mxfp8`. Two real gaps sit outside it:

**Sub-4-bit codebook weights.** The W4A8 layout packs int4 codes with a per-tensor Lloyd-Max codebook and fp8 group scales. The same scheme works at 3 and 2 bits and produces meaningfully smaller files — but the fused CUDA kernel only unpacks 4-bit nibbles, so nothing native can read them. This loader dequantizes them in pure PyTorch instead: slower than the int8 GEMM path, but it runs, which is the difference between a usable tier and a dead file.

**GGUF architecture metadata.** Some models are identified not by tensor names but by a `config` blob in the safetensors `__metadata__` — Wan-Animate-2 selects `model_type: "animate2"` that way. GGUF has no equivalent field, so a converted GGUF loads as the wrong variant: it renders, but silently ignores the conditioning that makes the model what it is. This loader supplies that value explicitly.

---

## Nodes

### W4A8 Unet Loader (Rebels)

Loads codebook-quantized `.safetensors` at **4, 3 or 2 bits**, plus ConvRot W4A4.

- Memory-mapped: an 18 GB checkpoint costs no RAM up front
- Registers with ComfyUI's dynamic-VRAM system, so weights stream and offload like any other model
- Reports true packed size so the memory manager budgets correctly
- Row-chunked dequant keeps peak VRAM near the layer size, not the tensor's worst case
- `activation_bits`: `auto` (read from the file), or force 16/8/6/4 — emulated, for quality comparison only, no speed benefit

Format is detected per tensor from the file's own config, so 4-bit and 3-bit layers can coexist in one checkpoint.

### GGUF Unet Loader + model_type (Rebels)

ComfyUI-GGUF's loader plus an explicit `model_type`.

- Merges into whatever metadata the GGUF already carries rather than replacing it
- Works with current and older ComfyUI-GGUF versions
- Logs the resolved model class on load, so you can confirm the variant actually took

Set `model_type` to `animate2` for Wan-Animate-2. Leave it blank to behave like the stock loader.

---

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/RealRebelAI/Rebels_w3a8_Loader
```

Restart ComfyUI. The GGUF node additionally requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) to be installed — it loads through that pack rather than duplicating it.

---

## Which loader do I need?

| File | Loader |
|---|---|
| w4a8 safetensors | **stock Load Diffusion Model** — native kernels, fastest |
| w4a4 safetensors | **stock Load Diffusion Model** — native |
| w3a8 / w2a8 safetensors | **W4A8 Unet Loader (Rebels)** |
| GGUF, metadata-selected architecture | **GGUF Unet Loader + model_type (Rebels)** |
| GGUF, ordinary architecture | stock Unet Loader (GGUF) |

If a format loads natively, use the native path. This pack is for what doesn't.

---

## Requirements

- ComfyUI with the native quantization registry (0.30.0+)
- PyTorch with `float8_e4m3fn` support
- ComfyUI-GGUF, for the GGUF node only
- Any CUDA GPU — the software dequant path has no compute-capability floor

---

## Notes

Sub-4-bit tiers trade quality for size, and the trade is steep. Measured weight reconstruction error against bf16, on a video DiT:

| Format | bits/weight | relL2 |
|---|---|---|
| w4a8 | 4.50 | ~7% |
| w3a8 | 3.50 | ~15% |
| w2a8 | 2.50 | ~31% |

w4a8 is the quality tier. w3a8 is usable on some architectures and not others. w2a8 exists to be measured. Test before shipping — a file that loads is not a file that works.

---

## Credits

- [city96](https://github.com/city96/ComfyUI-GGUF) — ComfyUI-GGUF
- [Kijai](https://github.com/kijai) — the W4A8 int8-codebook layout
- [Comfy-Org](https://github.com/Comfy-Org/comfy-kitchen) — comfy-kitchen and the native quant registry

Built by [RealRebelAI](https://huggingface.co/realrebelai) · [X](https://x.com/realrebelai)
