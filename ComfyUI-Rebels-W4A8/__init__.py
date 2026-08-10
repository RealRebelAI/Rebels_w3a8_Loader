"""Rebels W4A8 / GGUF loaders.

Loads nodes.py by absolute path rather than a relative import, so the pack
registers correctly no matter how ComfyUI (or a diagnostic) imports it.
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "rebels_w4a8_nodes", os.path.join(_here, "nodes.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

NODE_CLASS_MAPPINGS = _mod.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _mod.NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
