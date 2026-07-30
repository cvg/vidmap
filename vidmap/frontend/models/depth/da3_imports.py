"""Import helpers for Depth Anything 3 wrappers."""

import builtins
import os
from contextlib import contextmanager


@contextmanager
def optional_xformers_disabled():
    """Force optional xformers imports to look unavailable when requested."""
    if os.environ.get("XFORMERS_DISABLED") != "1":
        yield
        return

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "xformers" or name.startswith("xformers."):
            raise ImportError("xformers disabled by XFORMERS_DISABLED=1")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = original_import
