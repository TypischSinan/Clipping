"""Make CUDA DLLs discoverable on Windows.

The pip packages nvidia-cublas-cu12 / nvidia-cudnn-cu12 place their DLLs under
site-packages/nvidia/*/bin. On Linux the loader finds them via RPATH; on Windows
it does not - the directory has to be registered explicitly, otherwise
CTranslate2 fails with "cublas64_12.dll is not found".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_registered = False


def register_cuda_dlls() -> list[Path]:
    """Register the NVIDIA DLL directories. Idempotent, no-op outside Windows."""
    global _registered
    if _registered or sys.platform != "win32":
        return []

    try:
        import nvidia
    except ImportError:
        _registered = True
        return []

    added: list[Path] = []
    for package_root in nvidia.__path__:
        for bin_dir in sorted(Path(package_root).glob("*/bin")):
            if not bin_dir.is_dir():
                continue
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass
            added.append(bin_dir)

    # CTranslate2 loads cuBLAS through LoadLibraryA, which only searches PATH -
    # add_dll_directory alone is not enough for that.
    if added:
        prefix = os.pathsep.join(str(p) for p in added)
        os.environ["PATH"] = prefix + os.pathsep + os.environ.get("PATH", "")

    _registered = True
    return added
