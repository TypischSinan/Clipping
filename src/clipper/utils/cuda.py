"""CUDA-DLLs auf Windows auffindbar machen.

Die pip-Pakete nvidia-cublas-cu12 / nvidia-cudnn-cu12 legen ihre DLLs unter
site-packages/nvidia/*/bin ab. Unter Linux findet der Loader sie ueber RPATH,
unter Windows nicht - dort muss das Verzeichnis explizit registriert werden,
sonst scheitert CTranslate2 mit "cublas64_12.dll is not found".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_registered = False


def register_cuda_dlls() -> list[Path]:
    """Registriert die NVIDIA-DLL-Verzeichnisse. Idempotent, no-op ausserhalb Windows."""
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

    # CTranslate2 laedt cuBLAS ueber LoadLibraryA, und das durchsucht nur PATH -
    # add_dll_directory allein reicht dafuer nicht aus.
    if added:
        prefix = os.pathsep.join(str(p) for p in added)
        os.environ["PATH"] = prefix + os.pathsep + os.environ.get("PATH", "")

    _registered = True
    return added
