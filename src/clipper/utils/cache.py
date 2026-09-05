"""Stage-Caching: jede Stufe schreibt ihr Ergebnis als JSON und ueberspringt sich,
wenn die Datei schon existiert. Spart bei Re-Runs den teuren Whisper-Durchlauf."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def cached_json(path: Path, force: bool, produce: Callable[[], Any]) -> Any:
    """Gibt geparstes JSON aus `path` zurueck oder ruft `produce` und schreibt das Ergebnis."""
    if path.exists() and not force:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    result = produce()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return result


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
