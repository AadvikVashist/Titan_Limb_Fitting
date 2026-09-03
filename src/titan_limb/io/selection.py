"""Read the chosen VIMS cube IDs."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast


def read_selected_cube_ids(path: Path) -> tuple[str, ...]:
    data = cast(Mapping[str, Mapping[str, str]], json.loads(path.read_text()))
    cube_ids = {
        value if value.startswith("C") else f"C{value}"
        for value in data["selected"].values()
    }
    return tuple(sorted(cube_ids))
