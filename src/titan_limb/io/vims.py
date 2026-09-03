"""Small, lazy boundary around PyVIMS cube loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from titan_limb.models.core import Channel


class VimsCube(Protocol):
    """Fields used by the processing package."""

    img_id: str
    bands: NDArray[np.integer]
    wvlns: NDArray[np.float64]
    eme: NDArray[np.float64]
    ground: NDArray[np.bool_]

    def __getitem__(self, band: int) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class CubePairPaths:
    cube_id: str
    directory: Path
    visible: Path
    infrared: Path


def find_cube_pair(cubes_dir: Path, cube_id: str) -> CubePairPaths:
    """Resolve and validate the visible and infrared files for one cube."""
    directory = cubes_dir / cube_id
    pair = CubePairPaths(
        cube_id=cube_id,
        directory=directory,
        visible=directory / f"{cube_id}_vis.cub",
        infrared=directory / f"{cube_id}_ir.cub",
    )
    missing = [path for path in (pair.visible, pair.infrared) if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"missing cube file(s): {names}")
    return pair


def load_cube(path: Path, channel: Channel) -> VimsCube:
    """Load one cube while keeping PyVIMS out of package import time."""
    from pyvims import VIMS  # noqa: PLC0415

    pyvims_channel = "vis" if channel is Channel.VISIBLE else "ir"
    return cast(VimsCube, VIMS(path.name, path.parent, channel=pyvims_channel))


def load_cube_pair(paths: CubePairPaths) -> tuple[VimsCube, VimsCube]:
    return (
        load_cube(paths.visible, Channel.VISIBLE),
        load_cube(paths.infrared, Channel.INFRARED),
    )
