"""Build selected radial profiles directly from a VIMS cube pair."""

from pathlib import Path

import numpy as np
import polars as pl

from titan_limb.io.vims import CubePairPaths, VimsCube, load_cube_pair
from titan_limb.models.core import Channel, Hemisphere
from titan_limb.processing.destripe import destripe_visible
from titan_limb.processing.geometry import distance_from_center, radial_line_indices
from titan_limb.processing.orientation import derive_transect_geometry
from titan_limb.processing.profiles import extract_profile, sort_and_filter_profile

SELECTED_PROFILE_SCHEMA = {
    "cube_id": pl.String,
    "band": pl.Int64,
    "wavelength_um": pl.Float64,
    "channel": pl.String,
    "hemisphere": pl.String,
    "slant_angle_degrees": pl.Int64,
    "actual_angle_degrees": pl.Float64,
    "north_orientation_degrees": pl.Float64,
    "illumination_degrees": pl.Float64,
    "center_row": pl.Float64,
    "center_column": pl.Float64,
    "filtered": pl.Boolean,
    "fit_filtered": pl.Boolean,
    "minimum_fit_emission_degrees": pl.Float64,
    "pixel_rows": pl.List(pl.Int64),
    "pixel_columns": pl.List(pl.Int64),
    "pixel_distances": pl.List(pl.Float64),
    "emission_angles": pl.List(pl.Float64),
    "brightness": pl.List(pl.Float64),
}


def _channel_profiles(
    cube: VimsCube,
    channel: Channel,
    incidence: np.ndarray,
) -> pl.DataFrame:
    geometry = derive_transect_geometry(cube.eme, cube.lat, incidence)
    cube_id = cube.img_id if cube.img_id.startswith("C") else f"C{cube.img_id}"
    distances = distance_from_center(cube.eme.shape, geometry.center.pixel)
    rows: list[dict[str, object]] = []
    sides = (
        (Hemisphere.NORTH, geometry.north_slant_degrees),
        (Hemisphere.SOUTH, geometry.south_slant_degrees),
    )
    for band, wavelength_um in zip(cube.bands, cube.wvlns, strict=True):
        band_number = int(band)
        image = np.asarray(cube[band_number])
        if channel is Channel.VISIBLE:
            image = destripe_visible(image, cube.ground)
        for hemisphere, slant in sides:
            actual_angle = geometry.north_orientation_degrees + slant
            indices = radial_line_indices(
                image.shape,
                geometry.center.subpixel,
                actual_angle,
            )
            raw_profile = extract_profile(image, cube.eme, distances, indices)
            profile = sort_and_filter_profile(raw_profile)
            rows.append(
                {
                    "cube_id": cube_id,
                    "band": band_number,
                    "wavelength_um": float(wavelength_um),
                    "channel": channel.value,
                    "hemisphere": hemisphere.value,
                    "slant_angle_degrees": slant,
                    "actual_angle_degrees": actual_angle,
                    "north_orientation_degrees": geometry.north_orientation_degrees,
                    "illumination_degrees": geometry.illumination_degrees,
                    "center_row": geometry.center.subpixel[0],
                    "center_column": geometry.center.subpixel[1],
                    "filtered": len(profile.emission_angles)
                    != len(raw_profile.emission_angles),
                    "fit_filtered": False,
                    "minimum_fit_emission_degrees": None,
                    "pixel_rows": profile.pixel_indices[:, 0].tolist(),
                    "pixel_columns": profile.pixel_indices[:, 1].tolist(),
                    "pixel_distances": profile.pixel_distances.tolist(),
                    "emission_angles": profile.emission_angles.tolist(),
                    "brightness": profile.brightness.tolist(),
                }
            )
    return pl.DataFrame(rows, schema=SELECTED_PROFILE_SCHEMA)


def build_selected_profiles(
    visible: VimsCube,
    infrared: VimsCube,
    *,
    legacy_ir_incidence_source: bool = True,
) -> pl.DataFrame:
    """Build two selected profiles per band for one cube pair."""
    infrared_incidence = visible.inc if legacy_ir_incidence_source else infrared.inc
    return pl.concat(
        [
            _channel_profiles(visible, Channel.VISIBLE, visible.inc),
            _channel_profiles(infrared, Channel.INFRARED, infrared_incidence),
        ]
    ).sort("band", "hemisphere")


def build_selected_profiles_from_paths(
    paths: CubePairPaths,
    *,
    legacy_ir_incidence_source: bool = True,
) -> pl.DataFrame:
    visible, infrared = load_cube_pair(paths)
    return build_selected_profiles(
        visible,
        infrared,
        legacy_ir_incidence_source=legacy_ir_incidence_source,
    )


def write_selected_profiles_from_paths(
    paths: CubePairPaths,
    output: Path,
    *,
    legacy_ir_incidence_source: bool = True,
) -> pl.DataFrame:
    result = build_selected_profiles_from_paths(
        paths,
        legacy_ir_incidence_source=legacy_ir_incidence_source,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="zstd", statistics=True)
    return result
