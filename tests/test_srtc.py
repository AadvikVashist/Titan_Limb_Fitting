"""Tests for SRTC++ data and models."""

from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from titan_limb.simulations.srtc import (
    build_srtc_mask,
    compare_srtc_results,
    fit_srtc_image,
    inventory_srtc_images,
    read_srtc_table,
    rebuild_srtc_images,
    srtc_emission_angles,
    train_srtc_models,
)

SRTC_NAME = "Aadvik0.93_0.01_1.00_0.10_0.20_0.30_0.40_p000.colorCCD.Jcube.hazecolor.tif"


def test_inventory_srtc_images_reads_six_parameters(tmp_path: Path) -> None:
    (tmp_path / SRTC_NAME).touch()

    result = inventory_srtc_images(tmp_path)

    assert result.height == 1
    assert result.item(0, "lower_haze") == 0.01
    assert result.item(0, "upper_gas") == 0.4


def test_rebuild_srtc_image_and_compare_saved_table(tmp_path: Path) -> None:
    rows, columns = np.indices((69, 69))
    radius = np.hypot(rows - 34, columns - 34)
    image = np.where(radius <= 28, 180 - radius * 2, 0).astype(np.uint8)
    path = tmp_path / SRTC_NAME
    Image.fromarray(image).save(path)

    mask = build_srtc_mask((path,))
    emission = srtc_emission_angles(mask)
    u_sum, usable = fit_srtc_image(image.astype(float), emission, mask)
    rebuilt = rebuild_srtc_images(tmp_path)
    saved = rebuilt.select(
        "lower_haze",
        "upper_haze",
        "lower_ssa",
        "upper_ssa",
        "lower_gas",
        "upper_gas",
        "u_sum",
        "usable",
    )

    assert np.isfinite(u_sum)
    assert usable is True
    assert rebuilt.height == 1
    assert compare_srtc_results(rebuilt, saved)["equal_usable_flags"] == 1


def test_read_srtc_table_normalizes_columns(tmp_path: Path) -> None:
    source = tmp_path / "srtc.csv"
    source.write_text(
        "Lower Haze,Upper Haze,Lower SSA,Upper SSA,Lower Gas,Upper Gas,"
        "Model Output µ1+µ2 Value,Darkened or Brightened,Usable\n"
        "0.1,0.2,0.3,0.4,0.5,0.6,-1.0,Brightened,yes\n"
    )

    result = read_srtc_table(source)

    assert result.item(0, "u_sum") == -1.0
    assert result.item(0, "usable") is True


def test_train_srtc_models_returns_all_metrics() -> None:
    rows = 50
    table = pl.DataFrame(
        {
            "lower_haze": [index / rows for index in range(rows)],
            "upper_haze": [index % 3 for index in range(rows)],
            "lower_ssa": [index % 5 for index in range(rows)],
            "upper_ssa": [index % 7 for index in range(rows)],
            "lower_gas": [index % 2 for index in range(rows)],
            "upper_gas": [index % 11 for index in range(rows)],
            "u_sum": [index / rows for index in range(rows)],
            "usable": [True] * rows,
        }
    )

    metrics, importance = train_srtc_models(table)

    assert metrics.height == 3
    assert importance.height == 6
