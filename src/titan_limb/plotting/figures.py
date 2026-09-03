"""Deterministic figures for the first global analysis outputs."""

from pathlib import Path

import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from titan_limb.config_seasons import SeasonPolicy

NORTH_COLOR = "#0072B2"
SOUTH_COLOR = "#D55E00"
VISIBLE_COLOR = "#0072B2"
INFRARED_COLOR = "#D55E00"
FIGURE_DPI = 180


def _new_figure() -> tuple[Figure, Axes]:
    figure = Figure(figsize=(9.0, 4.8))
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def _save_figure(figure: Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")


def plot_transition_timeline(transitions: pl.DataFrame, output: Path) -> None:
    """Plot every retained transition crossing against observation time."""
    plot_data = (
        transitions.filter(pl.col("wavelength_um").is_not_null())
        .with_columns(
            pl.when(pl.col("crossing_count") > 1)
            .then(pl.lit("multiple crossings"))
            .otherwise(pl.lit("single crossing"))
            .alias("crossing_case")
        )
        .sort("mid_time", "hemisphere", "crossing_index")
    )
    if plot_data.is_empty():
        raise ValueError("transition table contains no crossings")

    with sns.axes_style("whitegrid"), sns.plotting_context("paper", font_scale=1.1):
        figure, axis = _new_figure()
        sns.scatterplot(
            data=plot_data.to_dict(as_series=False),
            x="mid_time",
            y="wavelength_um",
            hue="hemisphere",
            style="crossing_case",
            hue_order=["north", "south"],
            style_order=["single crossing", "multiple crossings"],
            palette={"north": NORTH_COLOR, "south": SOUTH_COLOR},
            markers={"single crossing": "o", "multiple crossings": "X"},
            s=58,
            edgecolor="white",
            linewidth=0.6,
            ax=axis,
        )
        axis.set(
            title="Titan limb transition wavelengths",
            xlabel="Observation date (UTC)",
            ylabel="Crossing wavelength (µm)",
        )
        axis.legend(title=None, frameon=True, ncols=2)
        figure.autofmt_xdate(rotation=0, ha="center")
        figure.tight_layout()
    _save_figure(figure, output)


def summarize_asymmetry_by_wavelength(asymmetry: pl.DataFrame) -> pl.DataFrame:
    """Calculate the bandwise median and middle half across observations."""
    return (
        asymmetry.group_by("channel", "band")
        .agg(
            pl.col("wavelength_um").median().alias("wavelength_um"),
            pl.len().alias("observation_count"),
            pl.col("u_sum_difference").median().alias("median_difference"),
            pl.col("u_sum_difference")
            .quantile(0.25, interpolation="linear")
            .alias("lower_quartile"),
            pl.col("u_sum_difference")
            .quantile(0.75, interpolation="linear")
            .alias("upper_quartile"),
        )
        .sort("channel", "band")
    )


def plot_asymmetry_spectrum(asymmetry: pl.DataFrame, output: Path) -> pl.DataFrame:
    """Plot median north-minus-south coefficients with their middle half."""
    summary = summarize_asymmetry_by_wavelength(asymmetry)
    if summary.is_empty():
        raise ValueError("asymmetry table contains no paired rows")

    palette = {"visible": VISIBLE_COLOR, "infrared": INFRARED_COLOR}
    with sns.axes_style("whitegrid"), sns.plotting_context("paper", font_scale=1.1):
        figure, axis = _new_figure()
        sns.lineplot(
            data=summary.to_dict(as_series=False),
            x="wavelength_um",
            y="median_difference",
            hue="channel",
            hue_order=["visible", "infrared"],
            palette=palette,
            estimator=None,
            linewidth=1.8,
            ax=axis,
        )
        for channel, color in palette.items():
            channel_data = summary.filter(pl.col("channel") == channel)
            if channel_data.is_empty():
                continue
            axis.fill_between(
                channel_data.get_column("wavelength_um").to_numpy(),
                channel_data.get_column("lower_quartile").to_numpy(),
                channel_data.get_column("upper_quartile").to_numpy(),
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        axis.axhline(0, color="#333333", linewidth=0.9, linestyle="--")
        axis.set(
            title="North-south limb coefficient difference",
            xlabel="Wavelength (µm)",
            ylabel=r"North minus south, $u_1 + u_2$",
        )
        axis.legend(title="Channel", frameon=True)
        figure.tight_layout()
    _save_figure(figure, output)
    return summary


def plot_seasonal_timeline(
    cube_table: pl.DataFrame,
    summary: pl.DataFrame,
    output: Path,
    policy: SeasonPolicy,
) -> None:
    """Plot observation-level asymmetry with phase median intervals."""
    if cube_table.is_empty():
        raise ValueError("seasonal cube table contains no observations")
    palette = {"visible": VISIBLE_COLOR, "infrared": INFRARED_COLOR}
    season_spans = (
        ("northern winter", None, policy.northern_vernal_equinox),
        (
            "northern spring",
            policy.northern_vernal_equinox,
            policy.northern_summer_solstice,
        ),
        ("northern summer", policy.northern_summer_solstice, None),
    )
    date_min = cube_table.get_column("mid_time").min()
    date_max = cube_table.get_column("mid_time").max()
    if date_min is None or date_max is None:
        raise ValueError("seasonal cube table has no valid observation times")

    with sns.axes_style("whitegrid"), sns.plotting_context("paper", font_scale=1.1):
        figure = Figure(figsize=(9.0, 7.0))
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, 1, sharex=True)
        for axis, channel in zip(axes, ("visible", "infrared"), strict=True):
            channel_data = cube_table.filter(pl.col("channel") == channel)
            sns.scatterplot(
                data=channel_data.to_dict(as_series=False),
                x="mid_time",
                y="median_u_sum_difference",
                color=palette[channel],
                s=42,
                edgecolor="white",
                linewidth=0.5,
                ax=axis,
            )
            axis.axhline(0, color="#333333", linewidth=0.9, linestyle="--")
            for season, configured_start, configured_end in season_spans:
                start = configured_start or date_min
                end = configured_end or date_max
                result = summary.filter(
                    (pl.col("channel") == channel)
                    & (pl.col("northern_season") == season)
                ).row(0, named=True)
                if result["interval_available"]:
                    axis.fill_between(
                        [start, end],
                        result["bootstrap_lower"],
                        result["bootstrap_upper"],
                        color=palette[channel],
                        alpha=0.12,
                        linewidth=0,
                    )
                    axis.hlines(
                        result["median_u_sum_difference"],
                        start,
                        end,
                        color=palette[channel],
                        linewidth=1.5,
                    )
            axis.axvline(
                policy.northern_vernal_equinox,
                color="#555555",
                linewidth=0.9,
                linestyle=":",
            )
            axis.axvline(
                policy.northern_summer_solstice,
                color="#555555",
                linewidth=0.9,
                linestyle=":",
            )
            axis.set(
                title=channel.capitalize(),
                xlabel="",
                ylabel=r"Median north minus south, $u_1 + u_2$",
            )
        axes[-1].set_xlabel("Observation date (UTC)")
        axes[0].text(
            policy.northern_vernal_equinox,
            0.02,
            "N vernal equinox",
            transform=axes[0].get_xaxis_transform(),
            rotation=90,
            horizontalalignment="right",
            verticalalignment="bottom",
            color="#555555",
        )
        axes[0].text(
            policy.northern_summer_solstice,
            0.02,
            "N summer solstice",
            transform=axes[0].get_xaxis_transform(),
            rotation=90,
            horizontalalignment="right",
            verticalalignment="bottom",
            color="#555555",
        )
        figure.suptitle("Titan limb asymmetry across northern seasons")
        figure.autofmt_xdate(rotation=0, ha="center")
        figure.tight_layout()
    _save_figure(figure, output)
