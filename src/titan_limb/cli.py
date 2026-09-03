"""Command-line entry points for repeatable Titan limb work."""

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from titan_limb.analysis.asymmetry import write_asymmetry_parquet
from titan_limb.analysis.seasons import write_seasonal_parquet
from titan_limb.analysis.transitions import write_transition_parquet
from titan_limb.config import DEFAULT_CONFIG_PATH, load_settings
from titan_limb.config_bands import DEFAULT_BAND_CONFIG, load_band_policy
from titan_limb.config_seasons import DEFAULT_SEASON_CONFIG, load_season_policy
from titan_limb.fitting.quality import FitQualityPolicy, audit_fit_parquet
from titan_limb.io.legacy import read_selected_fit_directory, write_selected_fit_parquet
from titan_limb.io.legacy_profiles import write_profile_directory
from titan_limb.io.observations import (
    read_selected_observations,
    write_observations_parquet,
)
from titan_limb.manifest import (
    ValidationStatus,
    create_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from titan_limb.models.core import FitStatus

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
data_app = typer.Typer(no_args_is_help=True)
fits_app = typer.Typer(no_args_is_help=True)
analyze_app = typer.Typer(no_args_is_help=True)
plot_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(fits_app, name="fits")
app.add_typer(analyze_app, name="analyze")
app.add_typer(plot_app, name="plot")


@app.command()
def status(
    config: Annotated[Path, typer.Option()] = DEFAULT_CONFIG_PATH,
) -> None:
    settings = load_settings(config)
    typer.echo(f"data_dir={settings.data_dir}")
    typer.echo(f"artifact_dir={settings.artifact_dir}")


@data_app.command("manifest")
def manifest_command(
    data_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/reports/data-manifest.json"
    ),
) -> None:
    manifest = create_manifest(data_dir)
    write_manifest(manifest, output)
    typer.echo(f"files={len(manifest.entries)}")
    typer.echo(f"output={output.resolve()}")


@data_app.command("migrate-profiles")
def migrate_profiles_command(
    source_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/processed/legacy-profiles.parquet"
    ),
) -> None:
    report = write_profile_directory(source_dir, output)
    typer.echo(f"files={report.files}")
    typer.echo(f"rows={report.rows}")
    typer.echo(f"points={report.points}")
    typer.echo(f"output={output.resolve()}")


@data_app.command("observations")
def observations_command(
    nantes_csv: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "fitting_code/ingestion/data/combined_nantes.csv"
    ),
    selection_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "settings/s3xy_cubes.json"
    ),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/processed/observations.parquet"
    ),
) -> None:
    records = read_selected_observations(nantes_csv, selection_json)
    write_observations_parquet(records, output)
    typer.echo(f"rows={len(records)}")
    typer.echo(f"output={output.resolve()}")


@fits_app.command("audit")
def audit_fits_command(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/processed/fit-quality.parquet"
    ),
    minimum_r_squared: Annotated[float | None, typer.Option()] = None,
    maximum_absolute_coefficient: Annotated[float | None, typer.Option()] = None,
) -> None:
    policy = FitQualityPolicy(
        minimum_r_squared=minimum_r_squared,
        maximum_absolute_coefficient=maximum_absolute_coefficient,
    )
    result = audit_fit_parquet(source, output, policy)
    for row in result.group_by("quality_status").len().sort("quality_status").rows():
        typer.echo(f"{row[0]}={row[1]}")
    typer.echo(f"output={output.resolve()}")


@analyze_app.command("transitions")
def analyze_transitions_command(
    fits: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/processed/legacy-selected-fits.parquet"
    ),
    quality: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/processed/fit-quality.parquet"
    ),
    observations: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/processed/observations.parquet"
    ),
    bands: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = (
        DEFAULT_BAND_CONFIG
    ),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/processed/transitions.parquet"
    ),
) -> None:
    result = write_transition_parquet(
        fits, quality, observations, output, load_band_policy(bands)
    )
    typer.echo(f"rows={result.height}")
    typer.echo(
        f"crossings={result.filter(result['crossing_index'].is_not_null()).height}"
    )
    typer.echo(f"output={output.resolve()}")


@analyze_app.command("asymmetry")
def analyze_asymmetry_command(
    fits: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/processed/legacy-selected-fits.parquet"
    ),
    quality: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/processed/fit-quality.parquet"
    ),
    observations: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/processed/observations.parquet"
    ),
    bands: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = (
        DEFAULT_BAND_CONFIG
    ),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/processed/asymmetry.parquet"
    ),
) -> None:
    result = write_asymmetry_parquet(
        fits, quality, observations, output, load_band_policy(bands)
    )
    typer.echo(f"rows={result.height}")
    typer.echo(f"output={output.resolve()}")


@analyze_app.command("seasons")
def analyze_seasons_command(
    asymmetry: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/processed/asymmetry.parquet"
    ),
    seasons: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = (
        DEFAULT_SEASON_CONFIG
    ),
    cube_output: Annotated[Path, typer.Option()] = Path(
        "artifacts/processed/seasonal-cubes.parquet"
    ),
    summary_output: Annotated[Path, typer.Option()] = Path(
        "artifacts/processed/seasonal-summary.parquet"
    ),
) -> None:
    cube_table, summary = write_seasonal_parquet(
        asymmetry,
        cube_output,
        summary_output,
        load_season_policy(seasons),
    )
    typer.echo(f"cube_rows={cube_table.height}")
    typer.echo(f"summary_rows={summary.height}")
    typer.echo(f"cube_output={cube_output.resolve()}")
    typer.echo(f"summary_output={summary_output.resolve()}")


@plot_app.command("transitions")
def plot_transitions_command(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/processed/transitions.parquet"
    ),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/figures/transition-timeline.png"
    ),
) -> None:
    # Keep non-plot commands free of Matplotlib startup and cache work.
    from titan_limb.plotting.figures import plot_transition_timeline  # noqa: PLC0415

    plot_transition_timeline(pl.read_parquet(source), output)
    typer.echo(f"output={output.resolve()}")


@plot_app.command("asymmetry")
def plot_asymmetry_command(
    source: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/processed/asymmetry.parquet"
    ),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/figures/asymmetry-spectrum.png"
    ),
) -> None:
    # Keep non-plot commands free of Matplotlib startup and cache work.
    from titan_limb.plotting.figures import plot_asymmetry_spectrum  # noqa: PLC0415

    plot_asymmetry_spectrum(pl.read_parquet(source), output)
    typer.echo(f"output={output.resolve()}")


@plot_app.command("seasons")
def plot_seasons_command(
    cubes: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/processed/seasonal-cubes.parquet"
    ),
    summary: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/processed/seasonal-summary.parquet"
    ),
    seasons: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = (
        DEFAULT_SEASON_CONFIG
    ),
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/figures/seasonal-asymmetry.png"
    ),
) -> None:
    # Keep non-plot commands free of Matplotlib startup and cache work.
    from titan_limb.plotting.figures import plot_seasonal_timeline  # noqa: PLC0415

    plot_seasonal_timeline(
        pl.read_parquet(cubes),
        pl.read_parquet(summary),
        output,
        load_season_policy(seasons),
    )
    typer.echo(f"output={output.resolve()}")


@data_app.command("validate")
def validate_command(
    data_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/reports/data-manifest.json"
    ),
) -> None:
    result = validate_manifest(data_dir, read_manifest(manifest))
    typer.echo(result.model_dump_json(indent=2))
    if result.status is ValidationStatus.INVALID:
        raise typer.Exit(code=1)


@data_app.command("migrate-selected-fits")
def migrate_selected_fits_command(
    source_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()] = Path(
        "artifacts/processed/legacy-selected-fits.parquet"
    ),
) -> None:
    records = read_selected_fit_directory(source_dir)
    write_selected_fit_parquet(records, output)
    failures = sum(record.status is FitStatus.FAILED for record in records)
    typer.echo(f"rows={len(records)}")
    typer.echo(f"failed={failures}")
    typer.echo(f"output={output.resolve()}")
