"""Command-line entry points for repeatable Titan limb work."""

from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from titan_limb.analysis.asymmetry import write_asymmetry_parquet
from titan_limb.analysis.seasons import write_seasonal_parquet
from titan_limb.analysis.sensitivity import write_sensitivity_table
from titan_limb.analysis.transitions import write_transition_parquet
from titan_limb.analysis.trust import write_band_trust_parquet
from titan_limb.config import ProjectSettings, load_settings
from titan_limb.config_bands import load_band_policy
from titan_limb.config_seasons import load_season_policy
from titan_limb.fitting.quality import FitQualityPolicy, audit_fit_parquet
from titan_limb.io.legacy import read_selected_fit_directory, write_selected_fit_parquet
from titan_limb.io.legacy_profiles import write_profile_directory
from titan_limb.io.observations import (
    read_selected_observations,
    write_observations_parquet,
)
from titan_limb.io.selection import read_selected_cube_ids
from titan_limb.manifest import (
    ValidationStatus,
    create_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from titan_limb.models.core import FitStatus
from titan_limb.pipeline import build_raw_dataset
from titan_limb.simulations.srtc import write_srtc_analysis
from titan_limb.validation.raw import write_raw_validation, write_raw_validation_gate

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
data_app = typer.Typer(no_args_is_help=True)
fits_app = typer.Typer(no_args_is_help=True)
analyze_app = typer.Typer(no_args_is_help=True)
plot_app = typer.Typer(no_args_is_help=True)
simulation_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(fits_app, name="fits")
app.add_typer(analyze_app, name="analyze")
app.add_typer(plot_app, name="plot")
app.add_typer(simulation_app, name="simulate")


@app.callback()
def configure(
    context: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            help="Path to project settings. Packaged defaults are used if omitted."
        ),
    ] = None,
) -> None:
    context.obj = load_settings(config)


def _settings(context: typer.Context) -> ProjectSettings:
    settings = context.obj
    if not isinstance(settings, ProjectSettings):
        raise RuntimeError("project settings were not initialized")
    return settings


@app.command()
def status(
    context: typer.Context,
    config: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = load_settings(config) if config is not None else _settings(context)
    typer.echo(f"project_dir={settings.project_dir}")
    typer.echo(f"data_dir={settings.data_dir}")
    typer.echo(f"artifact_dir={settings.artifact_dir}")
    typer.echo(f"config_dir={settings.config_dir}")


@data_app.command("manifest")
def manifest_command(
    context: typer.Context,
    data_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    data_dir = settings.data_path(data_dir)
    output = settings.artifact_path(output, "reports", "data-manifest.json")
    manifest = create_manifest(data_dir)
    write_manifest(manifest, output)
    typer.echo(f"files={len(manifest.entries)}")
    typer.echo(f"output={output.resolve()}")


@data_app.command("migrate-profiles")
def migrate_profiles_command(
    context: typer.Context,
    source_dir: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    source_dir = settings.data_path(source_dir, "sorted_and_filtered")
    output = settings.artifact_path(output, "processed", "legacy-profiles.parquet")
    report = write_profile_directory(source_dir, output)
    typer.echo(f"files={report.files}")
    typer.echo(f"rows={report.rows}")
    typer.echo(f"points={report.points}")
    typer.echo(f"output={output.resolve()}")


@data_app.command("observations")
def observations_command(
    context: typer.Context,
    nantes_csv: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    selection_json: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    nantes_csv = settings.project_path(
        nantes_csv, "fitting_code", "ingestion", "data", "combined_nantes.csv"
    )
    selection_json = settings.project_path(
        selection_json, "settings", "s3xy_cubes.json"
    )
    output = settings.artifact_path(output, "processed", "observations.parquet")
    records = read_selected_observations(nantes_csv, selection_json)
    write_observations_parquet(records, output)
    typer.echo(f"rows={len(records)}")
    typer.echo(f"output={output.resolve()}")


@data_app.command("build-raw")
def build_raw_command(
    context: typer.Context,
    cubes_dir: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    selection_json: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    output_dir: Annotated[Path | None, typer.Option()] = None,
    minimum_emission_degrees: Annotated[float, typer.Option()] = 25.0,
    resume: Annotated[bool, typer.Option()] = True,
) -> None:
    settings = _settings(context)
    cubes_dir = settings.data_path(cubes_dir, "original_cubes")
    selection_json = settings.project_path(
        selection_json, "settings", "s3xy_cubes.json"
    )
    output_dir = settings.artifact_path(output_dir, "raw")
    report = build_raw_dataset(
        cubes_dir,
        read_selected_cube_ids(selection_json),
        output_dir,
        minimum_emission_degrees=minimum_emission_degrees,
        resume=resume,
    )
    for key, value in asdict(report).items():
        typer.echo(f"{key}={value}")
    typer.echo(f"output={output_dir.resolve()}")


@data_app.command("validate-raw")
def validate_raw_command(
    context: typer.Context,
    profiles: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    fits: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    legacy_dir: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    output_dir: Annotated[Path | None, typer.Option()] = None,
    maximum_changed_profiles: Annotated[int | None, typer.Option(min=0)] = None,
    maximum_u1_drift: Annotated[float | None, typer.Option(min=0.0)] = None,
    maximum_u2_drift: Annotated[float | None, typer.Option(min=0.0)] = None,
) -> None:
    settings = _settings(context)
    profiles = settings.artifact_path(profiles, "raw", "profiles.parquet")
    fits = settings.artifact_path(fits, "raw", "fits.parquet")
    legacy_dir = settings.data_path(legacy_dir, "selected_fits")
    output_dir = settings.artifact_path(output_dir, "reports")
    summary = write_raw_validation(profiles, fits, legacy_dir, output_dir)
    gate = write_raw_validation_gate(
        summary,
        output_dir / "raw-validation-gate.json",
        maximum_changed_profiles=maximum_changed_profiles,
        maximum_u1_drift=maximum_u1_drift,
        maximum_u2_drift=maximum_u2_drift,
    )
    for key, value in asdict(summary).items():
        typer.echo(f"{key}={value}")
    typer.echo(f"gate={gate.status}")
    typer.echo(f"output={output_dir.resolve()}")
    if not gate.passed:
        raise typer.Exit(code=1)


@simulation_app.command("srtc")
def simulate_srtc_command(
    context: typer.Context,
    source: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    image_dir: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    output_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    source = settings.data_path(source, "Titan SRTC++ Analysis.csv")
    image_dir = settings.data_path(image_dir, "SRTC++", "v1+v2")
    output_dir = settings.artifact_path(output_dir, "simulations")
    metrics, importance = write_srtc_analysis(source, image_dir, output_dir)
    for row in metrics.iter_rows(named=True):
        typer.echo(f"{row['model']}_r_squared={row['r_squared']}")
    typer.echo(f"top_feature={importance.item(0, 'feature')}")
    typer.echo(f"output={output_dir.resolve()}")


@fits_app.command("audit")
def audit_fits_command(
    context: typer.Context,
    source: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path | None, typer.Option()] = None,
    minimum_r_squared: Annotated[float | None, typer.Option()] = None,
    maximum_absolute_coefficient: Annotated[float | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    source = settings.artifact_path(source, "processed", "legacy-selected-fits.parquet")
    output = settings.artifact_path(output, "processed", "fit-quality.parquet")
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
    context: typer.Context,
    fits: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    quality: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    observations: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    bands: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    fits = settings.artifact_path(fits, "processed", "legacy-selected-fits.parquet")
    quality = settings.artifact_path(quality, "processed", "fit-quality.parquet")
    observations = settings.artifact_path(
        observations, "processed", "observations.parquet"
    )
    bands = settings.config_path(bands, "bands.toml")
    output = settings.artifact_path(output, "processed", "transitions.parquet")
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
    context: typer.Context,
    fits: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    quality: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    observations: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    bands: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    fits = settings.artifact_path(fits, "processed", "legacy-selected-fits.parquet")
    quality = settings.artifact_path(quality, "processed", "fit-quality.parquet")
    observations = settings.artifact_path(
        observations, "processed", "observations.parquet"
    )
    bands = settings.config_path(bands, "bands.toml")
    output = settings.artifact_path(output, "processed", "asymmetry.parquet")
    result = write_asymmetry_parquet(
        fits, quality, observations, output, load_band_policy(bands)
    )
    typer.echo(f"rows={result.height}")
    typer.echo(f"output={output.resolve()}")


@analyze_app.command("seasons")
def analyze_seasons_command(
    context: typer.Context,
    asymmetry: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    seasons: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    cube_output: Annotated[Path | None, typer.Option()] = None,
    summary_output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    asymmetry = settings.artifact_path(asymmetry, "processed", "asymmetry.parquet")
    seasons = settings.config_path(seasons, "seasons.toml")
    cube_output = settings.artifact_path(
        cube_output, "processed", "seasonal-cubes.parquet"
    )
    summary_output = settings.artifact_path(
        summary_output, "processed", "seasonal-summary.parquet"
    )
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


@analyze_app.command("trust")
def analyze_trust_command(
    context: typer.Context,
    fits: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    quality: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    observations: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    fits = settings.artifact_path(fits, "raw", "fits.parquet")
    quality = settings.artifact_path(quality, "raw", "fit-quality.parquet")
    observations = settings.artifact_path(
        observations, "processed", "observations.parquet"
    )
    output = settings.artifact_path(output, "processed", "band-trust.parquet")
    result = write_band_trust_parquet(fits, quality, observations, output)
    typer.echo(f"rows={result.height}")
    typer.echo(f"trusted={result.filter(pl.col('trusted')).height}")
    typer.echo(f"output={output.resolve()}")


@analyze_app.command("sensitivity")
def analyze_sensitivity_command(
    context: typer.Context,
    profiles: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    observations: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    bands: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    seasons: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    profiles = settings.artifact_path(profiles, "raw", "sorted-profiles.parquet")
    observations = settings.artifact_path(
        observations, "processed", "observations.parquet"
    )
    bands = settings.config_path(bands, "bands.toml")
    seasons = settings.config_path(seasons, "seasons.toml")
    output = settings.artifact_path(output, "processed", "sensitivity.parquet")
    result = write_sensitivity_table(
        profiles,
        observations,
        output,
        load_band_policy(bands),
        load_season_policy(seasons),
    )
    typer.echo(f"rows={result.height}")
    typer.echo(f"output={output.resolve()}")


@plot_app.command("transitions")
def plot_transitions_command(
    context: typer.Context,
    source: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    source = settings.artifact_path(source, "processed", "transitions.parquet")
    output = settings.artifact_path(output, "figures", "transition-timeline.png")
    # Keep non-plot commands free of Matplotlib startup and cache work.
    from titan_limb.plotting.figures import plot_transition_timeline  # noqa: PLC0415

    plot_transition_timeline(pl.read_parquet(source), output)
    typer.echo(f"output={output.resolve()}")


@plot_app.command("asymmetry")
def plot_asymmetry_command(
    context: typer.Context,
    source: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    source = settings.artifact_path(source, "processed", "asymmetry.parquet")
    output = settings.artifact_path(output, "figures", "asymmetry-spectrum.png")
    # Keep non-plot commands free of Matplotlib startup and cache work.
    from titan_limb.plotting.figures import plot_asymmetry_spectrum  # noqa: PLC0415

    plot_asymmetry_spectrum(pl.read_parquet(source), output)
    typer.echo(f"output={output.resolve()}")


@plot_app.command("seasons")
def plot_seasons_command(
    context: typer.Context,
    cubes: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    summary: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    seasons: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    cubes = settings.artifact_path(cubes, "processed", "seasonal-cubes.parquet")
    summary = settings.artifact_path(summary, "processed", "seasonal-summary.parquet")
    seasons = settings.config_path(seasons, "seasons.toml")
    output = settings.artifact_path(output, "figures", "seasonal-asymmetry.png")
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
    context: typer.Context,
    data_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    manifest: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    settings = _settings(context)
    data_dir = settings.data_path(data_dir)
    manifest = settings.artifact_path(manifest, "reports", "data-manifest.json")
    result = validate_manifest(data_dir, read_manifest(manifest))
    typer.echo(result.model_dump_json(indent=2))
    if result.status is ValidationStatus.INVALID:
        raise typer.Exit(code=1)


@data_app.command("migrate-selected-fits")
def migrate_selected_fits_command(
    context: typer.Context,
    source_dir: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = _settings(context)
    source_dir = settings.data_path(source_dir, "selected_fits")
    output = settings.artifact_path(output, "processed", "legacy-selected-fits.parquet")
    records = read_selected_fit_directory(source_dir)
    write_selected_fit_parquet(records, output)
    failures = sum(record.status is FitStatus.FAILED for record in records)
    typer.echo(f"rows={len(records)}")
    typer.echo(f"failed={failures}")
    typer.echo(f"output={output.resolve()}")
