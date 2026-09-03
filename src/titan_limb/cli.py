"""Command-line entry points for repeatable Titan limb work."""

from pathlib import Path
from typing import Annotated

import typer

from titan_limb.config import DEFAULT_CONFIG_PATH, load_settings
from titan_limb.manifest import (
    ValidationStatus,
    create_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")


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
