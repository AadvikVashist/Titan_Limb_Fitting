set shell := ["bash", "-cu"]

sync:
    UV_CACHE_DIR=.uv-cache uv sync

format:
    UV_CACHE_DIR=.uv-cache uv run ruff format src tests
    UV_CACHE_DIR=.uv-cache uv run ruff check --fix src tests

format-check:
    UV_CACHE_DIR=.uv-cache uv run ruff format --check src tests

lint:
    UV_CACHE_DIR=.uv-cache uv run ruff check src tests

typecheck:
    UV_CACHE_DIR=.uv-cache uv run ty check src tests

test:
    UV_CACHE_DIR=.uv-cache uv run pytest --cov=titan_limb --cov-report=term-missing

check: format-check lint typecheck test

manifest data_dir output="artifacts/reports/data-manifest.json":
    UV_CACHE_DIR=.uv-cache uv run titan-limb data manifest --data-dir "{{data_dir}}" --output "{{output}}"

validate-manifest data_dir manifest="artifacts/reports/data-manifest.json":
    UV_CACHE_DIR=.uv-cache uv run titan-limb data validate --data-dir "{{data_dir}}" --manifest "{{manifest}}"

migrate-selected-fits source_dir output="artifacts/processed/legacy-selected-fits.parquet":
    MPLCONFIGDIR=.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run titan-limb data migrate-selected-fits --source-dir "{{source_dir}}" --output "{{output}}"

migrate-profiles source_dir output="artifacts/processed/legacy-profiles.parquet":
    MPLCONFIGDIR=.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run titan-limb data migrate-profiles --source-dir "{{source_dir}}" --output "{{output}}"

observations nantes_csv="fitting_code/ingestion/data/combined_nantes.csv" selection_json="settings/s3xy_cubes.json" output="artifacts/processed/observations.parquet":
    UV_CACHE_DIR=.uv-cache uv run titan-limb data observations --nantes-csv "{{nantes_csv}}" --selection-json "{{selection_json}}" --output "{{output}}"

reference-test source_dir:
    TITAN_LEGACY_SELECTED_DIR="{{source_dir}}" MPLCONFIGDIR=.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest -m real_data

build-raw cubes_dir selection_json="settings/s3xy_cubes.json" output_dir="artifacts/raw":
    MPLCONFIGDIR=.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run titan-limb data build-raw --cubes-dir "{{cubes_dir}}" --selection-json "{{selection_json}}" --output-dir "{{output_dir}}"

validate-raw legacy_dir profiles="artifacts/raw/profiles.parquet" fits="artifacts/raw/fits.parquet":
    MPLCONFIGDIR=.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run titan-limb data validate-raw --profiles "{{profiles}}" --fits "{{fits}}" --legacy-dir "{{legacy_dir}}"

sensitivity profiles="artifacts/raw/sorted-profiles.parquet" observations="artifacts/processed/observations.parquet":
    MPLCONFIGDIR=.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run titan-limb analyze sensitivity --profiles "{{profiles}}" --observations "{{observations}}"

srtc source image_dir:
    MPLCONFIGDIR=.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run titan-limb simulate srtc --source "{{source}}" --image-dir "{{image_dir}}"

audit-fits source="artifacts/processed/legacy-selected-fits.parquet" output="artifacts/processed/fit-quality.parquet":
    UV_CACHE_DIR=.uv-cache uv run titan-limb fits audit --source "{{source}}" --output "{{output}}"

analyze-transitions fits="artifacts/processed/legacy-selected-fits.parquet" quality="artifacts/processed/fit-quality.parquet" observations="artifacts/processed/observations.parquet" output="artifacts/processed/transitions.parquet":
    UV_CACHE_DIR=.uv-cache uv run titan-limb analyze transitions --fits "{{fits}}" --quality "{{quality}}" --observations "{{observations}}" --bands configs/bands.toml --output "{{output}}"

analyze-asymmetry fits="artifacts/processed/legacy-selected-fits.parquet" quality="artifacts/processed/fit-quality.parquet" observations="artifacts/processed/observations.parquet" output="artifacts/processed/asymmetry.parquet":
    UV_CACHE_DIR=.uv-cache uv run titan-limb analyze asymmetry --fits "{{fits}}" --quality "{{quality}}" --observations "{{observations}}" --bands configs/bands.toml --output "{{output}}"

analyze-seasons asymmetry="artifacts/processed/asymmetry.parquet" cube_output="artifacts/processed/seasonal-cubes.parquet" summary_output="artifacts/processed/seasonal-summary.parquet":
    UV_CACHE_DIR=.uv-cache uv run titan-limb analyze seasons --asymmetry "{{asymmetry}}" --seasons configs/seasons.toml --cube-output "{{cube_output}}" --summary-output "{{summary_output}}"

figures transitions="artifacts/processed/transitions.parquet" asymmetry="artifacts/processed/asymmetry.parquet" output_dir="artifacts/figures":
    MPLCONFIGDIR=.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run titan-limb plot transitions --source "{{transitions}}" --output "{{output_dir}}/transition-timeline.png"
    MPLCONFIGDIR=.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run titan-limb plot asymmetry --source "{{asymmetry}}" --output "{{output_dir}}/asymmetry-spectrum.png"

seasonal-figure cubes="artifacts/processed/seasonal-cubes.parquet" summary="artifacts/processed/seasonal-summary.parquet" output="artifacts/figures/seasonal-asymmetry.png":
    MPLCONFIGDIR=.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run titan-limb plot seasons --cubes "{{cubes}}" --summary "{{summary}}" --seasons configs/seasons.toml --output "{{output}}"
