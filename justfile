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

reference-test source_dir:
    TITAN_LEGACY_SELECTED_DIR="{{source_dir}}" MPLCONFIGDIR=.cache/matplotlib UV_CACHE_DIR=.uv-cache uv run pytest -m real_data

audit-fits source="artifacts/processed/legacy-selected-fits.parquet" output="artifacts/processed/fit-quality.parquet":
    UV_CACHE_DIR=.uv-cache uv run titan-limb fits audit --source "{{source}}" --output "{{output}}"

analyze-transitions fits="artifacts/processed/legacy-selected-fits.parquet" quality="artifacts/processed/fit-quality.parquet" output="artifacts/processed/transitions.parquet":
    UV_CACHE_DIR=.uv-cache uv run titan-limb analyze transitions --fits "{{fits}}" --quality "{{quality}}" --bands configs/bands.toml --output "{{output}}"

analyze-asymmetry fits="artifacts/processed/legacy-selected-fits.parquet" quality="artifacts/processed/fit-quality.parquet" output="artifacts/processed/asymmetry.parquet":
    UV_CACHE_DIR=.uv-cache uv run titan-limb analyze asymmetry --fits "{{fits}}" --quality "{{quality}}" --bands configs/bands.toml --output "{{output}}"
