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
