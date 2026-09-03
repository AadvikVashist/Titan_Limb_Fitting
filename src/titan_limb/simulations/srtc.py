"""Typed SRTC++ table ingestion and fixed-seed model checks."""

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURES = (
    "lower_haze",
    "upper_haze",
    "lower_ssa",
    "upper_ssa",
    "lower_gas",
    "upper_gas",
)
TARGET = "u_sum"
SRTC_PROFILE_POINTS = 1000
SRTC_SMOOTHING_SIGMA = 20.0
MASK_THRESHOLD = 50.0
MAXIMUM_PROFILE_EMISSION_DEGREES = 75.0
MAXIMUM_NOISE = 35.0
MINIMUM_PEAK_BRIGHTNESS = 60.0
FILE_PATTERN = re.compile(
    r"^Aadvik0\.93_(?P<values>[0-9._]+)_p000\.colorCCD\.Jcube\.hazecolor\.tif$"
)


def read_srtc_table(path: Path) -> pl.DataFrame:
    """Read the old result table into stable names and types."""
    source = pl.read_csv(path)
    renamed = source.rename(
        {
            "Lower Haze": "lower_haze",
            "Upper Haze": "upper_haze",
            "Lower SSA": "lower_ssa",
            "Upper SSA": "upper_ssa",
            "Lower Gas": "lower_gas",
            "Upper Gas": "upper_gas",
            "Model Output µ1+µ2 Value": TARGET,
            "Darkened or Brightened": "classification",
            "Usable": "usable",
        }
    )
    return renamed.with_columns(
        pl.col("usable").cast(pl.String).str.to_lowercase().eq("yes"),
        pl.col(FEATURES).cast(pl.Float64),
        pl.col(TARGET).cast(pl.Float64),
    )


def inventory_srtc_images(image_dir: Path) -> pl.DataFrame:
    """Parse the six input values encoded in each SRTC++ TIFF name."""
    rows: list[dict[str, str | float]] = []
    for path in sorted(image_dir.glob("*.tif")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        values = [float(value) for value in match.group("values").split("_")]
        if len(values) != len(FEATURES):
            continue
        rows.append(
            {"file_name": path.name, **dict(zip(FEATURES, values, strict=True))}
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def _read_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64)


def build_srtc_mask(paths: tuple[Path, ...]) -> np.ndarray:
    """Build the shared disk mask from the mean synthetic image."""
    if not paths:
        raise ValueError("no SRTC++ TIFF images found")
    total = np.zeros_like(_read_grayscale(paths[0]))
    for path in paths:
        total += _read_grayscale(path)
    return total / len(paths) > MASK_THRESHOLD


def srtc_emission_angles(mask: np.ndarray) -> np.ndarray:
    """Recreate the emission-angle field used by the old image analysis."""
    center = np.mean(np.nonzero(mask), axis=1)
    radius = int(np.sqrt(np.count_nonzero(mask) / np.pi))
    rows, columns = np.indices(mask.shape)
    distance = np.hypot(columns - center[0], rows - center[1])
    emission = np.full(mask.shape, 90.0)
    inside = distance <= radius
    emission[inside] = np.degrees(
        np.arccos(np.sqrt(radius**2 - distance[inside] ** 2) / radius)
    )
    return emission


def _quadratic(mu: np.ndarray, intensity: float, u1: float, u2: float) -> np.ndarray:
    return intensity * (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2)


def _profile_by_emission(
    image: np.ndarray, emission: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = emission < MAXIMUM_PROFILE_EMISSION_DEGREES
    x = emission[selected]
    y = image[selected]
    unique, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=y)
    counts = np.bincount(inverse)
    return unique, sums / counts


def _image_quality(image: np.ndarray, emission: np.ndarray, mask: np.ndarray) -> bool:
    integer_emission = emission[mask].astype(np.uint16)
    brightness = image[mask]
    unique = np.unique(integer_emission)
    groups = [brightness[integer_emission == value] for value in unique]
    noise = sum(float(np.std(group)) * len(group) for group in groups) / len(brightness)
    maximum_mean = max(float(np.mean(group)) for group in groups)
    return noise <= MAXIMUM_NOISE and maximum_mean >= MINIMUM_PEAK_BRIGHTNESS


def fit_srtc_image(
    image: np.ndarray, emission: np.ndarray, mask: np.ndarray
) -> tuple[float, bool]:
    """Rebuild one synthetic image's limb sum and old quality flag."""
    x, y = _profile_by_emission(image * mask, emission)
    grid = np.linspace(float(x.min()), float(x.max()), SRTC_PROFILE_POINTS)
    interpolated = PchipInterpolator(x, gaussian_filter1d(y, sigma=1))(grid)
    smoothed = gaussian_filter1d(interpolated, sigma=SRTC_SMOOTHING_SIGMA)
    parameters, _ = curve_fit(
        _quadratic,
        np.cos(np.deg2rad(grid)),
        smoothed,
        p0=[1.0, 0.5, 0.5],
        bounds=([0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
    )
    return float(parameters[1] + parameters[2]), _image_quality(image, emission, mask)


def rebuild_srtc_images(image_dir: Path) -> pl.DataFrame:
    """Recompute the result table directly from every synthetic TIFF."""
    paths = tuple(sorted(image_dir.glob("*.tif")))
    mask = build_srtc_mask(paths)
    emission = srtc_emission_angles(mask)
    rows: list[dict[str, str | float | bool]] = []
    inventory = inventory_srtc_images(image_dir)
    parameters = {row["file_name"]: row for row in inventory.iter_rows(named=True)}
    for path in paths:
        if path.name not in parameters:
            continue
        u_sum, usable = fit_srtc_image(_read_grayscale(path), emission, mask)
        source = parameters[path.name]
        rows.append(
            {
                **source,
                TARGET: u_sum,
                "classification": "Darkened" if u_sum >= 0 else "Brightened",
                "usable": usable,
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def compare_srtc_results(
    rebuilt: pl.DataFrame, saved: pl.DataFrame
) -> dict[str, float | int]:
    """Compare rebuilt image results with the preserved CSV by input case."""
    joined = rebuilt.join(
        saved.select(*FEATURES, TARGET, "usable"),
        on=FEATURES,
        suffix="_saved",
        validate="1:1",
    ).with_columns((pl.col(TARGET) - pl.col(f"{TARGET}_saved")).abs().alias("drift"))
    return {
        "joined_rows": joined.height,
        "equal_usable_flags": int(
            (joined.get_column("usable") == joined.get_column("usable_saved")).sum()
        ),
        "median_absolute_u_sum_drift": float(
            np.median(joined.get_column("drift").to_numpy())
        ),
        "maximum_absolute_u_sum_drift": float(
            np.max(joined.get_column("drift").to_numpy())
        ),
    }


def train_srtc_models(table: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit three fixed-seed models and return test metrics and RF importance."""
    usable = table.filter(pl.col("usable"))
    x = usable.select(FEATURES).to_numpy()
    y = usable.get_column(TARGET).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    random_forest = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    models = {
        "random_forest": random_forest,
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, random_state=42
        ),
        "scaled_svr": make_pipeline(StandardScaler(), SVR(kernel="rbf")),
    }
    metric_rows: list[dict[str, str | int | float]] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        metric_rows.append(
            {
                "model": name,
                "training_rows": len(x_train),
                "test_rows": len(x_test),
                "r_squared": float(r2_score(y_test, prediction)),
                "mean_absolute_error": float(mean_absolute_error(y_test, prediction)),
                "root_mean_squared_error": float(
                    mean_squared_error(y_test, prediction) ** 0.5
                ),
            }
        )
    importance = pl.DataFrame(
        {"feature": FEATURES, "importance": random_forest.feature_importances_}
    ).sort("importance", descending=True)
    return pl.from_dicts(metric_rows), importance


def write_srtc_analysis(
    csv_path: Path, image_dir: Path, output_dir: Path
) -> tuple[pl.DataFrame, pl.DataFrame]:
    saved = read_srtc_table(csv_path)
    inventory = inventory_srtc_images(image_dir)
    table = rebuild_srtc_images(image_dir)
    comparison = compare_srtc_results(table, saved)
    metrics, importance = train_srtc_models(table)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output_dir / "srtc-results.parquet")
    inventory.write_parquet(output_dir / "srtc-image-inventory.parquet")
    metrics.write_parquet(output_dir / "srtc-model-metrics.parquet")
    importance.write_parquet(output_dir / "srtc-feature-importance.parquet")
    report = {
        "result_rows": table.height,
        "usable_rows": table.filter(pl.col("usable")).height,
        "image_rows": inventory.height,
        "models": metrics.to_dicts(),
        "saved_comparison": comparison,
    }
    (output_dir / "srtc-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return metrics, importance
