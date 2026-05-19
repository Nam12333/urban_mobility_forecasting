"""Urban mobility demand forecasting without third-party dependencies.

The script can use the UCI Bike Sharing `hour.csv` file when present. If the
dataset is not available, it generates a realistic synthetic sample so the
project can run offline in a fresh virtual environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


@dataclass
class Record:
    season: int
    month: int
    hour: int
    holiday: int
    weekday: int
    workingday: int
    weather: int
    temp: float
    humidity: float
    windspeed: float
    count: float


def download_uci_dataset() -> Path:
    """Download and extract the UCI Bike Sharing dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = DATA_DIR / "bike_sharing_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "bike_sharing_dataset.zip"

    if not zip_path.exists():
        print("Downloading UCI Bike Sharing dataset...")
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)

    hour_csv = raw_dir / "hour.csv"
    if not hour_csv.exists():
        raise FileNotFoundError("UCI archive did not contain hour.csv")

    return hour_csv


def load_uci_hour_csv(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append(
                Record(
                    season=int(row["season"]),
                    month=int(row["mnth"]),
                    hour=int(row["hr"]),
                    holiday=int(row["holiday"]),
                    weekday=int(row["weekday"]),
                    workingday=int(row["workingday"]),
                    weather=int(row["weathersit"]),
                    temp=float(row["temp"]),
                    humidity=float(row["hum"]),
                    windspeed=float(row["windspeed"]),
                    count=float(row["cnt"]),
                )
            )
    return records


def generate_synthetic_records(rows: int = 720, seed: int = 42) -> list[Record]:
    """Create an offline sample with realistic demand patterns."""
    rng = random.Random(seed)
    records: list[Record] = []

    for idx in range(rows):
        hour = idx % 24
        day = idx // 24
        weekday = day % 7
        month = (day // 30) % 12 + 1
        season = 1 if month in (12, 1, 2) else 2 if month in (3, 4, 5) else 3 if month in (6, 7, 8) else 4
        holiday = 1 if day in (6, 41, 92, 180, 260) else 0
        workingday = 1 if weekday < 5 and not holiday else 0
        weather = rng.choices([1, 2, 3], weights=[0.62, 0.28, 0.10])[0]

        seasonal_temp = 0.48 + 0.32 * math.sin((month - 3) / 12 * 2 * math.pi)
        temp = min(1.0, max(0.05, seasonal_temp + rng.gauss(0, 0.06)))
        humidity = min(1.0, max(0.2, 0.58 + (weather - 1) * 0.12 + rng.gauss(0, 0.08)))
        windspeed = min(0.9, max(0.02, 0.22 + rng.gauss(0, 0.08)))

        commute_peak = 1.0 if workingday and hour in (7, 8, 17, 18) else 0.0
        leisure_peak = 1.0 if not workingday and 11 <= hour <= 17 else 0.0
        night_penalty = 1.0 if hour <= 5 else 0.0
        weather_penalty = {1: 0, 2: 42, 3: 120}[weather]

        count = (
            95
            + 280 * temp
            - 115 * humidity
            - 70 * windspeed
            + 150 * commute_peak
            + 120 * leisure_peak
            + 65 * workingday
            - 85 * night_penalty
            - weather_penalty
            + rng.gauss(0, 24)
        )
        records.append(
            Record(
                season=season,
                month=month,
                hour=hour,
                holiday=holiday,
                weekday=weekday,
                workingday=workingday,
                weather=weather,
                temp=temp,
                humidity=humidity,
                windspeed=windspeed,
                count=max(1.0, round(count, 0)),
            )
        )

    return records


def circular(value: int, period: int) -> tuple[float, float]:
    radians = 2 * math.pi * value / period
    return math.sin(radians), math.cos(radians)


def record_to_features(record: Record) -> dict[str, float]:
    hour_sin, hour_cos = circular(record.hour, 24)
    month_sin, month_cos = circular(record.month, 12)
    weekday_sin, weekday_cos = circular(record.weekday, 7)
    return {
        "temp": record.temp,
        "humidity": record.humidity,
        "windspeed": record.windspeed,
        "workingday": float(record.workingday),
        "holiday": float(record.holiday),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "weekday_sin": weekday_sin,
        "weekday_cos": weekday_cos,
        "commute_hour": float(record.hour in (7, 8, 17, 18)),
        "leisure_hour": float(11 <= record.hour <= 17),
        "night_hour": float(record.hour <= 5),
        "weather_clear": float(record.weather == 1),
        "weather_mist": float(record.weather == 2),
        "weather_bad": float(record.weather >= 3),
        "season_winter": float(record.season == 1),
        "season_spring": float(record.season == 2),
        "season_summer": float(record.season == 3),
        "season_fall": float(record.season == 4),
    }


def build_matrix(records: list[Record]) -> tuple[list[str], list[list[float]], list[float]]:
    feature_names = sorted(record_to_features(records[0]).keys())
    matrix = []
    target = []
    for record in records:
        features = record_to_features(record)
        matrix.append([features[name] for name in feature_names])
        target.append(record.count)
    return feature_names, matrix, target


def train_test_split(
    x_values: list[list[float]],
    y_values: list[float],
    test_ratio: float = 0.2,
    seed: int = 7,
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    indices = list(range(len(x_values)))
    random.Random(seed).shuffle(indices)
    split = int(len(indices) * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return (
        [x_values[i] for i in train_idx],
        [x_values[i] for i in test_idx],
        [y_values[i] for i in train_idx],
        [y_values[i] for i in test_idx],
    )


def fit_standardizer(x_train: list[list[float]]) -> tuple[list[float], list[float]]:
    columns = list(zip(*x_train))
    means = [statistics.fmean(column) for column in columns]
    stds = [statistics.pstdev(column) or 1.0 for column in columns]
    return means, stds


def transform(x_values: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    return [[(value - means[i]) / stds[i] for i, value in enumerate(row)] for row in x_values]


def predict_linear(x_values: list[list[float]], weights: list[float], intercept: float) -> list[float]:
    return [sum(weight * value for weight, value in zip(weights, row)) + intercept for row in x_values]


def train_ridge_regression(
    x_train: list[list[float]],
    y_train: list[float],
    learning_rate: float = 0.025,
    epochs: int = 900,
    l2: float = 0.03,
) -> tuple[list[float], float]:
    weights = [0.0 for _ in x_train[0]]
    intercept = statistics.fmean(y_train)
    n_rows = len(x_train)

    for _ in range(epochs):
        grad_weights = [0.0 for _ in weights]
        grad_intercept = 0.0

        for row, actual in zip(x_train, y_train):
            predicted = sum(weight * value for weight, value in zip(weights, row)) + intercept
            error = predicted - actual
            grad_intercept += error
            for i, value in enumerate(row):
                grad_weights[i] += error * value

        intercept -= learning_rate * grad_intercept / n_rows
        for i in range(len(weights)):
            regularization = l2 * weights[i]
            weights[i] -= learning_rate * ((grad_weights[i] / n_rows) + regularization)

    return weights, intercept


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    mae = statistics.fmean(abs(a - p) for a, p in zip(actual, predicted))
    rmse = math.sqrt(statistics.fmean((a - p) ** 2 for a, p in zip(actual, predicted)))
    mean_actual = statistics.fmean(actual)
    ss_total = sum((a - mean_actual) ** 2 for a in actual)
    ss_residual = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    r2 = 1 - ss_residual / ss_total if ss_total else 0.0
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 3)}


def top_features(feature_names: list[str], weights: list[float], limit: int = 8) -> list[dict[str, float | str]]:
    ranked = sorted(zip(feature_names, weights), key=lambda item: abs(item[1]), reverse=True)
    return [{"feature": name, "weight": round(weight, 3)} for name, weight in ranked[:limit]]


def load_records(download: bool = False) -> tuple[str, list[Record]]:
    hour_csv = DATA_DIR / "hour.csv"
    if download:
        hour_csv = download_uci_dataset()

    if hour_csv.exists():
        return str(hour_csv), load_uci_hour_csv(hour_csv)

    return "generated synthetic sample", generate_synthetic_records()


def run_pipeline(download: bool = False) -> dict[str, object]:
    source, records = load_records(download=download)
    feature_names, x_values, y_values = build_matrix(records)
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values)
    means, stds = fit_standardizer(x_train)
    x_train_scaled = transform(x_train, means, stds)
    x_test_scaled = transform(x_test, means, stds)

    baseline_value = statistics.fmean(y_train)
    baseline_predictions = [baseline_value for _ in y_test]

    weights, intercept = train_ridge_regression(x_train_scaled, y_train)
    model_predictions = predict_linear(x_test_scaled, weights, intercept)

    report = {
        "dataset_source": source,
        "records": len(records),
        "features": len(feature_names),
        "baseline": metrics(y_test, baseline_predictions),
        "ridge_regression": metrics(y_test, model_predictions),
        "top_features": top_features(feature_names, weights),
        "example_predictions": [
            {"actual": round(actual, 1), "predicted": round(predicted, 1)}
            for actual, predicted in list(zip(y_test, model_predictions))[:8]
        ],
    }
    return report


def write_report(report: dict[str, object]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "model_report.json"
    txt_path = REPORTS_DIR / "summary.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "Urban Mobility Demand Forecasting",
        "=" * 35,
        f"Dataset: {report['dataset_source']}",
        f"Records: {report['records']}",
        f"Features: {report['features']}",
        "",
        f"Baseline: {report['baseline']}",
        f"Ridge Regression: {report['ridge_regression']}",
        "",
        "Most influential features:",
    ]
    for feature in report["top_features"]:
        lines.append(f"- {feature['feature']}: {feature['weight']}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast urban bike-sharing demand.")
    parser.add_argument("--download", action="store_true", help="Download UCI Bike Sharing dataset before running.")
    args = parser.parse_args()

    report = run_pipeline(download=args.download)
    write_report(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

