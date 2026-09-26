"""Preprocess the Intel Berkeley Research Lab temperature dataset into the
point cloud used to build the WDBO paper's "Temperature" benchmark.

Raw readings (`data.txt`) and sensor locations (`mote_locs.txt`) are expected
in `DATA_DIR`, in the format published at
https://db.csail.mit.edu/labdata/labdata.html. See README.md in this folder
for the full methodology and why sensor filtering is needed.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths import DATA_DIR

READING_COLUMNS = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]
MINUTES_PER_DAY = 24 * 60


def load_readings(data_path: Path) -> pd.DataFrame:
    """Parse the raw whitespace-separated sensor log.

    A small fraction of lines in this dataset are truncated (dropped radio
    packets), so malformed rows are skipped rather than raising.
    """
    df = pd.read_csv(data_path, sep=r"\s+", names=READING_COLUMNS, header=None, engine="c", on_bad_lines="skip")
    df = df.dropna(subset=["date", "time", "moteid", "temperature", "humidity", "voltage"])
    df["moteid"] = pd.to_numeric(df["moteid"], errors="coerce")
    df = df.dropna(subset=["moteid"])
    df["moteid"] = df["moteid"].astype(int)
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    return df.dropna(subset=["timestamp"])


def load_locations(locs_path: Path) -> pd.DataFrame:
    """Parse the `moteid x y` sensor location table."""
    locs = pd.read_csv(locs_path, sep=r"\s+", names=["moteid", "x", "y"], header=None).dropna()
    locs["moteid"] = locs["moteid"].astype(int)
    return locs


def select_day(readings: pd.DataFrame, date: str | None) -> tuple[pd.DataFrame, str]:
    """Restrict readings to a single calendar day (defaults to the earliest one)."""
    date = date or str(readings["date"].min())
    day_readings = readings[readings["date"] == date]
    if day_readings.empty:
        raise ValueError(f"No readings found for date {date!r}")
    return day_readings, date


def drop_implausible_readings(
    readings: pd.DataFrame,
    temperature_range: tuple[float, float],
    humidity_range: tuple[float, float],
    voltage_range: tuple[float, float],
) -> pd.DataFrame:
    """Drop garbage packets (e.g. the well-documented sub-2V brownout readings
    that report physically impossible temperature/humidity values)."""
    plausible = (
        readings["temperature"].between(*temperature_range)
        & readings["humidity"].between(*humidity_range)
        & readings["voltage"].between(*voltage_range)
    )
    return readings[plausible]


def bin_by_time(readings: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    """Average raw readings into fixed-width per-sensor time bins.

    This both denoises individual readings and keeps the point cloud handed
    to the spatio-temporal interpolator (see objective.py) small enough to
    invert directly.
    """
    minute_of_day = readings["timestamp"].dt.hour * 60 + readings["timestamp"].dt.minute
    bin_id = minute_of_day // bin_minutes
    return (
        readings.assign(bin=bin_id)
        .groupby(["moteid", "bin"], as_index=False)["temperature"]
        .mean()
    )


def select_reliable_sensors(binned: pd.DataFrame, n_bins_per_day: int, min_coverage: float) -> list[int]:
    """Keep sensors that reported data in at least `min_coverage` of the day's bins.

    The raw deployment has 54 known sensor locations, two of which (see
    README.md) never reported a single reading in the whole 5-week
    deployment. The rest have wildly uneven delivery rates because of
    wireless link quality, so a minimum-coverage bar is the standard way
    (used by every paper reusing this dataset) to keep only the sensors
    reliable enough to support a clean interpolation.
    """
    coverage = binned.groupby("moteid").size() / n_bins_per_day
    kept = coverage[coverage >= min_coverage].index
    return sorted(int(moteid) for moteid in kept)


def build_point_cloud(binned: pd.DataFrame, locs: pd.DataFrame, kept_sensors: list[int], bin_minutes: int):
    """Join binned readings with sensor locations and normalize into the unit cube.

    Returns `(points, temperature)` where `points[:, 0:2]` are spatial
    coordinates normalized to [0, 1]^2 over the kept sensors' bounding box,
    and `points[:, 2]` is time-of-day normalized to [0, 1].
    """
    kept_locs = locs[locs["moteid"].isin(kept_sensors)].set_index("moteid")
    binned = binned[binned["moteid"].isin(kept_sensors)]

    x_bounds = (kept_locs["x"].min(), kept_locs["x"].max())
    y_bounds = (kept_locs["y"].min(), kept_locs["y"].max())

    x = (binned["moteid"].map(kept_locs["x"]).to_numpy() - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
    y = (binned["moteid"].map(kept_locs["y"]).to_numpy() - y_bounds[0]) / (y_bounds[1] - y_bounds[0])
    t = (binned["bin"].to_numpy() * bin_minutes) / MINUTES_PER_DAY

    points = np.stack([x, y, t], axis=1)
    return points, binned["temperature"].to_numpy(), x_bounds, y_bounds


def preprocess(
    data_path: Path,
    locs_path: Path,
    date: str | None,
    bin_minutes: int,
    min_coverage: float,
    temperature_range: tuple[float, float],
    humidity_range: tuple[float, float],
    voltage_range: tuple[float, float],
) -> dict:
    """Run the full pipeline and return everything needed for `np.savez`."""
    readings = load_readings(data_path)
    locs = load_locations(locs_path)
    readings = readings[readings["moteid"].isin(locs["moteid"])]

    day_readings, date = select_day(readings, date)
    n_bins_per_day = MINUTES_PER_DAY // bin_minutes

    clean_readings = drop_implausible_readings(day_readings, temperature_range, humidity_range, voltage_range)
    binned = bin_by_time(clean_readings, bin_minutes)
    kept_sensors = select_reliable_sensors(binned, n_bins_per_day, min_coverage)
    points, temperature, x_bounds, y_bounds = build_point_cloud(binned, locs, kept_sensors, bin_minutes)

    print(f"Sensor locations known (data/mote_locs.txt):        {locs['moteid'].nunique()}")
    print(f"Sensors reporting at least one reading on {date}:   {day_readings['moteid'].nunique()}")
    print(f"Sensors kept (>= {min_coverage:.0%} bin coverage):        {len(kept_sensors)}")
    print(f"Time bins/day: {n_bins_per_day} ({bin_minutes} min each) | points: {len(temperature)}")

    return {
        "points": points,
        "temperature": temperature,
        "source_date": date,
        "bin_minutes": bin_minutes,
        "kept_sensors": np.array(kept_sensors),
        "x_bounds": np.array(x_bounds),
        "y_bounds": np.array(y_bounds),
        "min_coverage": min_coverage,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DATA_DIR / "data.txt")
    parser.add_argument("--locs-file", type=Path, default=DATA_DIR / "mote_locs.txt")
    parser.add_argument("--date", type=str, default=None, help="Calendar day to use, e.g. 2004-02-28. Defaults to the earliest day in the dataset.")
    parser.add_argument("--bin-minutes", type=int, default=10, help="Width of the per-sensor time-averaging bins.")
    parser.add_argument("--min-coverage", type=float, default=0.85, help="Minimum fraction of a day's bins a sensor must cover to be kept.")
    parser.add_argument("--temperature-range", type=float, nargs=2, default=(0.0, 50.0))
    parser.add_argument("--humidity-range", type=float, nargs=2, default=(0.0, 100.0))
    parser.add_argument("--voltage-range", type=float, nargs=2, default=(2.0, 3.0))
    parser.add_argument("--output", type=Path, default=DATA_DIR / "processed.npz")
    args = parser.parse_args()

    result = preprocess(
        data_path=args.data_file,
        locs_path=args.locs_file,
        date=args.date,
        bin_minutes=args.bin_minutes,
        min_coverage=args.min_coverage,
        temperature_range=tuple(args.temperature_range),
        humidity_range=tuple(args.humidity_range),
        voltage_range=tuple(args.voltage_range),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **result)
    print(f"Saved preprocessed dataset to {args.output}")


if __name__ == "__main__":
    main()
