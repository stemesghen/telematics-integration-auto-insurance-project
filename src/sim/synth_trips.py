# builds trips parquet/csv

"""
Build trip-level aggregates from ingested telemetry JSONL files.

Input:  data_ingest/<policy_id>__<trip_id>.jsonl
Each line: {"policy_id":..., "trip_id":..., "ts": ISO8601, "lat":.., "lon":..,
            "speed_mps":.., "accel_mps2":.., "heading":..,
            "braking_flag":0/1, "phone_usage_flag":0/1, "road_speed_limit":..}

Output: data/trips.csv with columns:
policy_id,trip_id,start_ts,end_ts,duration_s,miles,avg_speed_mps,
harsh_brake_ct,phone_usage_ct,overspeed_ratio,night_ratio
"""
from __future__ import annotations
import json, math, glob
from pathlib import Path
import pandas as pd

IN_DIR  = Path("data_ingest")
OUT_DIR = Path("data"); OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "trips.csv"

EARTH_M = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    if None in (lat1, lon1, lat2, lon2):
        return 0.0
    r = EARTH_M
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlambda = math.radians((lon2 or 0) - (lon1 or 0))
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return 2*r*math.atan2(math.sqrt(a), math.sqrt(1-a))

def is_night(ts: pd.Timestamp) -> bool:
    # simple heuristic: 22:00–06:00 considered night
    h = ts.hour
    return h >= 22 or h < 6

def process_file(path: Path) -> dict | None:
    policy_id, trip_id = path.stem.split("__", 1) if "__" in path.stem else ("unknown", path.stem)
    pts = []
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                pts.append(r)
            except json.JSONDecodeError:
                continue
    if not pts:
        return None

    # sort by timestamp
    for r in pts:
        # ensure keys exist
        r.setdefault("speed_mps", 0.0)
        r.setdefault("road_speed_limit", r.get("speed_mps", 0.0))
        r.setdefault("braking_flag", 0)
        r.setdefault("phone_usage_flag", 0)

    df = pd.DataFrame(pts)
    # parse datetimes safely
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.sort_values("ts").dropna(subset=["ts"])
    if df.empty:
        return None

    # distance
    dist_m = 0.0
    lat = df["lat"].tolist()
    lon = df["lon"].tolist()
    for i in range(1, len(df)):
        dist_m += haversine_m(lat[i-1], lon[i-1], lat[i], lon[i])

    start_ts = df["ts"].iloc[0]
    end_ts   = df["ts"].iloc[-1]
    duration_s = max(0.0, (end_ts - start_ts).total_seconds())

    miles = dist_m / 1609.34
    avg_speed = df.get("speed_mps", pd.Series([0.0]*len(df))).mean()

    # safety features
    harsh_brake_ct  = int(df.get("braking_flag", 0).fillna(0).astype(int).sum())
    phone_usage_ct  = int(df.get("phone_usage_flag", 0).fillna(0).astype(int).sum())

    # overspeed ratio: share of points exceeding limit + small tolerance (0.5 m/s ~ 1.1 mph)
    if "road_speed_limit" in df and "speed_mps" in df:
        overspeed = (df["speed_mps"].fillna(0) > (df["road_speed_limit"].fillna(0) + 0.5)).mean()
    else:
        overspeed = 0.0

    # night ratio in UTC for POC
    night_ratio = df["ts"].apply(is_night).mean()

    return {
        "policy_id": policy_id,
        "trip_id": trip_id,
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "duration_s": duration_s,
        "miles": miles,
        "avg_speed_mps": avg_speed,
        "harsh_brake_ct": harsh_brake_ct,
        "phone_usage_ct": phone_usage_ct,
        "overspeed_ratio": overspeed,
        "night_ratio": night_ratio,
    }

def main():
    files = sorted(IN_DIR.glob("*.jsonl"))
    if not files:
        print(f"No input files in {IN_DIR.resolve()}. Did you POST to /ingest/telemetry?")
        return

    rows = []
    for p in files:
        rec = process_file(p)
        if rec:
            rows.append(rec)

    if not rows:
        print("No valid trips parsed.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} trips → {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()

