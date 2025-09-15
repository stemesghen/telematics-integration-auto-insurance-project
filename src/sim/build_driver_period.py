from pathlib import Path
import numpy as np
import pandas as pd

TRIPS = Path("data/trips.csv")
OUT   = Path("data/driver_period.csv")

def main():
    if not TRIPS.exists():
        raise SystemExit(f"Missing {TRIPS}. Generate trips or POST telemetry first.")

    trips = pd.read_csv(TRIPS)
    trips.columns = [c.strip() for c in trips.columns]

    # drop any old demo row if present
    if "policy_id" in trips.columns:
        trips = trips[trips["policy_id"].astype(str) != "demo"].copy()

    # timestamp parsing
    for col in ("start_ts", "end_ts"):
        trips[col] = pd.to_datetime(trips[col], errors="coerce", utc=True)

    pre = len(trips)
    trips = trips.dropna(subset=["start_ts", "end_ts"]).copy()
    print(f"[debug] loaded trips: {pre}, after ts dropna: {len(trips)}")
    if trips.empty:
        raise SystemExit("All trip timestamps became null; check data/trips.csv contents.")

    # Ensure numeric columns exist (safe defaults if missing)
    defaults = {
        "miles": 0.0,
        "duration_s": 0.0,
        "harsh_brake_ct": 0.0,
        "phone_usage_ct": 0.0,
        "avg_speed_mps": np.nan,   # fallback if NaN/missing
        "overspeed_ratio": 0.0,
        "night_ratio": 0.0,
    }
    for c, v in defaults.items():
        if c not in trips.columns:
            trips[c] = v
        trips[c] = pd.to_numeric(trips[c], errors="coerce")

    # Fallback for avg_speed_mps if NaN: distance/time
    mask = trips["avg_speed_mps"].isna()
    good = mask & (trips["duration_s"] > 0)
    trips.loc[good, "avg_speed_mps"] = (
        trips.loc[good, "miles"] * 1609.34 / trips.loc[good, "duration_s"]
    )
    trips["avg_speed_mps"]   = trips["avg_speed_mps"].fillna(0.0)
    trips["phone_usage_ct"]  = trips["phone_usage_ct"].fillna(0.0)
    trips["overspeed_ratio"] = trips["overspeed_ratio"].clip(lower=0).fillna(0.0)
    trips["night_ratio"]     = trips["night_ratio"].clip(lower=0).fillna(0.0)

    # Monthly period (make tz-naive for .to_period)
    start_naive = trips["start_ts"].dt.tz_localize(None)
    trips["period_start"] = start_naive.dt.to_period("M").dt.start_time

    # Weights for weighted means
    trips["w_miles"] = trips["miles"].clip(lower=1e-6)
    trips["w_dur"]   = trips["duration_s"].clip(lower=1.0)

    g = trips.groupby(["policy_id", "period_start"], dropna=False)

    # Aggregation: baseline + engineered
    agg = g.agg(
        trip_ct=("trip_id", "count"),
        exposure_miles=("miles", "sum"),
        duration_s=("duration_s", "sum"),
        harsh_brake_ct=("harsh_brake_ct", "sum"),
        phone_usage_ct=("phone_usage_ct", "sum"),
        # weighted mean speed by miles
        mean_speed_mps=("avg_speed_mps",
            lambda x: (x * trips.loc[x.index, "w_miles"]).sum() /
                      trips.loc[x.index, "w_miles"].sum()),
        # weighted mean overspeed by duration
        avg_overspeed_ratio=("overspeed_ratio",
            lambda x: (x * trips.loc[x.index, "w_dur"]).sum() /
                      trips.loc[x.index, "w_dur"].sum()),
        # variability across trips
        speed_var_across_trips=("avg_speed_mps", "var"),
        # night miles proxy
        night_miles=("night_ratio", lambda x: (x * trips.loc[x.index, "miles"]).sum()),
    ).reset_index()

    # Derived features
    agg["harsh_brake_per_100mi"] = agg["harsh_brake_ct"] / agg["exposure_miles"].clip(lower=1e-6) * 100.0
    agg["miles_per_trip"]        = agg["exposure_miles"] / agg["trip_ct"].clip(lower=1)
    agg["night_miles_ratio"]     = agg["night_miles"] / agg["exposure_miles"].clip(lower=1e-6)
    agg["phone_usage_per_hr"]    = agg["phone_usage_ct"] / (agg["duration_s"].clip(lower=1) / 3600.0)

    # Keep baseline six (for compatibility) and map from engineered
    agg["overspeed_ratio"] = agg["avg_overspeed_ratio"].fillna(0.0).clip(0, 1)
    agg["night_ratio"]     = agg["night_miles_ratio"].fillna(0.0).clip(0, 1)

    final_cols = [
        # baseline six used elsewhere
        "policy_id", "period_start", "trip_ct", "exposure_miles", "duration_s",
        "harsh_brake_per_100mi", "overspeed_ratio", "night_ratio",
        # engineered extras (trainer uses these)
        "avg_overspeed_ratio", "night_miles_ratio", "phone_usage_per_hr",
        "mean_speed_mps", "miles_per_trip", "speed_var_across_trips",
    ]
    agg = agg[final_cols].fillna(0.0)

    OUT.parent.mkdir(exist_ok=True)
    agg.to_csv(OUT, index=False)
    print(f"Wrote {len(agg)} rows → {OUT.resolve()}")
    print("unique drivers:", agg["policy_id"].nunique())
    print(agg.head(5))

if __name__ == "__main__":
    main()

