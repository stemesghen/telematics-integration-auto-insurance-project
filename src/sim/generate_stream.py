#!/usr/bin/env python3
"""
Emit synthetic telematics to your ingest endpoint.

Usage:
  python src/sim/generate_stream.py \
    --drivers 5 --days 1 --trips-per-day 2 --hz 1 \
    --post http://127.0.0.1:8080/ingest/telemetry \
    --api-key dev-secret
"""
from __future__ import annotations
import argparse, math, random, time, uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple
import os
import requests

EARTH_M = 6371000.0

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def step_latlon(lat: float, lon: float, bearing_deg: float, meters: float) -> Tuple[float, float]:
    if meters <= 0:
        return lat, lon
    rad = math.pi / 180.0
    lat_rad = lat * rad
    dlat = (meters / EARTH_M) * (180.0 / math.pi) * math.sin(bearing_deg * rad)
    dlng = (meters / (EARTH_M * max(1e-6, math.cos(abs(lat_rad))))) * (180.0 / math.pi) * math.cos(bearing_deg * rad)
    return lat + dlat, lon + dlng

def gen_trip_points(
    start_dt: datetime,
    minutes: int,
    hz: int,
    start_lat: float,
    start_lon: float,
    base_speed_mps: float,
    overspeed_bias: float = 0.0,
    brake_rate_per_min: float = 0.6,
) -> List[Dict[str, Any]]:
    n = minutes * 60 * hz
    lat, lon = start_lat, start_lon
    heading = random.uniform(0, 360)
    points: List[Dict[str, Any]] = []

    for i in range(n):
        ts = start_dt + timedelta(seconds=i / hz)

        heading = (heading + random.uniform(-5, 5)) % 360
        target = max(0.0, base_speed_mps + overspeed_bias + random.uniform(-1.5, 1.5))
        if random.random() < 0.02:
            target *= random.uniform(0.3, 0.7)

        if points:
            prev = points[-1]["speed_mps"]
            ax = max(-4.0, min(3.0, (target - prev) * 0.5))  
            speed = max(0.0, prev + ax / max(1, hz))
        else:
            speed, ax = max(0.0, target), 0.0

        # rare harsh brake events
        if random.random() < (brake_rate_per_min / (60 * max(1, hz))):
            ax = random.uniform(-5.0, -2.5)
            speed = max(0.0, speed + ax / max(1, hz))

        meters = speed / max(1, hz)
        lat, lon = step_latlon(lat, lon, heading, meters)

        points.append({
            "ts": iso(ts),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "speed_mps": round(speed, 3),
            "ax_mps2": round(ax, 3),     
        })
    return points

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivers", type=int, default=10)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--trips-per-day", type=int, default=2, dest="trips_per_day")
    ap.add_argument("--hz", type=int, default=1)
    ap.add_argument("--minutes-per-trip", type=int, default=12, dest="minutes_per_trip")
    ap.add_argument("--post", required=True, help="e.g. http://127.0.0.1:8080/ingest/telemetry")
    ap.add_argument("--api-key", default=os.getenv("API_KEY", "dev-secret"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    session = requests.Session()
    headers = {"Content-Type": "application/json", "x-api-key": args.api_key}

    base_lat, base_lon = 38.9072, -77.0369  # DC-ish
    total_points, start_clock = 0, time.time()

    for d in range(args.drivers):
        policy_id = f"driver_{d:05d}"
        base_speed = random.uniform(8.0, 16.0)   # appprox 18–36 mph
        overspeed_bias = random.uniform(-0.5, 2.5)

        for day in range(args.days):
            day_dt = datetime.now(timezone.utc) + timedelta(days=day)
            for t in range(args.trips_per_day):
                trip_tag = uuid.uuid4().hex[:8]
                trip_id = f"{policy_id}_{day_dt.date()}_{t}_{trip_tag}"
                start_dt = day_dt + timedelta(hours=random.randint(6, 22),
                                              minutes=random.randint(0, 59))
                jitter_lat = base_lat + random.uniform(-0.05, 0.05)
                jitter_lon = base_lon + random.uniform(-0.05, 0.05)

                points = gen_trip_points(start_dt, args.minutes_per_trip, args.hz,
                                         jitter_lat, jitter_lon, base_speed, overspeed_bias)
                payload = {"policy_id": policy_id, "trip_id": trip_id, "points": points}

                try:
                    r = session.post(args.post, json=payload, headers=headers, timeout=15)
                    if r.status_code != 200:
                        print(f"[WARN] POST failed {r.status_code}: {r.text[:200]}")
                    else:
                        received = r.json().get("received", 0)
                        total_points += received
                        print(f"[OK] {policy_id} {trip_id} -> received={received}")
                except Exception as e:
                    print(f"[ERR] POST exception: {e}")
                time.sleep(0.05)

    elapsed = time.time() - start_clock
    print(f"Done. Posted ~{total_points} points in {elapsed:.1f}s.")

if __name__ == "__main__":
    main()

