# src/sim/synth_claims.py
#!/usr/bin/env python3
"""
Synthetic claims with a strong, monotonic link to risk features.
- p(claim) = sigmoid(b + w·x), with b calibrated to ~30% prevalence
- More harsh braking / overspeed / night results in the  higher claim prob
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

DP  = Path("data/driver_period.csv")
OUT = Path("data/claims.csv")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiplier", type=float, default=1.0, help="scale the final prob")
    ap.add_argument("--cap",        type=float, default=0.95, help="max prob clip")
    ap.add_argument("--target-prev",type=float, default=0.30, help="target average prevalence")
    ap.add_argument("--seed",       type=int,   default=42,   help="rng seed")
    ap.add_argument("--min-claims", type=int,   default=30,   help="guaranteed minimum claims")
    return ap.parse_args()

def main():
    args = parse_args()
    if not DP.exists():
        raise SystemExit(f"Missing {DP}. Run build_driver_period first.")

    rng = np.random.default_rng(args.seed)
    dp  = pd.read_csv(DP, parse_dates=["period_start"]).copy()

    # guards
    for c, d in [("harsh_brake_per_100mi",0.0),("overspeed_ratio",0.0),("night_ratio",0.0)]:
        if c not in dp.columns: dp[c] = d
        dp[c] = pd.to_numeric(dp[c], errors="coerce").fillna(d)

    # weights 
    w_hb, w_ov, w_ng = 0.025, 4.0, 2.5  # hb per 100mi gets modest slope; speed/night stronger
    r = (w_hb*dp["harsh_brake_per_100mi"] +
         w_ov*dp["overspeed_ratio"] +
         w_ng*dp["night_ratio"])

    # calibrate intercept so average probability ≈ target prevalence
    target = float(np.clip(args.target_prev, 0.05, 0.9))
    logit  = lambda x: np.log(x/(1-x))
    b      = logit(target) - r.mean()

    p = 1.0/(1.0 + np.exp(-(b + r)))
    p = np.clip(args.multiplier * p, 0.001, args.cap)

    rows = []
    for pid, ps, prob in zip(dp["policy_id"], dp["period_start"], p):
        if rng.random() < prob:
            loss_dt = ps + pd.Timedelta(days=int(rng.integers(1, 90)))
            paid    = float(rng.lognormal(mean=8.3, sigma=0.75))
            rows.append({
                "policy_id": pid,
                "claim_id": f"c_{rng.integers(1_000_000_000)}",
                "loss_dt": loss_dt.isoformat(),
                "at_fault_flag": 1,
                "paid_severity_usd": paid
            })

    # ensure minimum claims for learnability
    if len(rows) < args.min_claims and len(dp) > 0:
        need = args.min_claims - len(rows)
        risk_order = r.sort_values(ascending=False).index[:need]
        for idx in risk_order:
            ps = dp.loc[idx, "period_start"]
            loss_dt = ps + pd.Timedelta(days=int(rng.integers(1, 90)))
            paid    = float(rng.lognormal(mean=8.3, sigma=0.75))
            rows.append({
                "policy_id": dp.loc[idx,"policy_id"],
                "claim_id": f"c_{rng.integers(1_000_000_000)}",
                "loss_dt": loss_dt.isoformat(),
                "at_fault_flag": 1,
                "paid_severity_usd": paid
            })

    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {OUT} rows: {len(rows)}")

if __name__ == "__main__":
    main()

