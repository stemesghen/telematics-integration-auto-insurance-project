import pandas as pd
from pathlib import Path

DP = Path("data/driver_period.csv")
CLAIMS = Path("data/claims.csv")
OUT = Path("data/driver_period_labeled.csv")

def main():
    if not DP.exists(): raise SystemExit(f"Missing {DP}")
    dp = pd.read_csv(DP, parse_dates=["period_start"])
    if not CLAIMS.exists():
        dp["label_claim_next90d"] = 0
        dp.to_csv(OUT, index=False)
        print(f"No claims found; wrote zeros -> {OUT}"); return

    claims = pd.read_csv(CLAIMS, parse_dates=["loss_dt"])
    dp["label_window_end"] = dp["period_start"] + pd.Timedelta(days=90)

    # fast join by policy then row-wise filter
    dp["label_claim_next90d"] = 0
    for pid, group in dp.groupby("policy_id", group_keys=False):
        c = claims[(claims.policy_id == pid) & (claims.at_fault_flag == 1)]
        if c.empty: continue
        # for each row, check if any loss_dt in (period_start, period_start+90]
        mask = group.apply(lambda r: ((c.loss_dt > r.period_start) & (c.loss_dt <= r.label_window_end)).any(), axis=1)
        dp.loc[group.index, "label_claim_next90d"] = mask.astype(int)

    dp.drop(columns=["label_window_end"], inplace=True)
    dp.to_csv(OUT, index=False)
    print(f"Labeled -> {OUT} (positives={dp['label_claim_next90d'].sum()})")

if __name__ == "__main__":
    main()

