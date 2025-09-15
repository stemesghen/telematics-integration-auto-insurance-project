import os, numpy as np, pandas as pd, joblib

BASELINE   = float(os.getenv("BASELINE_PREV", "0.30"))
SLOPE_K    = float(os.getenv("PRICING_SLOPE", "0.5"))
FACTOR_MIN = float(os.getenv("FACTOR_MIN", "0.90"))
FACTOR_MAX = float(os.getenv("FACTOR_MAX", "1.10"))

def _predict(M, X):
    if hasattr(M, "predict_proba"): return M.predict_proba(X)[:,1]
    z = M.decision_function(X); return 1/(1+np.exp(-z))

df = pd.read_csv("data/driver_period.csv", parse_dates=["period_start"])
# pick latest period per policy
latest = df.sort_values(["policy_id","period_start"]).groupby("policy_id").tail(1).reset_index(drop=True)

model, features = joblib.load("models/behavior_model.joblib")
for c in features:
    if c not in latest.columns: latest[c] = 0.0
X = latest[features].fillna(0.0).values
p = _predict(model, X)

factor = np.clip(1 + SLOPE_K*(p - BASELINE), FACTOR_MIN, FACTOR_MAX)

out = latest[["policy_id","period_start"]].copy()
out["risk_p"] = p
out["telematics_factor"] = factor
out.to_csv("data/pricing_preview.csv", index=False)
print("Wrote -> data/pricing_preview.csv")

