from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

Path("models").mkdir(exist_ok=True)
DF = pd.read_csv("data/driver_period_labeled.csv")

FEATURES = [
    "exposure_miles","trip_ct","harsh_brake_per_100mi","duration_s",
    "avg_overspeed_ratio","night_miles_ratio","phone_usage_per_hr",
    "mean_speed_mps","miles_per_trip","speed_var_across_trips"
]
for c in FEATURES:
    if c not in DF.columns:
        DF[c] = 0.0
X = DF[FEATURES].fillna(0.0).values
y = DF["label_claim_next90d"].astype(int).values
groups = DF["policy_id"].astype(str).values

def proba_from_model(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:,1]
    z = m.decision_function(X)
    return 1/(1+np.exp(-z))

def fold_model():
    base = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])
    return base

# 5-fold stratified CV on all rows (report mean +/- std)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = []
for tr, te in skf.split(X, y):
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    pos = int(ytr.sum())
    model = fold_model()
    model = CalibratedClassifierCV(model, method=("isotonic" if pos >= 10 else "sigmoid"), cv=3)
    model.fit(Xtr, ytr)
    p = proba_from_model(model, Xte)
    cv_metrics.append([
        roc_auc_score(yte, p),
        average_precision_score(yte, p),
        brier_score_loss(yte, p),
        log_loss(yte, p, labels=[0,1]),
    ])
cv_metrics = np.array(cv_metrics)
print(f"CV (5-fold) AUROC  mean±std: {cv_metrics[:,0].mean():.3f} ± {cv_metrics[:,0].std():.3f}")
print(f"CV (5-fold) AUPRC  mean±std: {cv_metrics[:,1].mean():.3f} ± {cv_metrics[:,1].std():.3f}")
print(f"CV (5-fold) Brier  mean±std: {cv_metrics[:,2].mean():.3f} ± {cv_metrics[:,2].std():.3f}")
print(f"CV (5-fold) LogLoss mean±std: {cv_metrics[:,3].mean():.3f} ± {cv_metrics[:,3].std():.3f}")

# Group-aware holdout to avoid policy leakage
gss = GroupShuffleSplit(test_size=0.25, random_state=42)
tr_idx, te_idx = next(gss.split(X, y, groups))
Xtr, Xte, ytr, yte = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
pos_tr = int(ytr.sum())

if pos_tr == 0 or pos_tr == len(ytr):
    pconst = float(y.mean()) if y.mean() > 0 else 0.001
    class ConstantModel:
        def __init__(self, p): self.p = float(p)
        def predict_proba(self, X):
            n = len(X); return np.column_stack([np.full(n, 1-self.p), np.full(n, self.p)])
        def decision_function(self, X):
            return np.full(len(X), np.log(self.p/(1-self.p+1e-12)))
    final_model = ConstantModel(pconst)
else:
    final_model = fold_model()
    final_model = CalibratedClassifierCV(final_model, method=("isotonic" if pos_tr >= 10 else "sigmoid"), cv=5)
    final_model.fit(Xtr, ytr)

pte = proba_from_model(final_model, Xte)
print("Holdout AUROC:", roc_auc_score(yte, pte))
print("Holdout AUPRC:", average_precision_score(yte, pte))
print("Holdout Brier:", brier_score_loss(yte, pte))
print("Holdout LogLoss:", log_loss(yte, pte, labels=[0,1]))

joblib.dump((final_model, FEATURES), "models/behavior_model.joblib")
print("Saved -> models/behavior_model.joblib")

