
# Telematics UBI POC

End-to-end **usage-based insurance (UBI)** proof-of-concept: simulate telematics → aggregate to monthly driver-period features → train a calibrated risk model → map to a capped pricing factor → expose **/score** and **/pricing** APIs (plus an optional dashboard).


## Table of Contents

* [Quick Start](#quick-start)
* [Run the Pipeline](#run-the-pipeline)
* [Call the API](#call-the-api)
* [Evaluate](#evaluate)
* [Makefile (convenience targets)](#makefile-convenience-targets)
* [Features — baseline & engineered](#features--baseline--engineered)
* [Technical Requirements — how this POC meets them](#technical-requirements--how-this-poc-meets-them)
* [Evaluation Criteria — how this POC addresses them](#evaluation-criteria--how-this-poc-addresses-them)
* [Bin Helpers (optional)](#bin-helpers-optional)
* [Docker & Compose (optional)](#docker--compose-optional)
* [Repo Layout (reference)](#repo-layout-reference)
* [Troubleshooting](#troubleshooting)

---

## How to Run — pick one

> Prereqs:
> • **Docker Desktop** for Docker options
> • **Python 3.10+** for local/Makefile/bin options

### Option 1 — Easiest (Docker Compose: API + UI)

```bash
docker compose up --build
```

* API: [http://localhost:8080](http://localhost:8080)  (Swagger at `/docs`)
* UI (dashboard): [http://localhost:8501](http://localhost:8501)
* Shared artifacts are mounted: `./models`, `./data`, `./data_ingest`

**Regenerate data/model (optional):**

```bash
docker compose exec api bash -lc "
  python src/sim/synth_trips.py &&
  python src/sim/build_driver_period.py &&
  python src/sim/synth_claims.py --multiplier 1.0 --cap 0.95 --target-prev 0.35 --min-claims 50 &&
  python src/sim/label_next90.py &&
  python src/model/train_behavior_model.py &&
  python src/model/price_telematics.py
"
```

**Stop:** `Ctrl+C` (in the compose window) or `docker compose down`

---

### Option 2 — Local dev (Makefile)

```bash
make setup                 # create venv + install deps
make run API_KEY=dev-secret  # start API on :8080
make pipeline              # synth → build → claims → label → train → preview
```

Health:

```bash
curl -s http://localhost:8080/healthz | python -m json.tool
```

Dashboard:

```bash
pip install streamlit
streamlit run src/user_interface/user_dashboard.py
```

---

### Option 3 — Local dev (bin/ helpers)

```bash
chmod +x bin/*
API_KEY=dev-secret ./bin/api        # run API with auto-reload
./bin/pipeline                      # end-to-end data → model → preview
./bin/dashboard                     # launch Streamlit dashboard
./bin/ingest_demo                   # simulate & POST telemetry
./bin/score_example                 # sample /score
./bin/price_example driver_00001    # sample /pricing
```

---

### Option 4 — Dockerfile only (API container, no UI)

```bash
docker build -t ubi-api:latest .
mkdir -p models data data_ingest
docker run -it --rm -p 8080:8080 \
  -e API_KEY=dev-secret \
  -e BASELINE_PREV=0.30 -e PRICING_SLOPE=0.5 -e FACTOR_MIN=0.90 -e FACTOR_MAX=1.10 \
  -v "$PWD/models:/app/models" -v "$PWD/data:/app/data" -v "$PWD/data_ingest:/app/data_ingest" \
  ubi-api:latest
```

---

## Quick Start (detailed commands)

### 1) Local setup (Python 3.10+ recommended; 3.11 OK)

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export API_KEY=dev-secret
```

### 2) Run the API (Terminal A)

```bash
uvicorn --app-dir src API.fast_api:app --host 0.0.0.0 --port 8080 --reload
```

Health check:

```bash
curl -s http://127.0.0.1:8080/healthz | python -m json.tool
```

> **Tip:** The API auto-loads `models/behavior_model.joblib` (and its saved feature names) when present.

---

## Run the Pipeline (Terminal B)

**Simulate live ingestion** to `data_ingest/`:

```bash
python src/sim/generate_stream.py \
  --drivers 5 --days 1 --trips-per-day 2 --hz 1 \
  --post http://127.0.0.1:8080/ingest/telemetry \
  --api-key dev-secret
```

```bash
python src/sim/synth_trips.py
python src/sim/build_driver_period.py
python src/sim/synth_claims.py --multiplier 1.0 --cap 0.95 --target-prev 0.35 --min-claims 50
python src/sim/label_next90.py
python src/model/train_behavior_model.py
python src/model/price_telematics.py
```

**Outputs:**

* `data/trips.csv` — simulated trips
* `data/driver_period.csv` — monthly features per policy
* `data/claims.csv` — synthetic claims
* `data/driver_period_labeled.csv` — features + next-90d labels
* `models/behavior_model.joblib` — saved *(model, FEATURE\_NAMES)*
* `data/pricing_preview.csv` — risk and capped factor preview

**Dashboard:**

```bash
pip install streamlit
streamlit run src/user_interface/user_dashboard.py   # http://localhost:8501
```

---

## Call the API

Auth header (POC): `x-api-key: dev-secret`

```bash
# /score – send any/all features; missing default to 0
curl -s -X POST "http://127.0.0.1:8080/score" \
  -H "x-api-key: dev-secret" -H "Content-Type: application/json" \
  -d '{"exposure_miles":60,"trip_ct":6,"harsh_brake_per_100mi":150,"duration_s":3600,
       "avg_overspeed_ratio":0.45,"night_miles_ratio":0.10,"phone_usage_per_hr":2.0,
       "mean_speed_mps":12.0,"miles_per_trip":10.0,"speed_var_across_trips":0.005}' \
  | python -m json.tool

# /pricing – uses latest features for that driver from data/driver_period.csv
curl -s "http://127.0.0.1:8080/pricing/driver_00001" \
  -H "x-api-key: dev-secret" | python -m json.tool
```

---
---

## Features — baseline & engineered

Below is a quick glossary of every feature used in modeling and pricing. Each item includes **what it is**, **how it’s computed**, and **why it matters for risk**.

### Baseline six (used by API/pricing)

1. **`exposure_miles`** *(miles)*
   **What:** Total miles the driver logged in the month.
   **How:** `exposure_miles = Σ_i miles_i`
   **Risk intuition:** More miles → more time at risk (exposure). Often positively related to claim frequency; pricing caps/smoothing keep this stable.

2. **`trip_ct`** *(count)*
   **What:** Number of distinct trips in the month.
   **How:** `trip_ct = count(trip_id)`
   **Risk intuition:** Many short trips (stop-and-go) can raise event risk; interacts with `miles_per_trip`.

3. **`harsh_brake_per_100mi`** *(events per 100 miles)*
   **What:** Frequency of harsh braking normalized by distance.
   **How:** `harsh_brake_per_100mi = (Σ_i harsh_brake_ct_i) / max(exposure_miles, 1e-6) * 100`
   **Risk intuition:** Strong positive signal—abrupt decels capture tail-risk moments.

4. **`duration_s`** *(seconds)*
   **What:** Total driving time in the month.
   **How:** `duration_s = Σ_i duration_s_i`
   **Risk intuition:** Another exposure proxy; helpful when miles are noisy.

5. **`overspeed_ratio`** *(0–1)*
   **What:** Share of time spent above the posted speed limit.
   **How (weighted mean):**
   `overspeed_ratio = (Σ_i overspeed_ratio_i * duration_s_i) / (Σ_i duration_s_i)`
   **Risk intuition:** Clear positive signal; more time overspeeding → higher crash likelihood/severity.

6. **`night_ratio`** *(0–1)*
   **What:** Share of driving occurring at night.
   **How (miles-weighted):**
   `night_miles = Σ_i night_ratio_i * miles_i`
   `night_ratio = night_miles / max(exposure_miles, 1e-6)`
   **Risk intuition:** Night driving has reduced visibility and fatigue risk—generally increases frequency/severity.

### Engineered extras (for lift + diagnostics)

7. **`phone_usage_per_hr`** *(events per hour)*
   **What:** Normalized phone-use events per driving hour.
   **How:** `phone_usage_per_hr = (Σ_i phone_usage_ct_i) / (Σ_i duration_s_i / 3600)`
   **Risk intuition:** Distraction is a strong causal factor—expect positive relationship.

8. **`mean_speed_mps`** *(meters/second)*
   **What:** Average cruising speed across trips, weighted by miles.
   **How:** `mean_speed_mps = (Σ_i avg_speed_mps_i * miles_i) / (Σ_i miles_i)`
   **Risk intuition:** Non-monotone; very low speeds can indicate congestion; very high speeds increase severity. Useful for GBM; LR may need transforms/interactions.

9. **`miles_per_trip`** *(miles)*
   **What:** Typical trip length.
   **How:** `miles_per_trip = exposure_miles / max(trip_ct, 1)`
   **Risk intuition:** Many very short trips → more merges/turns/intersections (risk); very long trips → fatigue (risk).

10. **`speed_var_across_trips`** *(mps², variance)*
    **What:** Variability of average speeds across trips in the month.
    **How:** `var({avg_speed_mps_i})`
    **Risk intuition:** Captures heterogeneous patterns (urban vs highway mix); higher variance may imply inconsistent context or behavior.

11. **`night_miles_ratio`** *(0–1)*
    **What:** Fraction of miles driven at night (miles-weighted).
    **How:** `night_miles_ratio = night_miles / exposure_miles`
    **Risk intuition:** Same as `night_ratio`; this is the more robust, miles-weighted version.

> Weighted engineered forms (`avg_overspeed_ratio`, `night_miles_ratio`) feed the baseline fields so downstream code is consistent.

---

## Evaluate

**Modeling approach:** Calibrated Logistic Regression (transparent & stable). Calibration (isotonic/Platt) improves Brier/probability quality.

**Latest example metrics (your run may vary):**

* **5-fold CV:** AUROC **0.678 ± 0.124**, AUPRC **0.719 ± 0.108**, Brier **0.225 ± 0.053**, LogLoss **0.988 ± 0.802**
* **Group holdout:** AUROC **0.788**, AUPRC **0.868**, Brier **0.191**, LogLoss **0.575**

**Pricing mapping (also used by preview script):**

```
factor = clip(1 + PRICING_SLOPE * (risk_p - BASELINE_PREV), FACTOR_MIN, FACTOR_MAX)
```

Defaults: `BASELINE_PREV=0.30`, `PRICING_SLOPE=0.5`, `FACTOR_MIN=0.90`, `FACTOR_MAX=1.10`

---

## Bin Helpers (optional)

After `chmod +x bin/*`:

| Script             | Purpose                                   | Example                                 |
| ------------------ | ----------------------------------------- | --------------------------------------- |
| `bin/api`          | Run API locally with auto-reload          | `API_KEY=dev-secret ./bin/api`          |
| `bin/pipeline`     | End-to-end data → model → pricing preview | `./bin/pipeline`                        |
| `bin/ingest_demo`  | Simulate & POST telemetry                 | `API_KEY=dev-secret ./bin/ingest_demo`  |
| `bin/dashboard`    | Launch Streamlit dashboard                | `./bin/dashboard`                       |
| `bin/docker-build` | Build container image                     | `TAG=ubi-api:latest ./bin/docker-build` |
| `bin/docker-run`   | Run API container with mounted volumes    | `API_KEY=dev-secret ./bin/docker-run`   |
| `bin/clean`        | Remove generated artifacts                | `./bin/clean`                           |
| `bin/zip`          | Create submission archive                 | `./bin/zip Project_Name`                |

---

## Docker & Compose (optional)

**Dockerfile:**

```bash
docker build -t ubi-api:latest .
mkdir -p models data data_ingest
docker run -it --rm -p 8080:8080 \
  -e API_KEY=dev-secret \
  -e BASELINE_PREV=0.30 -e PRICING_SLOPE=0.5 -e FACTOR_MIN=0.90 -e FACTOR_MAX=1.10 \
  -v "$PWD/models:/app/models" -v "$PWD/data:/app/data" -v "$PWD/data_ingest:/app/data_ingest" \
  ubi-api:latest
```

**docker-compose.yml (API + UI):**

```yaml
services:
  api:
    build: .
    container_name: ubi-api
    ports: ["8080:8080"]
    environment:
      API_KEY: "dev-secret"
      BASELINE_PREV: "0.30"
      PRICING_SLOPE: "0.5"
      FACTOR_MIN: "0.90"
      FACTOR_MAX: "1.10"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./data_ingest:/app/data_ingest

  ui:
    image: python:3.11-slim
    container_name: ubi-ui
    working_dir: /app
    command: bash -lc "pip install --no-cache-dir streamlit pandas && streamlit run src/user_interface/user_dashboard.py --server.address 0.0.0.0 --server.port 8501"
    ports: ["8501:8501"]
    volumes: [".:/app"]
    depends_on: [api]
```

Run:

```bash
docker compose up --build
```

---

## Quick checks (any option)

* Health:

  ```bash
  curl -s http://localhost:8080/healthz | python -m json.tool
  ```
* Pricing:

  ```bash
  curl -s "http://localhost:8080/pricing/driver_00001" -H "x-api-key: dev-secret" | python -m json.tool
  ```
* Score:

  ```bash
  curl -s -X POST "http://localhost:8080/score" \
    -H "x-api-key: dev-secret" -H "Content-Type: application/json" \
    -d '{"exposure_miles":60,"trip_ct":6,"harsh_brake_per_100mi":150,"duration_s":3600,
         "avg_overspeed_ratio":0.45,"night_miles_ratio":0.10,"phone_usage_per_hr":2.0,
         "mean_speed_mps":12.0,"miles_per_trip":10.0,"speed_var_across_trips":0.005}' \
    | python -m json.tool
  ```

---

## Ports / Troubleshooting

* API uses **8080**, UI uses **8501**. If busy, stop the other process or change port mappings.
* `model not loaded` → run the pipeline to create `models/behavior_model.joblib`.
* `404` from `/pricing/{policy_id}` → ensure the policy exists in `data/driver_period.csv`.
* Class collapse during training → generate more data or tune `--target-prev` / `--min-claims`.

---
