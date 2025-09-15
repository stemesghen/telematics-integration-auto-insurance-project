# .PHONY: setup run api

# # Create & activate a venv, then install dependencies
# setup:
# 	python3 -m venv .venv
# 	. .venv/bin/activate && pip install --upgrade pip
# 	. .venv/bin/activate && pip install "fastapi[all]" uvicorn

# # Run the API locally
# run:
# 	. .venv/bin/activate && uvicorn --app-dir src API.fast_api:app --host 0.0.0.0 --port 8080

# make clean:
# 	rm -rf data_ingest data/*.csv models/behavior_model.joblib

# Makefile
.PHONY: help setup run health synth build claims label train preview pipeline clean docker-build docker-run

# -------- Config --------
PY        := python
APP       := API.fast_api:app
UVICORN   := uvicorn --app-dir src $(APP) --host 0.0.0.0 --port 8080
API_KEY  ?= dev-secret

# -------- Help --------
help:
	@echo "Targets:"
	@echo "  setup        Create venv and install requirements.txt"
	@echo "  run          Run the FastAPI server locally (uses API_KEY=$(API_KEY))"
	@echo "  health       Check /healthz"
	@echo "  synth        Generate synthetic trips"
	@echo "  build        Build driver-period features"
	@echo "  claims       Synthesize claims (label prevalence knobs)"
	@echo "  label        Label next-90d"
	@echo "  train        Train behavior model and save models/behavior_model.joblib"
	@echo "  preview      Produce data/pricing_preview.csv using same pricing mapping"
	@echo "  pipeline     Run synth -> build -> claims -> label -> train -> preview"
	@echo "  clean        Remove data, models, and ingest files"
	@echo "  docker-build Build Docker image ubi-api:latest"
	@echo "  docker-run   Run image with ports/volumes/envs mounted"

# -------- Local dev --------
setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt

run:
	. .venv/bin/activate && API_KEY=$(API_KEY) $(UVICORN)

health:
	curl -s http://localhost:8080/healthz | python -m json.tool || true

# -------- Data pipeline (POC) --------
synth:
	. .venv/bin/activate && $(PY) src/sim/synth_trips.py

build:
	. .venv/bin/activate && $(PY) src/sim/build_driver_period.py

claims:
	. .venv/bin/activate && $(PY) src/sim/synth_claims.py --multiplier 1.0 --cap 0.95 --target-prev 0.35 --min-claims 50

label:
	. .venv/bin/activate && $(PY) src/sim/label_next90.py

train:
	. .venv/bin/activate && $(PY) src/model/train_behavior_model.py

preview:
	. .venv/bin/activate && $(PY) src/model/price_telematics.py

pipeline: synth build claims label train preview

clean:
	rm -rf data_ingest/*.jsonl data/*.csv models/*.joblib

# -------- Docker helpers --------
docker-build:
	docker build -t ubi-api:latest .

docker-run:
	docker run -it --rm \
		-p 8080:8080 \
		-e API_KEY=$${API_KEY:-dev-secret} \
		-e BASELINE_PREV=$${BASELINE_PREV:-0.30} \
		-e PRICING_SLOPE=$${PRICING_SLOPE:-0.5} \
		-e FACTOR_MIN=$${FACTOR_MIN:-0.90} \
		-e FACTOR_MAX=$${FACTOR_MAX:-1.10} \
		-v "$$PWD/models:/app/models" \
		-v "$$PWD/data:/app/data" \
		-v "$$PWD/data_ingest:/app/data_ingest" \
		ubi-api:latest
