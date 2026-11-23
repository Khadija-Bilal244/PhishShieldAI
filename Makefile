.PHONY: help install test lint repro status push live monitor update rollback registry api up down airflow load

help:          ## list commands
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'
install:       ## install everything (CPU torch)
	pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-dev.txt && pip install -e .
test:          ## unit tests
	pytest --cov=src/phishshield
lint:          ## ruff
	ruff check src tests airflow
repro:         ## run / update the DVC pipeline
	dvc repro
status:        ## what changed?
	dvc status && dvc metrics show
push:          ## data+models -> DagsHub
	dvc push
live:          ## collect today's real OpenPhish + Tranco batch
	python -m phishshield.data.collect_live
monitor:       ## drift check on the latest stream batch (B=path to override)
	python -m phishshield.drift.monitor --batch $${B:-data/stream/batch_07.csv}
update:        ## execute the action chosen by the last drift check
	python -m phishshield.pipeline.update --batch $${B:-data/stream/batch_07.csv}
rollback:      ## previous champion back to Production
	python -m phishshield.registry.promote --rollback
registry:      ## show model versions
	python -m phishshield.registry.promote --status
api:           ## API + web UI on http://localhost:8000
	uvicorn phishshield.api.main:app --reload --port 8000
up:            ## Docker: API + Prometheus + Grafana
	docker compose up -d --build
down:          ## stop Docker stack
	docker compose down
airflow:       ## Airflow on http://localhost:8080 (after `make up`)
	docker compose -f docker-compose.airflow.yml up -d --build
load:          ## traffic for the Grafana demo
	python scripts/load_test.py --n 500
