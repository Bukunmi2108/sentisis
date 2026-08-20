.PHONY: data train eval serve requirements up down logs redis lint test test-unit test-integration

data:
	python -m model.scripts.prepare_data

train:
	python -m nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=1800 model/notebooks/train_baseline.ipynb

eval:
	python -m model.scripts.evaluate_all

serve:
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

requirements:
	uv export --no-dev --no-emit-project --no-hashes \
	  --format requirements-txt > requirements.txt

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

redis:
	docker run -d --rm -p 6379:6379 --name sentisis-redis redis:7-alpine

lint:
	ruff format --check .
	ruff check .
	pyright

test:
	pytest

test-unit:
	pytest tests/unit

test-integration:
	pytest tests/integration
