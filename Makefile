.PHONY: data lint test

data:
	python -m model.scripts.prepare_data

lint:
	ruff format --check .
	ruff check .
	pyright

test:
	pytest
