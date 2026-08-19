.PHONY: data train eval lint test

data:
	python -m model.scripts.prepare_data

train:
	python -m nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=1800 model/notebooks/train_baseline.ipynb

eval:
	python -m model.scripts.evaluate_all

lint:
	ruff format --check .
	ruff check .
	pyright

test:
	pytest
