train:
	python scripts/train.py --config configs/medium_lambda.yaml

run-all:
	python scripts/run_experiments.py

test:
	pytest tests/ --cov=src --cov-report=term-missing -v

lint:
	flake8 src/ scripts/ tests/
	black --check src/ scripts/ tests/

format:
	black src/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
