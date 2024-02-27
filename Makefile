.PHONY: start
start:
	poetry run python cmdh/main.py $(ARGS)

.PHONY: lint
lint:
	poetry run python -m mypy **/*.py

.PHONY: format
format:
	poetry run python -m black **/*.py
