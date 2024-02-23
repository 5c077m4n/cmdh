start:
	poetry run python cmdh/main.py

lint:
	poetry run python -m mypy **/*.py

format:
	poetry run python -m black **/*.py
