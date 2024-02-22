start:
	poetry run python cmdh/main.py

lint:
	poetry run python -m mypy cmdh/main.py
