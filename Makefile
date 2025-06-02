lint:
	uv run pre-commit run --files python/**/**.py pyproject.toml Makefile

install:
	uv run pip install -e ".[dev]"

test:
	uv run benchmark/benchmark.py
