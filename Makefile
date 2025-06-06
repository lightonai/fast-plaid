lint:
	cargo clean
	uv run --extra dev pre-commit run --files python/**/**.py pyproject.toml Makefile

install:
	cargo clean
	uv pip install torch==2.7.0
	uv run pip install -e ".[dev]"

test:
	cargo clean
	uv run benchmark/benchmark.py

evaluate:
	uv run benchmark/launch.py
