lint:
	cargo clean
	uv pip install torch==2.8.0
	uv run --extra dev pre-commit run --files python/**/**/**.py

install:
	cargo clean
	uv pip install torch==2.8.0
	uv pip install -e ".[dev]"

test:
	cargo clean
	uv run tests/test.py

evaluate:
	uv run benchmark/benchmark.py
