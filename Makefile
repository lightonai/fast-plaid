lint:
	cargo clean
	uv pip install torch==2.9.0
	uv run --extra dev pre-commit run --files python/**/**/**.py

install:
	cargo clean
	uv pip install torch==2.9.0
	uv pip install -e ".[dev]"

test:
	cargo clean
	uv run pytest tests/test.py

evaluate:
	CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 uv run mprof run --interval 0.5 python docs/benchmark/benchmark.py
	mprof plot -o msmarco_usage.png
