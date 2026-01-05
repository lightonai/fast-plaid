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
	uv run python docs/benchmark/benchmark.py
	rm -rf *.dat
	
evaluate-test:
	mprof run --include-children uv run python test.py && mprof plot -o chart_test.png
	rm -rf *.dat	
