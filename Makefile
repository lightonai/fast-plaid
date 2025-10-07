lint:
	cargo clean
	uv pip install torch==2.8.0
	uv run --extra dev pre-commit run --files python/**/**/**.py

install:
	cargo clean
	pip install torch==2.8.0
	pip install -e ".[dev]"

test:
	cargo clean
	uv run tests/test.py

evaluate:
	CUDA_VISIBLE_DEVICES= mprof run test.py --dataset scifact

plot:
	mprof plot -o memory_usage_plot.png

# {'map': np.float64(0.7053047493871797), 'ndcg@10': np.float64(0.7426470677876745), 'ndcg@100': np.float64(0.7530787384591732), 'recall@10': np.float64(0.855), 'recall@100': np.float64(0.8936666666666667)}