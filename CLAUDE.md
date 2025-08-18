# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastPlaid is a high-performance multi-vector search engine built in Rust with Python bindings via PyO3. It implements the PLAID (Per-Token Late Interaction Dense Search) algorithm for ColBERT-style late interaction retrieval, providing faster indexing and query speeds compared to the original PLAID implementation.

## Build System & Dependencies

This is a hybrid Rust/Python project using:
- **Rust**: Core search engine implementation using `tch` (PyTorch bindings) for tensor operations
- **Python**: High-level API and utilities using Maturin for building the extension module
- **Build tool**: Uses `uv` for Python package management and Maturin for Rust-Python binding compilation

### Key Dependencies
- `tch = "0.20.0"` - PyTorch bindings for Rust
- `pyo3 = "0.24.0"` - Python bindings
- `fastkmeans` - K-means clustering for centroids
- `torch >= 2.6.0` - PyTorch for embeddings
- `duckdb >= 1.1.0` - SQL database for metadata filtering (both Rust and Python)
- `serde_json` - JSON handling for document metadata

## Development Commands

### Building & Installation
```bash
make install  # Clean build and install with dev dependencies
```

### Linting
```bash
make lint     # Run pre-commit hooks on Python files
```

### Testing
```bash
make test     # Run benchmark tests
python tests/test.py  # Run basic functionality tests
pytest        # Run pytest with configured options
```

### Evaluation
```bash
make evaluate # Run full benchmark evaluation
```

### Manual Development
```bash
# Development build
cargo build
# or for Python module
uv run pip install -e ".[dev]"

# Run tests individually
uv run python tests/test.py
```

## Architecture

### Core Components

**Rust Layer (`rust/`):**
- `lib.rs` - Main PyO3 module with Python function bindings
- `index/` - Index creation and update logic (`create.rs`, `update.rs`)  
- `search/` - Search functionality (`search.rs`, `load.rs`, `tensor.rs`, `padding.rs`)
- `utils/` - Utility functions including residual codecs

**Python Layer (`python/fast_plaid/`):**
- `search/fast_plaid.py` - Main `FastPlaid` class providing high-level API
- `evaluation/` - Benchmarking and evaluation utilities

**Database Layer:**
- `utils/metadata_db.rs` - DuckDB interface for document metadata storage and filtering
- `metadata.duckdb` - SQLite-compatible database file stored in index directory

### Key Data Flow

1. **Index Creation**: Python collects document embeddings → computes K-means centroids → Rust creates quantized index → stores metadata in DuckDB (if provided)
2. **Search**: Python prepares queries → DuckDB pre-filters centroids (if filter provided) → Rust performs vector search with late interaction → DuckDB filters final results → returns ranked results  
3. **Update**: Rust adds new documents to existing index without recomputing centroids → updates DuckDB with new metadata (if provided)

### Device Handling
- Supports CPU and CUDA devices
- Multi-GPU support with automatic parallelization
- Device specification: `"cpu"`, `"cuda"`, `"cuda:0"`, or list `["cuda:0", "cuda:1"]`

### Index Structure
Indexes are stored as directory containing:
- `metadata.json` - Index configuration and statistics
- `centroids.npy` - K-means centroids for vector quantization
- `*.codes.npy`, `*.residuals.npy` - Quantized vectors and residuals
- `ivf.npy`, `ivf_lengths.npy` - Inverted file structure
- `doclens.*.json` - Document length metadata
- `metadata.duckdb` - SQLite-compatible database with document metadata and centroid mappings (optional)

## Testing Strategy

- `tests/test.py` - Basic functionality test creating small index and verifying search results
- `tests/test_filtering.py` - Metadata filtering functionality tests
- `benchmark/` - Performance benchmarking against reference datasets
- Pytest configuration excludes slow/web tests by default (use `-m slow` or `-m web` to include)

## Key Configuration Parameters

**Index Creation:**
- `kmeans_niters` - K-means iterations (default: 4)
- `max_points_per_centroid` - Clustering constraint (default: 256)
- `nbits` - Quantization bits (default: 4)
- `n_samples_kmeans` - Samples for clustering (auto-calculated if None)

**Search:**
- `batch_size` - Internal processing batch size (default: 2^18)
- `n_full_scores` - Candidates for re-ranking (default: 8192)  
- `n_ivf_probe` - IVF probes for recall vs speed tradeoff (default: 8)
- `top_k` - Results to return per query (default: 10)

## Metadata Filtering

FastPlaid supports metadata-based filtering using DuckDB:

### Usage
```python
# Create index with metadata
documents_metadata = [
    {"category": "science", "year": 2023, "author": "Alice"},
    {"category": "tech", "year": 2024, "author": "Bob"},
    # ... more documents
]

fast_plaid.create(
    documents_embeddings=embeddings,
    documents_metadata=documents_metadata
)

# Search with filtering
results = fast_plaid.search(
    queries_embeddings=queries,
    filter_query="metadata->>'category' = 'science' AND CAST(metadata->>'year' AS INTEGER) >= 2023"
)
```

### Filter Query Syntax
- Uses DuckDB SQL syntax with JSON operators
- `metadata->>'key'` extracts string values from JSON metadata
- `CAST(metadata->>'key' AS INTEGER/FLOAT)` for numeric comparisons
- Standard SQL operators: `=`, `!=`, `<`, `>`, `>=`, `<=`, `AND`, `OR`, `IN`, etc.
- JSON path expressions: `metadata->'nested'->>'field'`

### Performance Notes
- **Centroid Pre-filtering**: Only searches centroids containing documents that match the filter
- **Result Post-filtering**: Final results are filtered by metadata before returning
- Metadata database is optional and backwards compatible
- Uses efficient indexed lookups for common query patterns

## Development Notes

- Torch library path resolution is handled automatically via `_load_torch_path()`
- Multi-GPU processing uses either multiprocessing (spawn) or joblib parallel execution
- Index updates are incremental but don't recompute centroids (may reduce accuracy over time)
- All tensor operations use `float16` for memory efficiency
- Progress bars shown during search operations by default
- DuckDB database is created automatically when metadata is provided during index creation