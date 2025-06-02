import os
import shutil

import torch
from fast_plaid import search


def test():
    """Ensure that the Fast-PLAiD search index can be created and queried correctly."""
    index_name = "test_index"

    # Clean up previous index directory
    if os.path.exists(index_name):
        shutil.rmtree(index_name)
    os.makedirs(index_name, exist_ok=True)

    # Initialise the search index
    index = search.FastPlaid(
        index=index_name,
    )

    documents_embeddings = [torch.randn(300, 128) for _ in range(100)]

    queries_embeddings = torch.randn(10, 30, 128)

    index.create(
        documents_embeddings=documents_embeddings,
        kmeans_niters=4,
    )

    results = index.search(queries_embeddings=queries_embeddings, k=10)

    assert len(results) == 10, (
        f"Expected 10 sets of query results, but got {len(results)}"
    )

    assert all(len(query_res) == 10 for query_res in results), (
        "Expected each query to have 10 results"
    )

    print("✅ Test passed: Results have the correct shape (10, 10).")

    shutil.rmtree(index_name, ignore_errors=True)
