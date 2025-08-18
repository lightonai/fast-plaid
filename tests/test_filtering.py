import os
import shutil

import torch
from fast_plaid import search


def test_filtering():
    """Test the metadata filtering functionality."""
    index_name = "test_filtering_index"

    if os.path.exists(index_name):
        shutil.rmtree(index_name)

    index = search.FastPlaid(
        index=index_name,
        device="cpu",
    )

    # Create documents with metadata
    documents_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(20)]
    documents_metadata = []

    for i in range(20):
        metadata = {
            "id": i,
            "category": "science" if i < 10 else "technology",
            "year": 2020 + (i % 5),
            "title": f"Document {i}",
        }
        documents_metadata.append(metadata)

    # Create index with metadata
    index.create(
        documents_embeddings=documents_embeddings,
        kmeans_niters=4,
        documents_metadata=documents_metadata,
    )

    queries_embeddings = torch.randn(5, 30, 128, device="cpu")

    # Test 1: No filter (should return all results)
    results_no_filter = index.search(
        queries_embeddings=queries_embeddings, top_k=10, show_progress=False
    )
    assert len(results_no_filter) == 5, "Should have 5 query results"
    assert all(len(query_res) == 10 for query_res in results_no_filter), (
        "Each query should have 10 results without filter"
    )

    # Test 2: Filter by category = 'science' (should return documents 0-9)
    results_science = index.search(
        queries_embeddings=queries_embeddings,
        top_k=10,
        filter_query="metadata->>'category' = 'science'",
        show_progress=False,
    )
    assert len(results_science) == 5, "Should have 5 query results"

    # Check that all returned document IDs are in range 0-9 (science category)
    science_docs = set(range(10))
    for query_results in results_science:
        for doc_id, _ in query_results:
            assert doc_id in science_docs, (
                f"Document {doc_id} should be in science category"
            )

    # Test 3: Filter by category = 'technology' (should return documents 10-19)
    results_tech = index.search(
        queries_embeddings=queries_embeddings,
        top_k=10,
        filter_query="metadata->>'category' = 'technology'",
        show_progress=False,
    )
    assert len(results_tech) == 5, "Should have 5 query results"

    # Check that all returned document IDs are in range 10-19 (technology category)
    tech_docs = set(range(10, 20))
    for query_results in results_tech:
        for doc_id, _ in query_results:
            assert doc_id in tech_docs, (
                f"Document {doc_id} should be in technology category"
            )

    # Test 4: Filter by year >= 2022
    results_recent = index.search(
        queries_embeddings=queries_embeddings,
        top_k=10,
        filter_query="CAST(metadata->>'year' AS INTEGER) >= 2022",
        show_progress=False,
    )
    assert len(results_recent) == 5, "Should have 5 query results"

    # Test 5: Complex filter (science AND recent years)
    results_science_recent = index.search(
        queries_embeddings=queries_embeddings,
        top_k=10,
        filter_query="metadata->>'category' = 'science' AND CAST(metadata->>'year' AS INTEGER) >= 2022",
        show_progress=False,
    )
    assert len(results_science_recent) == 5, "Should have 5 query results"

    print("✅ All filtering tests passed!")

    # Test update with metadata
    new_documents_embeddings = [torch.randn(40, 128, device="cpu") for _ in range(5)]
    new_metadata = []
    for i in range(5):
        metadata = {
            "id": 20 + i,
            "category": "biology",
            "year": 2025,
            "title": f"New Document {20 + i}",
        }
        new_metadata.append(metadata)

    index.update(
        documents_embeddings=new_documents_embeddings, documents_metadata=new_metadata
    )

    # Test filtering after update
    results_biology = index.search(
        queries_embeddings=queries_embeddings,
        top_k=10,
        filter_query="metadata->>'category' = 'biology'",
        show_progress=False,
    )
    assert len(results_biology) == 5, "Should have 5 query results after update"

    print("✅ Update with filtering tests passed!")

    shutil.rmtree(index_name, ignore_errors=True)


if __name__ == "__main__":
    test_filtering()
