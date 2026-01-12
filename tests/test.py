import os
import shutil
from datetime import date

import pytest
import torch
from fast_plaid import filtering, search


@pytest.fixture
def test_index_path(tmp_path):
    """Create a temporary index path for testing."""
    import gc

    index_path = str(tmp_path / "test_index")
    os.makedirs(index_path, exist_ok=True)
    yield index_path
    # Force garbage collection to release memory-mapped file handles on Windows
    gc.collect()
    # Cleanup
    if os.path.exists(index_path):
        shutil.rmtree(index_path)


@pytest.fixture
def fast_plaid_index(test_index_path):
    """Create a FastPlaid instance for testing."""
    return search.FastPlaid(index=test_index_path, device="cpu")


class TestBasicCreateAndSearch:
    """Tests for basic index creation and search functionality."""

    def test_create_and_search_basic(self, test_index_path):
        """Ensure that the Fast-PLAiD search index can be created and queried correctly."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(300, 128, device="cpu") for _ in range(100)]
        queries_embeddings = torch.randn(10, 30, 128, device="cpu")

        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
        results = index.search(queries_embeddings=queries_embeddings, top_k=10)

        assert len(results) == 10, (
            f"Expected 10 sets of query results, but got {len(results)}"
        )
        assert all(len(query_res) == 10 for query_res in results), (
            "Expected each query to have 10 results"
        )

    def test_create_with_uniform_length_documents(self, test_index_path):
        """Test creating index with uniform length documents."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        # Create documents with uniform token counts (using list format)
        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(50)]
        queries_embeddings = torch.randn(5, 30, 128, device="cpu")

        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
        results = index.search(queries_embeddings=queries_embeddings, top_k=5)

        assert len(results) == 5, "Expected 5 sets of query results"
        assert all(len(query_res) == 5 for query_res in results), (
            "Expected each query to have 5 results"
        )

    def test_small_index(self, test_index_path):
        """Test creating a small index with few documents."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(50, 64, device="cpu") for _ in range(10)]
        queries_embeddings = torch.randn(3, 20, 64, device="cpu")

        index.create(documents_embeddings=documents_embeddings, kmeans_niters=2)
        results = index.search(queries_embeddings=queries_embeddings, top_k=5)

        assert len(results) == 3, "Expected 3 sets of query results"
        assert all(len(query_res) == 5 for query_res in results), (
            "Expected each query to have 5 results"
        )

    def test_variable_length_documents(self, test_index_path):
        """Test creating index with variable length document embeddings."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        # Create documents with varying token counts
        documents_embeddings = [
            torch.randn(50, 128, device="cpu"),
            torch.randn(200, 128, device="cpu"),
            torch.randn(100, 128, device="cpu"),
            torch.randn(30, 128, device="cpu"),
            torch.randn(500, 128, device="cpu"),
        ] * 10  # 50 documents total

        queries_embeddings = torch.randn(5, 40, 128, device="cpu")

        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
        results = index.search(queries_embeddings=queries_embeddings, top_k=10)

        assert len(results) == 5, "Expected 5 sets of query results"
        assert all(len(query_res) == 10 for query_res in results), (
            "Expected each query to have 10 results"
        )


class TestUpdate:
    """Tests for index update functionality."""

    def test_update_adds_documents(self, test_index_path):
        """Test that updating an index adds new documents."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            # Create initial index with 50 documents
            initial_embeddings = [
                torch.randn(100, 128, device="cpu") for _ in range(50)
            ]
            index.create(documents_embeddings=initial_embeddings, kmeans_niters=4)

            # Search should find documents 0-49
            queries = torch.randn(2, 30, 128, device="cpu")
            initial_results = index.search(queries_embeddings=queries, top_k=10)

            for query_results in initial_results:
                for doc_id, _ in query_results:
                    assert 0 <= doc_id < 50, (
                        f"Document ID {doc_id} out of initial range"
                    )

            # Update with 50 more documents
            new_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(50)]
            index.update(documents_embeddings=new_embeddings)

            # Search again - should now be able to find documents 0-99
            updated_results = index.search(queries_embeddings=queries, top_k=50)

            # Verify we can find documents in the full range
            all_doc_ids = set()
            for query_results in updated_results:
                for doc_id, _ in query_results:
                    all_doc_ids.add(doc_id)
                    assert 0 <= doc_id < 100, (
                        f"Document ID {doc_id} out of updated range"
                    )
        finally:
            # Ensure index is closed to release file handles on Windows
            index.close()

    def test_multiple_updates(self, test_index_path):
        """Test multiple sequential updates to the index."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            # Create initial index
            initial_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(20)]
            index.create(documents_embeddings=initial_embeddings, kmeans_niters=4)

            # Perform multiple updates
            for batch_idx in range(3):
                update_embeddings = [
                    torch.randn(50, 128, device="cpu") for _ in range(10)
                ]
                index.update(documents_embeddings=update_embeddings)

            # Should now have 20 + 3*10 = 50 documents
            queries = torch.randn(2, 30, 128, device="cpu")
            results = index.search(queries_embeddings=queries, top_k=25)

            for query_results in results:
                for doc_id, _ in query_results:
                    assert 0 <= doc_id < 50, f"Document ID {doc_id} out of range"
        finally:
            # Ensure index is closed to release file handles on Windows
            index.close()

    def test_update_delete_update_with_metadata(self, test_index_path):
        """Test update-delete-update sequence with metadata.

        Ensures buffer is properly managed to prevent phantom documents.
        """
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            embedding_dim = 128

            # Create initial documents with metadata
            initial_embeddings = [torch.randn(10, embedding_dim) for _ in range(3)]
            initial_metadata = [
                {"name": "Alice", "category": "A", "join_date": date(2023, 5, 17)},
                {"name": "Bob", "category": "B", "join_date": date(2021, 6, 21)},
                {"name": "Alex", "category": "A", "join_date": date(2023, 8, 1)},
            ]
            index.create(
                documents_embeddings=initial_embeddings, metadata=initial_metadata
            )
            random_query = torch.randn(1, 10, embedding_dim)

            # Verify initial state
            assert len(filtering.get(index=index.index)) == 3, (
                "Expected 3 documents after initial creation"
            )
            assert len(index.search(random_query, top_k=10)[0]) == 3, (
                "Expected 3 documents after initial creation"
            )

            # First update
            new_embeddings = [torch.randn(10, embedding_dim) for _ in range(1)]
            new_metadata = [
                {"name": "Charlie", "category": "B", "join_date": date(2020, 3, 15)},
            ]
            index.update(documents_embeddings=new_embeddings, metadata=new_metadata)

            assert len(filtering.get(index=index.index)) == 4, (
                "Expected 4 documents after update"
            )
            search_results = index.search(random_query, top_k=10)[0]
            assert len(search_results) == 4, (
                f"Expected 4 documents after update, got {len(search_results)}"
            )

            # Delete the last document
            index.delete(subset=[3])
            assert len(filtering.get(index=index.index)) == 3, (
                "Expected 3 documents after deletion"
            )
            search_results = index.search(random_query, top_k=10)[0]
            assert len(search_results) == 3, (
                f"Expected 3 documents after deletion, got {len(search_results)}"
            )

            # Second update - this is where the bug occurred
            index.update(documents_embeddings=new_embeddings, metadata=new_metadata)

            assert len(filtering.get(index=index.index)) == 4, (
                "Expected 4 documents after second update"
            )
            search_results = index.search(random_query, top_k=10)[0]

            # Verify that only valid document IDs are returned (0, 1, 2, 3)
            doc_ids = {doc_id for doc_id, _ in search_results}
            assert doc_ids.issubset({0, 1, 2, 3}), (
                f"Found invalid document IDs: {doc_ids - {0, 1, 2, 3}}"
            )

            assert len(search_results) == 4, (
                f"Expected 4 documents after second update, got {len(search_results)}"
            )
        finally:
            index.close()


class TestDelete:
    """Tests for index delete functionality."""

    def test_delete_single_document(self, test_index_path):
        """Test deleting a single document from the index."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(20)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        # Delete document 5
        index.delete(subset=[5])

        # Search and verify document 5 is not returned
        queries = torch.randn(5, 30, 128, device="cpu")
        results = index.search(queries_embeddings=queries, top_k=19)

        for query_results in results:
            for doc_id, _ in query_results:
                # After deletion, remaining docs are re-indexed 0-18
                assert 0 <= doc_id < 19, (
                    f"Document ID {doc_id} out of expected range (0-18)"
                )

    def test_delete_multiple_documents(self, test_index_path):
        """Test deleting multiple documents from the index."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(30)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        # Delete documents 0, 5, 10, 15
        index.delete(subset=[0, 5, 10, 15])

        # Should now have 26 documents (30 - 4)
        queries = torch.randn(3, 30, 128, device="cpu")
        results = index.search(queries_embeddings=queries, top_k=26)

        for query_results in results:
            for doc_id, _ in query_results:
                # After deletion, remaining docs are re-indexed 0-25
                assert 0 <= doc_id < 26, (
                    f"Document ID {doc_id} out of expected range (0-25)"
                )


class TestSubsetFiltering:
    """Tests for subset filtering during search."""

    def test_search_with_single_subset(self, test_index_path):
        """Test searching within a single subset applied to all queries."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(50)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        # Search only within documents [0, 5, 10, 15, 20]
        subset = [0, 5, 10, 15, 20]
        queries = torch.randn(3, 30, 128, device="cpu")
        results = index.search(queries_embeddings=queries, top_k=5, subset=subset)

        assert len(results) == 3, "Expected 3 sets of query results"

        for query_results in results:
            for doc_id, _ in query_results:
                assert doc_id in subset, f"Document ID {doc_id} not in subset {subset}"

    def test_search_with_per_query_subset(self, test_index_path):
        """Test searching with different subsets for each query."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(50)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        # Different subset for each query
        subsets = [
            [0, 1, 2, 3, 4],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
        ]
        queries = torch.randn(3, 30, 128, device="cpu")
        results = index.search(queries_embeddings=queries, top_k=5, subset=subsets)

        assert len(results) == 3, "Expected 3 sets of query results"

        for query_idx, query_results in enumerate(results):
            for doc_id, _ in query_results:
                assert doc_id in subsets[query_idx], (
                    f"Query {query_idx}: Document ID {doc_id} not in subset {subsets[query_idx]}"
                )


class TestMetadataFiltering:
    """Tests for metadata filtering using SQLite."""

    def test_create_with_metadata(self, test_index_path):
        """Test creating index with metadata."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(5)]
        metadata = [
            {"name": "doc1", "category": "A", "score": 0.9},
            {"name": "doc2", "category": "B", "score": 0.8},
            {"name": "doc3", "category": "A", "score": 0.7},
            {"name": "doc4", "category": "B", "score": 0.6},
            {"name": "doc5", "category": "A", "score": 0.5},
        ]

        index.create(
            documents_embeddings=documents_embeddings,
            metadata=metadata,
            kmeans_niters=2,
        )

        # Verify metadata was stored
        all_metadata = filtering.get(index=test_index_path)
        assert len(all_metadata) == 5, "Expected 5 metadata entries"

    def test_filtering_where_basic(self, test_index_path):
        """Test basic filtering with where clause."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(5)]
        metadata = [
            {"name": "doc1", "category": "A", "score": 0.9},
            {"name": "doc2", "category": "B", "score": 0.8},
            {"name": "doc3", "category": "A", "score": 0.7},
            {"name": "doc4", "category": "B", "score": 0.6},
            {"name": "doc5", "category": "A", "score": 0.5},
        ]

        index.create(
            documents_embeddings=documents_embeddings,
            metadata=metadata,
            kmeans_niters=2,
        )

        # Filter by category
        category_a_ids = filtering.where(
            index=test_index_path, condition="category = ?", parameters=("A",)
        )
        assert set(category_a_ids) == {0, 2, 4}, (
            f"Expected [0, 2, 4], got {category_a_ids}"
        )

        category_b_ids = filtering.where(
            index=test_index_path, condition="category = ?", parameters=("B",)
        )
        assert set(category_b_ids) == {1, 3}, f"Expected [1, 3], got {category_b_ids}"

    def test_filtering_where_with_numeric_condition(self, test_index_path):
        """Test filtering with numeric conditions."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(5)]
        metadata = [
            {"name": "doc1", "score": 0.9},
            {"name": "doc2", "score": 0.8},
            {"name": "doc3", "score": 0.7},
            {"name": "doc4", "score": 0.6},
            {"name": "doc5", "score": 0.5},
        ]

        index.create(
            documents_embeddings=documents_embeddings,
            metadata=metadata,
            kmeans_niters=2,
        )

        high_score_ids = filtering.where(
            index=test_index_path, condition="score >= ?", parameters=(0.7,)
        )
        assert set(high_score_ids) == {0, 1, 2}, (
            f"Expected [0, 1, 2], got {high_score_ids}"
        )

    def test_filtering_get_with_condition(self, test_index_path):
        """Test getting metadata with condition filter."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(3)]
        metadata = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]

        index.create(
            documents_embeddings=documents_embeddings,
            metadata=metadata,
            kmeans_niters=2,
        )

        # Get metadata for age > 28
        results = filtering.get(
            index=test_index_path, condition="age > ?", parameters=(28,)
        )
        assert len(results) == 2, "Expected 2 results"
        names = {r["name"] for r in results}
        assert names == {"Alice", "Charlie"}, f"Expected Alice and Charlie, got {names}"

    def test_filtering_get_with_subset(self, test_index_path):
        """Test getting metadata by subset IDs."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(5)]
        metadata = [
            {"name": "doc0"},
            {"name": "doc1"},
            {"name": "doc2"},
            {"name": "doc3"},
            {"name": "doc4"},
        ]

        index.create(
            documents_embeddings=documents_embeddings,
            metadata=metadata,
            kmeans_niters=2,
        )

        # Get metadata for specific subset
        results = filtering.get(index=test_index_path, subset=[1, 3])
        assert len(results) == 2, "Expected 2 results"
        names = [r["name"] for r in results]
        assert names == ["doc1", "doc3"], f"Expected doc1 and doc3, got {names}"

    def test_update_with_metadata(self, test_index_path):
        """Test updating index with new metadata."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        # Create initial index
        initial_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(3)]
        initial_metadata = [
            {"name": "doc0", "category": "initial"},
            {"name": "doc1", "category": "initial"},
            {"name": "doc2", "category": "initial"},
        ]
        index.create(
            documents_embeddings=initial_embeddings,
            metadata=initial_metadata,
            kmeans_niters=2,
        )

        # Update with new documents and metadata
        new_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(2)]
        new_metadata = [
            {"name": "doc3", "category": "updated"},
            {"name": "doc4", "category": "updated"},
        ]
        index.update(documents_embeddings=new_embeddings, metadata=new_metadata)

        # Verify all metadata is present
        all_metadata = filtering.get(index=test_index_path)
        assert len(all_metadata) == 5, "Expected 5 metadata entries"

        # Verify we can filter by the new category
        updated_ids = filtering.where(
            index=test_index_path, condition="category = ?", parameters=("updated",)
        )
        assert set(updated_ids) == {3, 4}, f"Expected [3, 4], got {updated_ids}"

    def test_filtering_with_date(self, test_index_path):
        """Test filtering with date fields."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(3)]
        metadata = [
            {"name": "doc0", "created": date(2023, 1, 1)},
            {"name": "doc1", "created": date(2023, 6, 15)},
            {"name": "doc2", "created": date(2024, 1, 1)},
        ]

        index.create(
            documents_embeddings=documents_embeddings,
            metadata=metadata,
            kmeans_niters=2,
        )

        # Filter by date
        recent_ids = filtering.where(
            index=test_index_path, condition="created > ?", parameters=("2023-03-01",)
        )
        assert set(recent_ids) == {1, 2}, f"Expected [1, 2], got {recent_ids}"

    def test_search_with_metadata_filter(self, test_index_path):
        """Test combining metadata filtering with search."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(10)]
        metadata = [
            {"category": "tech"},
            {"category": "sports"},
            {"category": "tech"},
            {"category": "sports"},
            {"category": "tech"},
            {"category": "news"},
            {"category": "news"},
            {"category": "tech"},
            {"category": "sports"},
            {"category": "news"},
        ]

        index.create(
            documents_embeddings=documents_embeddings,
            metadata=metadata,
            kmeans_niters=2,
        )

        # Get tech document IDs
        tech_ids = filtering.where(
            index=test_index_path, condition="category = ?", parameters=("tech",)
        )
        assert set(tech_ids) == {0, 2, 4, 7}, f"Expected [0, 2, 4, 7], got {tech_ids}"

        # Search only within tech documents
        queries = torch.randn(2, 30, 128, device="cpu")
        results = index.search(queries_embeddings=queries, top_k=4, subset=tech_ids)

        for query_results in results:
            for doc_id, _ in query_results:
                assert doc_id in tech_ids, (
                    f"Document ID {doc_id} not in tech subset {tech_ids}"
                )


class TestGetEmbeddings:
    """Tests for embedding reconstruction functionality."""

    def test_get_embeddings_basic(self, test_index_path):
        """Test reconstructing embeddings from the index."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(20)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        # Reconstruct embeddings for a subset of documents
        reconstructed = index.get_embeddings(subset=[0, 5, 10])

        assert len(reconstructed) == 3, (
            f"Expected 3 reconstructed embeddings, got {len(reconstructed)}"
        )

        # Check dimensions match
        for i, emb in enumerate(reconstructed):
            assert emb.dim() == 2, f"Expected 2D tensor for embedding {i}"
            assert emb.shape[1] == 128, (
                f"Expected embedding dimension 128, got {emb.shape[1]}"
            )

    def test_get_embeddings_empty_subset(self, test_index_path):
        """Test reconstructing embeddings with empty subset."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(10)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        # Empty subset should return empty list
        reconstructed = index.get_embeddings(subset=[])

        assert len(reconstructed) == 0, (
            f"Expected empty list, got {len(reconstructed)} embeddings"
        )


class TestQueryFormats:
    """Tests for different query embedding formats."""

    def test_query_as_list_of_tensors(self, test_index_path):
        """Test searching with queries as a list of tensors."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(30)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        # Queries as list of 2D tensors with different token counts
        queries_list = [
            torch.randn(25, 128, device="cpu"),
            torch.randn(50, 128, device="cpu"),
            torch.randn(10, 128, device="cpu"),
        ]

        results = index.search(queries_embeddings=queries_list, top_k=5)

        assert len(results) == 3, f"Expected 3 sets of results, got {len(results)}"

    def test_query_as_3d_tensor(self, test_index_path):
        """Test searching with queries as a 3D tensor."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(30)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        # Queries as 3D tensor [num_queries, tokens, dim]
        queries_tensor = torch.randn(5, 30, 128, device="cpu")

        results = index.search(queries_embeddings=queries_tensor, top_k=5)

        assert len(results) == 5, f"Expected 5 sets of results, got {len(results)}"

    def test_single_query(self, test_index_path):
        """Test searching with a single query."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(30)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        # Single query as 3D tensor [1, tokens, dim]
        single_query = torch.randn(1, 30, 128, device="cpu")

        results = index.search(queries_embeddings=single_query, top_k=5)

        assert len(results) == 1, f"Expected 1 set of results, got {len(results)}"
        assert len(results[0]) == 5, (
            f"Expected 5 results for the query, got {len(results[0])}"
        )


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_top_k_larger_than_index(self, test_index_path):
        """Test requesting more results than documents in the index."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(5)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=2)

        queries = torch.randn(2, 30, 128, device="cpu")
        results = index.search(queries_embeddings=queries, top_k=20)

        # Should return at most 5 results (the number of documents)
        for query_results in results:
            assert len(query_results) <= 5, (
                f"Expected at most 5 results, got {len(query_results)}"
            )

    def test_search_parameters(self, test_index_path):
        """Test different search parameters."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(50)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        queries = torch.randn(3, 30, 128, device="cpu")

        # Test with different n_ivf_probe values
        results_probe_2 = index.search(
            queries_embeddings=queries, top_k=10, n_ivf_probe=2
        )
        results_probe_16 = index.search(
            queries_embeddings=queries, top_k=10, n_ivf_probe=16
        )

        assert len(results_probe_2) == 3, "Expected 3 results with n_ivf_probe=2"
        assert len(results_probe_16) == 3, "Expected 3 results with n_ivf_probe=16"

    def test_different_nbits(self, test_index_path):
        """Test creating index with different nbits values."""
        # Test with nbits=2
        index_path_2bit = test_index_path + "_2bit"
        os.makedirs(index_path_2bit, exist_ok=True)

        index_2bit = None
        try:
            index_2bit = search.FastPlaid(index=index_path_2bit, device="cpu")

            documents_embeddings = [
                torch.randn(100, 128, device="cpu") for _ in range(30)
            ]
            index_2bit.create(
                documents_embeddings=documents_embeddings, kmeans_niters=4, nbits=2
            )

            queries = torch.randn(2, 30, 128, device="cpu")
            results = index_2bit.search(queries_embeddings=queries, top_k=5)

            assert len(results) == 2, "Expected 2 results with nbits=2"
        finally:
            if index_2bit is not None:
                index_2bit.close()
            if os.path.exists(index_path_2bit):
                shutil.rmtree(index_path_2bit)


class TestScoreConsistency:
    """Tests to verify score consistency and ordering."""

    def test_scores_are_sorted(self, test_index_path):
        """Verify that results are sorted by descending score."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(50)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        queries = torch.randn(5, 30, 128, device="cpu")
        results = index.search(queries_embeddings=queries, top_k=20)

        for query_results in results:
            scores = [score for _, score in query_results]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"Scores not sorted: {scores[i]} < {scores[i + 1]}"
                )

    def test_same_query_gives_same_results(self, test_index_path):
        """Verify that the same query gives consistent results."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(50)]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        query = torch.randn(1, 30, 128, device="cpu")

        results_1 = index.search(queries_embeddings=query, top_k=10)
        results_2 = index.search(queries_embeddings=query, top_k=10)

        # Same query should give same document IDs
        doc_ids_1 = [doc_id for doc_id, _ in results_1[0]]
        doc_ids_2 = [doc_id for doc_id, _ in results_2[0]]

        assert doc_ids_1 == doc_ids_2, (
            f"Inconsistent results: {doc_ids_1} != {doc_ids_2}"
        )


class TestFilteringModule:
    """Direct tests for the filtering module functions."""

    def test_filtering_delete_and_reindex(self, test_index_path):
        """Test that delete properly re-indexes the subset IDs."""
        # Create metadata directly
        metadata = [
            {"name": "doc0"},
            {"name": "doc1"},
            {"name": "doc2"},
            {"name": "doc3"},
            {"name": "doc4"},
        ]
        filtering.create(index=test_index_path, metadata=metadata)

        # Delete doc1 (index 1)
        filtering.delete(index=test_index_path, subset=[1])

        # After deletion, remaining docs should be re-indexed 0-3
        all_metadata = filtering.get(index=test_index_path)
        assert len(all_metadata) == 4, f"Expected 4 entries, got {len(all_metadata)}"

        subset_ids = [m["_subset_"] for m in all_metadata]
        assert subset_ids == [0, 1, 2, 3], f"Expected [0, 1, 2, 3], got {subset_ids}"

    def test_filtering_update_adds_columns(self, test_index_path):
        """Test that update can add new columns to the metadata."""
        # Create initial metadata
        initial_metadata = [
            {"name": "doc0", "category": "A"},
            {"name": "doc1", "category": "B"},
        ]
        filtering.create(index=test_index_path, metadata=initial_metadata)

        # Update with new metadata that has an additional column
        new_metadata = [
            {"name": "doc2", "category": "A", "extra_field": "value"},
        ]
        filtering.update(index=test_index_path, metadata=new_metadata)

        # Verify the new column exists
        all_metadata = filtering.get(index=test_index_path)
        assert len(all_metadata) == 3, f"Expected 3 entries, got {len(all_metadata)}"

        # The new column should exist for all rows (None for old rows)
        assert "extra_field" in all_metadata[2], "extra_field not found in metadata"
        assert all_metadata[2]["extra_field"] == "value", "extra_field has wrong value"


# Legacy test function for backwards compatibility
def test():
    """Ensure that the Fast-PLAiD search index can be created and queried correctly."""
    index_name = "test_index"

    if os.path.exists(index_name):
        shutil.rmtree(index_name, ignore_errors=True)
    os.makedirs(index_name, exist_ok=True)

    index = search.FastPlaid(
        index=index_name,
        device="cpu",
    )

    documents_embeddings = [torch.randn(300, 128, device="cpu") for _ in range(100)]

    queries_embeddings = torch.randn(10, 30, 128, device="cpu")

    index.create(
        documents_embeddings=documents_embeddings,
        kmeans_niters=4,
    )

    results = index.search(queries_embeddings=queries_embeddings, top_k=10)

    assert len(results) == 10, (
        f"Expected 10 sets of query results, but got {len(results)}"
    )

    assert all(len(query_res) == 10 for query_res in results), (
        "Expected each query to have 10 results"
    )

    print("Test passed: Results have the correct shape (10, 10).")

    index.close()
    shutil.rmtree(index_name, ignore_errors=True)
