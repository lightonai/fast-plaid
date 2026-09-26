import json
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


class TestSearchTokenScores:
    """Tests for search_token_scores returning per-token similarity matrices."""

    def test_search_token_scores_basic(self, test_index_path):
        """Test that search_token_scores returns correctly shaped token matrices."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        num_docs = 50
        doc_token_counts = [30 + i for i in range(num_docs)]
        documents_embeddings = [
            torch.randn(n_tok, 128, device="cpu") for n_tok in doc_token_counts
        ]
        query_tokens = 20
        queries_embeddings = torch.randn(3, query_tokens, 128, device="cpu")

        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
        results = index.search_token_scores(
            queries_embeddings=queries_embeddings, top_k=5
        )

        assert len(results) == 3, "Expected 3 sets of query results"
        for query_results in results:
            assert len(query_results) == 5, "Expected 5 results per query"
            for doc_id, score, token_scores in query_results:
                assert isinstance(doc_id, int)
                assert isinstance(score, float)
                assert isinstance(token_scores, torch.Tensor)
                # Shape should be [query_tokens, doc_tokens_for_this_doc]
                assert token_scores.shape[0] == query_tokens, (
                    f"Expected {query_tokens} query tokens, got {token_scores.shape[0]}"
                )
                expected_doc_tokens = doc_token_counts[doc_id]
                assert token_scores.shape[1] == expected_doc_tokens, (
                    f"Expected {expected_doc_tokens} doc tokens for doc {doc_id}, "
                    f"got {token_scores.shape[1]}"
                )

    def test_search_token_scores_consistency_with_search(self, test_index_path):
        """Test that search_token_scores returns the same rankings as search."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [
            torch.randn(50, 128, device="cpu") for _ in range(30)
        ]
        queries_embeddings = torch.randn(2, 20, 128, device="cpu")

        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)

        search_results = index.search(
            queries_embeddings=queries_embeddings, top_k=10
        )
        token_score_results = index.search_token_scores(
            queries_embeddings=queries_embeddings, top_k=10
        )

        for q_idx in range(len(search_results)):
            search_ids = [doc_id for doc_id, _ in search_results[q_idx]]
            token_ids = [doc_id for doc_id, _, _ in token_score_results[q_idx]]
            assert search_ids == token_ids, (
                f"Query {q_idx}: search and search_token_scores returned different rankings"
            )

            search_scores = [score for _, score in search_results[q_idx]]
            token_scores = [score for _, score, _ in token_score_results[q_idx]]
            for s1, s2 in zip(search_scores, token_scores):
                assert abs(s1 - s2) < 1e-3, (
                    f"Score mismatch: search={s1}, token_scores={s2}"
                )

    def test_search_token_scores_maxsim_values(self, test_index_path):
        """Test that manual MaxSim over token matrices matches returned scores."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [
            torch.randn(50, 128, device="cpu") for _ in range(30)
        ]
        queries_embeddings = torch.randn(3, 20, 128, device="cpu")

        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
        results = index.search_token_scores(
            queries_embeddings=queries_embeddings, top_k=10
        )

        for query_results in results:
            for doc_id, score, token_scores in query_results:
                # Manual MaxSim: for each query token, max similarity across
                # doc tokens, then sum. token_scores is [query_tokens, doc_tokens].
                manual_score = token_scores.max(dim=1).values.sum().item()
                assert abs(manual_score - score) < 0.1, (
                    f"Doc {doc_id}: manual MaxSim={manual_score:.4f} != "
                    f"returned score={score:.4f}"
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

    def test_update_immediately_after_create(self, test_index_path):
        """Update directly after create, with no search in between.

        create() defers device loads (self.indices holds None placeholders),
        so the first update() must reload the index itself. Regression test
        for _reload_index returning the caller's own dict: the reload branch
        in process_update then cleared the result through the alias and
        raised KeyError on the device key.

        start_from_scratch=0 keeps create() from saving embeddings.npy, as
        with a >1000-document index: without it, update() rebuilds the index
        from scratch and never reaches the lazy-reload branch under test.
        """
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            initial_embeddings = [
                torch.randn(100, 128, device="cpu") for _ in range(50)
            ]
            index.create(
                documents_embeddings=initial_embeddings,
                kmeans_niters=4,
                start_from_scratch=0,
            )

            new_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(50)]
            index.update(documents_embeddings=new_embeddings)

            queries = torch.randn(2, 30, 128, device="cpu")
            results = index.search(queries_embeddings=queries, top_k=50)

            for query_results in results:
                for doc_id, _ in query_results:
                    assert 0 <= doc_id < 100, (
                        f"Document ID {doc_id} out of updated range"
                    )
        finally:
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


class TestCompressOnly:
    """Tests for compress_only mode (no IVF construction)."""

    def test_compress_only_get_embeddings(self, test_index_path):
        """Test that get_embeddings works on a compress_only index."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(20)]
        index.create(
            documents_embeddings=documents_embeddings,
            kmeans_niters=4,
            compress_only=True,
        )

        # get_embeddings should work without IVF
        reconstructed = index.get_embeddings(subset=[0, 5, 10])

        assert len(reconstructed) == 3, (
            f"Expected 3 reconstructed embeddings, got {len(reconstructed)}"
        )
        for emb in reconstructed:
            assert emb.dim() == 2
            assert emb.shape[1] == 128

    def test_compress_only_no_ivf_files(self, test_index_path):
        """Test that compress_only skips writing ivf.npy and ivf_lengths.npy."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(20)]
        index.create(
            documents_embeddings=documents_embeddings,
            kmeans_niters=4,
            compress_only=True,
        )

        assert not os.path.exists(os.path.join(test_index_path, "ivf.npy"))
        assert not os.path.exists(os.path.join(test_index_path, "ivf_lengths.npy"))

    def test_compress_only_search_raises(self, test_index_path):
        """Test that search raises an error on a compress_only index."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(20)]
        index.create(
            documents_embeddings=documents_embeddings,
            kmeans_niters=4,
            compress_only=True,
        )

        queries_embeddings = torch.randn(2, 30, 128, device="cpu")
        with pytest.raises((RuntimeError, ValueError), match="compress_only"):
            index.search(queries_embeddings=queries_embeddings, top_k=5)

    def test_compress_only_metadata_flag(self, test_index_path):
        """Test that metadata.json contains compress_only field."""
        import json

        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(20)]
        index.create(
            documents_embeddings=documents_embeddings,
            kmeans_niters=4,
            compress_only=True,
        )

        with open(os.path.join(test_index_path, "metadata.json")) as f:
            metadata = json.load(f)

        assert metadata["compress_only"] is True

    def test_compress_only_update_and_get_embeddings(self, test_index_path):
        """Test that update works on a compress_only index and get_embeddings still works."""
        import json

        index = search.FastPlaid(index=test_index_path, device="cpu")

        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(20)]
        index.create(
            documents_embeddings=documents_embeddings,
            kmeans_niters=4,
            compress_only=True,
        )

        # Update with new documents
        new_embeddings = [torch.randn(80, 128, device="cpu") for _ in range(5)]
        index.update(documents_embeddings=new_embeddings)

        # IVF files should still not exist
        assert not os.path.exists(os.path.join(test_index_path, "ivf.npy"))
        assert not os.path.exists(os.path.join(test_index_path, "ivf_lengths.npy"))

        # compress_only flag should be preserved in metadata
        with open(os.path.join(test_index_path, "metadata.json")) as f:
            metadata = json.load(f)
        assert metadata["compress_only"] is True
        assert metadata["num_documents"] == 25

        # get_embeddings should work for both old and new documents
        reconstructed = index.get_embeddings(subset=[0, 10, 22])
        assert len(reconstructed) == 3
        for emb in reconstructed:
            assert emb.dim() == 2
            assert emb.shape[1] == 128


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

    def test_same_seed_gives_same_index(self, test_index_path):
        """Two independently-built indices with the same seed must return identical results."""
        rng = torch.Generator()
        rng.manual_seed(0)
        documents_embeddings = [torch.randn(30 + (i % 20), 128, generator=rng) for i in range(200)]
        queries = torch.randn(10, 25, 128, generator=rng)

        path_b = test_index_path + "_b"
        os.makedirs(path_b)

        index_a = search.FastPlaid(index=test_index_path, device="cpu")
        index_a.create(documents_embeddings=documents_embeddings, kmeans_niters=4, seed=42, n_samples_kmeans=50)
        results_a = index_a.search(queries_embeddings=queries, top_k=10)
        index_a.close()

        index_b = search.FastPlaid(index=path_b, device="cpu")
        index_b.create(documents_embeddings=documents_embeddings, kmeans_niters=4, seed=42, n_samples_kmeans=50)
        results_b = index_b.search(queries_embeddings=queries, top_k=10)
        index_b.close()
        shutil.rmtree(path_b)

        for q, (ra, rb) in enumerate(zip(results_a, results_b)):
            ids_a = [doc_id for doc_id, _ in ra]
            ids_b = [doc_id for doc_id, _ in rb]
            assert ids_a == ids_b, f"query {q}: non-deterministic results {ids_a} != {ids_b}"


class TestMetadataDocumentCount:
    """Tests for exact document count in metadata.json."""

    def _get_num_documents(self, index_path):
        """Helper to read num_documents from metadata.json."""
        import json

        metadata_path = os.path.join(index_path, "metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata.get("num_documents", 0)

    def test_create_sets_exact_document_count(self, test_index_path):
        """Test that creating an index sets the exact document count in metadata.json."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        num_docs = 25
        documents_embeddings = [
            torch.randn(50, 128, device="cpu") for _ in range(num_docs)
        ]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=2)

        assert self._get_num_documents(test_index_path) == num_docs, (
            f"Expected {num_docs} documents in metadata.json"
        )

    def test_update_increments_document_count_exactly(self, test_index_path):
        """Test that updating an index sets the exact document count in metadata.json."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            # Create initial index with 20 documents
            initial_docs = 20
            initial_embeddings = [
                torch.randn(50, 128, device="cpu") for _ in range(initial_docs)
            ]
            index.create(documents_embeddings=initial_embeddings, kmeans_niters=2)

            assert self._get_num_documents(test_index_path) == initial_docs, (
                f"Expected {initial_docs} documents after creation"
            )

            # First update with 10 documents
            update_1_docs = 10
            update_1_embeddings = [
                torch.randn(50, 128, device="cpu") for _ in range(update_1_docs)
            ]
            index.update(documents_embeddings=update_1_embeddings)

            expected_after_update_1 = initial_docs + update_1_docs
            assert self._get_num_documents(test_index_path) == expected_after_update_1, (
                f"Expected {expected_after_update_1} documents after first update, "
                f"got {self._get_num_documents(test_index_path)}"
            )

            # Second update with 15 documents
            update_2_docs = 15
            update_2_embeddings = [
                torch.randn(50, 128, device="cpu") for _ in range(update_2_docs)
            ]
            index.update(documents_embeddings=update_2_embeddings)

            expected_after_update_2 = expected_after_update_1 + update_2_docs
            assert self._get_num_documents(test_index_path) == expected_after_update_2, (
                f"Expected {expected_after_update_2} documents after second update, "
                f"got {self._get_num_documents(test_index_path)}"
            )
        finally:
            index.close()

    def test_delete_decrements_document_count_exactly(self, test_index_path):
        """Test that deleting documents sets the exact document count in metadata.json."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            # Create initial index with 30 documents
            initial_docs = 30
            initial_embeddings = [
                torch.randn(50, 128, device="cpu") for _ in range(initial_docs)
            ]
            index.create(documents_embeddings=initial_embeddings, kmeans_niters=2)

            assert self._get_num_documents(test_index_path) == initial_docs, (
                f"Expected {initial_docs} documents after creation"
            )

            # Delete 1 document
            index.delete(subset=[5])
            expected_after_delete_1 = initial_docs - 1
            assert self._get_num_documents(test_index_path) == expected_after_delete_1, (
                f"Expected {expected_after_delete_1} documents after deleting 1 document, "
                f"got {self._get_num_documents(test_index_path)}"
            )

            # Delete 4 more documents
            index.delete(subset=[0, 3, 10, 15])
            expected_after_delete_2 = expected_after_delete_1 - 4
            assert self._get_num_documents(test_index_path) == expected_after_delete_2, (
                f"Expected {expected_after_delete_2} documents after deleting 4 documents, "
                f"got {self._get_num_documents(test_index_path)}"
            )
        finally:
            index.close()

    def test_update_then_delete_maintains_exact_count(self, test_index_path):
        """Test that update followed by delete maintains exact document count."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            # Create initial index with 15 documents
            initial_docs = 15
            initial_embeddings = [
                torch.randn(50, 128, device="cpu") for _ in range(initial_docs)
            ]
            index.create(documents_embeddings=initial_embeddings, kmeans_niters=2)

            assert self._get_num_documents(test_index_path) == initial_docs

            # Update with 5 documents
            update_docs = 5
            update_embeddings = [
                torch.randn(50, 128, device="cpu") for _ in range(update_docs)
            ]
            index.update(documents_embeddings=update_embeddings)

            expected_after_update = initial_docs + update_docs
            assert self._get_num_documents(test_index_path) == expected_after_update, (
                f"Expected {expected_after_update} documents after update"
            )

            # Delete 3 documents (including one from the update)
            index.delete(subset=[2, 10, 17])
            expected_after_delete = expected_after_update - 3
            assert self._get_num_documents(test_index_path) == expected_after_delete, (
                f"Expected {expected_after_delete} documents after delete, "
                f"got {self._get_num_documents(test_index_path)}"
            )
        finally:
            index.close()

    def test_delete_then_update_maintains_exact_count(self, test_index_path):
        """Test that delete followed by update maintains exact document count."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            # Create initial index with 20 documents
            initial_docs = 20
            initial_embeddings = [
                torch.randn(50, 128, device="cpu") for _ in range(initial_docs)
            ]
            index.create(documents_embeddings=initial_embeddings, kmeans_niters=2)

            assert self._get_num_documents(test_index_path) == initial_docs

            # Delete 5 documents
            index.delete(subset=[0, 5, 10, 15, 19])
            expected_after_delete = initial_docs - 5
            assert self._get_num_documents(test_index_path) == expected_after_delete, (
                f"Expected {expected_after_delete} documents after delete"
            )

            # Update with 8 documents
            update_docs = 8
            update_embeddings = [
                torch.randn(50, 128, device="cpu") for _ in range(update_docs)
            ]
            index.update(documents_embeddings=update_embeddings)

            expected_after_update = expected_after_delete + update_docs
            assert self._get_num_documents(test_index_path) == expected_after_update, (
                f"Expected {expected_after_update} documents after update, "
                f"got {self._get_num_documents(test_index_path)}"
            )
        finally:
            index.close()

    def test_multiple_updates_and_deletes_exact_count(self, test_index_path):
        """Test exact document count after multiple interleaved updates and deletes."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            # Create initial index
            current_count = 10
            initial_embeddings = [
                torch.randn(50, 128, device="cpu") for _ in range(current_count)
            ]
            index.create(documents_embeddings=initial_embeddings, kmeans_niters=2)
            assert self._get_num_documents(test_index_path) == current_count

            # Update +5
            index.update(
                documents_embeddings=[
                    torch.randn(50, 128, device="cpu") for _ in range(5)
                ]
            )
            current_count += 5
            assert self._get_num_documents(test_index_path) == current_count

            # Delete 2
            index.delete(subset=[0, 7])
            current_count -= 2
            assert self._get_num_documents(test_index_path) == current_count

            # Update +3
            index.update(
                documents_embeddings=[
                    torch.randn(50, 128, device="cpu") for _ in range(3)
                ]
            )
            current_count += 3
            assert self._get_num_documents(test_index_path) == current_count

            # Delete 1
            index.delete(subset=[5])
            current_count -= 1
            assert self._get_num_documents(test_index_path) == current_count

            # Final verification
            assert self._get_num_documents(test_index_path) == 15, (
                f"Expected 15 documents after all operations, "
                f"got {self._get_num_documents(test_index_path)}"
            )
        finally:
            index.close()

    def test_document_count_matches_search_results(self, test_index_path):
        """Test that metadata.json count matches actual searchable documents."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            # Create index with 25 documents
            num_docs = 25
            documents_embeddings = [
                torch.randn(50, 128, device="cpu") for _ in range(num_docs)
            ]
            index.create(documents_embeddings=documents_embeddings, kmeans_niters=2)

            # Verify count in metadata.json
            metadata_count = self._get_num_documents(test_index_path)
            assert metadata_count == num_docs

            # Verify search returns all documents when requesting more than exist
            query = torch.randn(1, 30, 128, device="cpu")
            results = index.search(queries_embeddings=query, top_k=100)

            # Should get exactly num_docs results
            assert len(results[0]) == num_docs, (
                f"Search returned {len(results[0])} docs but metadata says {metadata_count}"
            )

            # After update
            index.update(
                documents_embeddings=[
                    torch.randn(50, 128, device="cpu") for _ in range(5)
                ]
            )
            metadata_count = self._get_num_documents(test_index_path)
            results = index.search(queries_embeddings=query, top_k=100)
            assert len(results[0]) == metadata_count, (
                f"After update: search returned {len(results[0])} docs but "
                f"metadata says {metadata_count}"
            )

            # After delete
            index.delete(subset=[0, 10, 20])
            metadata_count = self._get_num_documents(test_index_path)
            results = index.search(queries_embeddings=query, top_k=100)
            assert len(results[0]) == metadata_count, (
                f"After delete: search returned {len(results[0])} docs but "
                f"metadata says {metadata_count}"
            )
        finally:
            index.close()

    def test_document_count_with_metadata_db(self, test_index_path):
        """Test that metadata.json count matches metadata.db count after operations."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            # Create with metadata
            num_docs = 10
            documents_embeddings = [
                torch.randn(50, 128, device="cpu") for _ in range(num_docs)
            ]
            metadata = [{"name": f"doc{i}"} for i in range(num_docs)]
            index.create(
                documents_embeddings=documents_embeddings,
                metadata=metadata,
                kmeans_niters=2,
            )

            # Verify both counts match
            json_count = self._get_num_documents(test_index_path)
            db_count = len(filtering.get(index=test_index_path))
            assert json_count == db_count == num_docs, (
                f"Mismatch: json={json_count}, db={db_count}, expected={num_docs}"
            )

            # Update with metadata
            update_docs = 5
            update_embeddings = [
                torch.randn(50, 128, device="cpu") for _ in range(update_docs)
            ]
            update_metadata = [{"name": f"new_doc{i}"} for i in range(update_docs)]
            index.update(
                documents_embeddings=update_embeddings, metadata=update_metadata
            )

            expected_count = num_docs + update_docs
            json_count = self._get_num_documents(test_index_path)
            db_count = len(filtering.get(index=test_index_path))
            assert json_count == db_count == expected_count, (
                f"After update: json={json_count}, db={db_count}, "
                f"expected={expected_count}"
            )

            # Delete some documents
            index.delete(subset=[2, 7, 12])
            expected_count -= 3
            json_count = self._get_num_documents(test_index_path)
            db_count = len(filtering.get(index=test_index_path))
            assert json_count == db_count == expected_count, (
                f"After delete: json={json_count}, db={db_count}, "
                f"expected={expected_count}"
            )
        finally:
            index.close()


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


class TestFreeze:
    """Tests for the freeze() API that marks an index immutable."""

    def test_storage_is_single_copy_and_freeze_keeps_results(self, test_index_path):
        """Payload shards are dropped at first load; freeze() changes nothing on disk."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            documents_embeddings = [
                torch.randn(100, 128, device="cpu") for _ in range(80)
            ]
            queries_embeddings = torch.randn(5, 30, 128, device="cpu")

            index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
            results_before = index.search(
                queries_embeddings=queries_embeddings, top_k=10
            )

            # The index is single-copy from the first load: no payload shards.
            shard_files = [
                f
                for f in os.listdir(test_index_path)
                if (f.endswith(".codes.npy") or f.endswith(".residuals.npy"))
                and not f.startswith("merged_")
            ]
            assert shard_files == [], (
                f"expected no per-shard payload files, got {shard_files}"
            )
            assert os.path.exists(os.path.join(test_index_path, "merged_codes.npy"))
            assert os.path.exists(
                os.path.join(test_index_path, "merged_residuals.npy")
            )

            index.freeze()

            results_after = index.search(
                queries_embeddings=queries_embeddings, top_k=10
            )
            assert results_before == results_after, (
                "freeze() must not change search results"
            )
        finally:
            index.close()

    def test_freeze_persists_across_reload(self, test_index_path):
        """A frozen index reloads correctly from disk in a fresh instance."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            documents_embeddings = [
                torch.randn(80, 128, device="cpu") for _ in range(60)
            ]
            queries_embeddings = torch.randn(3, 30, 128, device="cpu")

            index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
            index.freeze()
            results_before = index.search(
                queries_embeddings=queries_embeddings, top_k=10
            )
        finally:
            index.close()

        reloaded = search.FastPlaid(index=test_index_path, device="cpu")
        try:
            results_after = reloaded.search(
                queries_embeddings=queries_embeddings, top_k=10
            )
            assert results_before == results_after
        finally:
            reloaded.close()

    def test_freeze_update_raises(self, test_index_path):
        """update() on a frozen index must fail loudly rather than corrupt it."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            documents_embeddings = [
                torch.randn(60, 128, device="cpu") for _ in range(40)
            ]
            index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
            index.freeze()

            new_embeddings = [torch.randn(60, 128, device="cpu") for _ in range(10)]
            with pytest.raises(RuntimeError, match="frozen"):
                index.update(documents_embeddings=new_embeddings)
        finally:
            index.close()

    def test_freeze_delete_raises(self, test_index_path):
        """delete() on a frozen index must fail loudly."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            documents_embeddings = [
                torch.randn(60, 128, device="cpu") for _ in range(40)
            ]
            index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
            index.freeze()

            with pytest.raises(RuntimeError, match="frozen"):
                index.delete(subset=[0, 1, 2])
        finally:
            index.close()

    def test_freeze_idempotent(self, test_index_path):
        """Calling freeze() twice should be a no-op the second time."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            documents_embeddings = [
                torch.randn(60, 128, device="cpu") for _ in range(40)
            ]
            index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
            index.freeze()
            index.freeze()  # second call: must not raise
        finally:
            index.close()

    def test_freeze_unfreeze_roundtrip_byte_identical(self, tmp_path):
        """freeze() then unfreeze() must restore every file byte-for-byte.

        Build an index with several shards, snapshot its directory, then
        freeze and unfreeze the original. Every file in the unfrozen index
        must match the snapshot exactly (modulo merged_*.manifest.json,
        which is a cache of mtimes regenerated on load).
        """
        import filecmp
        import gc
        import shutil

        original_path = str(tmp_path / "original")
        snapshot_path = str(tmp_path / "snapshot")
        os.makedirs(original_path, exist_ok=True)

        try:
            index = search.FastPlaid(index=original_path, device="cpu")
            # Force several shards so the roundtrip exercises real chunking.
            documents_embeddings = [
                torch.randn(80, 128, device="cpu") for _ in range(120)
            ]
            index.create(
                documents_embeddings=documents_embeddings,
                kmeans_niters=4,
                batch_size=40,
            )
            index.close()
            gc.collect()

            # Snapshot the pristine on-disk index.
            shutil.copytree(original_path, snapshot_path)

            # Round-trip: freeze then unfreeze the original.
            index = search.FastPlaid(index=original_path, device="cpu")
            index.freeze()
            index.unfreeze()
            index.close()
            gc.collect()

            # Manifests are mtime caches, not source data; ignored intentionally.
            ignored = {"merged_codes.manifest.json", "merged_residuals.manifest.json"}

            original_files = {
                f for f in os.listdir(original_path) if f not in ignored
            }
            snapshot_files = {
                f for f in os.listdir(snapshot_path) if f not in ignored
            }
            assert original_files == snapshot_files, (
                f"file set diverged: only-in-original={original_files - snapshot_files}, "
                f"only-in-snapshot={snapshot_files - original_files}"
            )

            mismatches = []
            for name in sorted(original_files):
                a = os.path.join(original_path, name)
                b = os.path.join(snapshot_path, name)
                if not filecmp.cmp(a, b, shallow=False):
                    mismatches.append(name)

            assert mismatches == [], (
                f"freeze/unfreeze roundtrip changed file bytes: {mismatches}"
            )
        finally:
            shutil.rmtree(original_path, ignore_errors=True)
            shutil.rmtree(snapshot_path, ignore_errors=True)

    def test_unfreeze_restores_update_capability(self, test_index_path):
        """After unfreeze() the index must accept update() again."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            documents_embeddings = [
                torch.randn(60, 128, device="cpu") for _ in range(40)
            ]
            index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
            index.freeze()
            index.unfreeze()

            new_embeddings = [torch.randn(60, 128, device="cpu") for _ in range(10)]
            index.update(documents_embeddings=new_embeddings)

            queries = torch.randn(2, 30, 128, device="cpu")
            results = index.search(queries_embeddings=queries, top_k=10)
            for query_results in results:
                for doc_id, _ in query_results:
                    assert 0 <= doc_id < 50
        finally:
            index.close()

    def test_unfreeze_idempotent(self, test_index_path):
        """Calling unfreeze() on a non-frozen index is a no-op."""
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            documents_embeddings = [
                torch.randn(60, 128, device="cpu") for _ in range(40)
            ]
            index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
            index.unfreeze()  # never frozen → no-op, must not raise
        finally:
            index.close()



class TestSingleCopyStorage:
    """The merged files are the only durable payload copy (issue #53)."""

    def _payload_shards(self, path):
        return sorted(
            f
            for f in os.listdir(path)
            if (f.endswith(".codes.npy") or f.endswith(".residuals.npy"))
            and not f.startswith("merged_")
        )

    def _build(self, path, n_docs=60, doc_len=60):
        index = search.FastPlaid(index=path, device="cpu")
        documents_embeddings = [
            torch.randn(doc_len, 128, device="cpu") for _ in range(n_docs)
        ]
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
        return index

    def test_update_stays_single_copy_and_is_searchable(self, test_index_path):
        """update() stages what Rust needs, then returns to a single copy."""
        index = self._build(test_index_path, n_docs=40)
        try:
            new_embeddings = [torch.randn(60, 128, device="cpu") for _ in range(10)]
            index.update(documents_embeddings=new_embeddings)

            assert self._payload_shards(test_index_path) == [], (
                "payload shards must be dropped again after update()"
            )
            queries = torch.randn(2, 30, 128, device="cpu")
            results = index.search(queries_embeddings=queries, top_k=50)
            seen = {doc_id for r in results for doc_id, _ in r}
            assert any(doc_id >= 40 for doc_id in seen), (
                "updated documents must be retrievable"
            )
        finally:
            index.close()

    def test_delete_stays_single_copy_and_is_searchable(self, test_index_path):
        """delete() re-materializes shards for Rust, then drops them again."""
        index = self._build(test_index_path, n_docs=40)
        try:
            index.delete(subset=[0, 1, 2])

            assert self._payload_shards(test_index_path) == [], (
                "payload shards must be dropped again after delete()"
            )
            queries = torch.randn(2, 30, 128, device="cpu")
            results = index.search(queries_embeddings=queries, top_k=37)
            for query_results in results:
                assert len(query_results) == 37
        finally:
            index.close()

    def test_double_copy_index_upgrades_on_load(self, test_index_path, monkeypatch):
        """An index from an older version (shards + merged) converts in place."""
        monkeypatch.setenv("FAST_PLAID_KEEP_SHARDS", "1")
        index = self._build(test_index_path, n_docs=40)
        queries = torch.randn(2, 30, 128, device="cpu")
        results_before = index.search(queries_embeddings=queries, top_k=10)
        index.close()
        assert self._payload_shards(test_index_path) != []

        # Recreate the exact legacy on-disk state: no merged-manifest flags,
        # no frozen/mutable markers, numpy-written merged headers.
        for suffix in ("codes", "residuals"):
            manifest_path = os.path.join(
                test_index_path, f"merged_{suffix}.manifest.json"
            )
            with open(manifest_path) as f:
                manifest = json.load(f)
            for entry in manifest.values():
                entry.pop("merged", None)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
        meta_path = os.path.join(test_index_path, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta.pop("frozen", None)
        meta.pop("mutable", None)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        monkeypatch.delenv("FAST_PLAID_KEEP_SHARDS")
        index = search.FastPlaid(index=test_index_path, device="cpu")
        try:
            results_after = index.search(queries_embeddings=queries, top_k=10)
            assert results_before == results_after, (
                "conversion must not change search results"
            )
            assert self._payload_shards(test_index_path) == [], (
                "conversion must drop the redundant payload copy"
            )
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta.get("frozen") is True and meta.get("mutable") is True, (
                "converted metadata must be read-only-safe for older versions"
            )
        finally:
            index.close()

    def test_manifest_loss_recovers_from_doclens(self, test_index_path):
        """Deleting the manifests must not make the index appear empty."""
        index = self._build(test_index_path, n_docs=40)
        queries = torch.randn(2, 30, 128, device="cpu")
        results_before = index.search(queries_embeddings=queries, top_k=10)
        index.close()

        for suffix in ("codes", "residuals"):
            os.remove(
                os.path.join(test_index_path, f"merged_{suffix}.manifest.json")
            )

        index = search.FastPlaid(index=test_index_path, device="cpu")
        try:
            results_after = index.search(queries_embeddings=queries, top_k=10)
            assert results_before == results_after, (
                "the doclens files define the merged layout; losing the manifest "
                "must be recoverable"
            )
        finally:
            index.close()

    def test_stale_staging_shard_is_reabsorbed(self, test_index_path):
        """A staging file surviving a crash is folded back in, not duplicated."""
        from fast_plaid.search import storage

        index = self._build(test_index_path, n_docs=40)
        queries = torch.randn(2, 30, 128, device="cpu")
        results_before = index.search(queries_embeddings=queries, top_k=10)
        index.close()

        # Simulate a crash after re-materialization: shard 0 exists on disk again.
        storage.materialize_shards(test_index_path, 1, "codes", [0])
        storage.materialize_shards(test_index_path, 1, "residuals", [0])
        assert self._payload_shards(test_index_path) != []

        index = search.FastPlaid(index=test_index_path, device="cpu")
        try:
            results_after = index.search(queries_embeddings=queries, top_k=10)
            assert results_before == results_after
            assert self._payload_shards(test_index_path) == [], (
                "recovered staging files must be dropped again"
            )
        finally:
            index.close()

    def test_keep_shards_opt_out(self, test_index_path, monkeypatch):
        """FAST_PLAID_KEEP_SHARDS=1 preserves the double-copy layout."""
        monkeypatch.setenv("FAST_PLAID_KEEP_SHARDS", "1")
        index = self._build(test_index_path, n_docs=40)
        try:
            assert self._payload_shards(test_index_path) != [], (
                "opt-out must keep payload shards on disk"
            )
            meta_path = os.path.join(test_index_path, "metadata.json")
            with open(meta_path) as f:
                meta = json.load(f)
            assert "mutable" not in meta, (
                "opt-out must not stamp single-copy metadata"
            )
        finally:
            index.close()

    def test_legacy_frozen_index_unfreezes_and_updates(self, test_index_path):
        """An index frozen by an older version becomes mutable again."""
        index = self._build(test_index_path, n_docs=40)
        index.close()

        # Recreate the legacy frozen state: no payload shards (already true),
        # frozen flag without the mutable marker.
        meta_path = os.path.join(test_index_path, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["frozen"] = True
        meta.pop("mutable", None)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        index = search.FastPlaid(index=test_index_path, device="cpu")
        try:
            new_embeddings = [torch.randn(60, 128, device="cpu") for _ in range(5)]
            with pytest.raises(RuntimeError, match="frozen"):
                index.update(documents_embeddings=new_embeddings)

            index.unfreeze()
            index.update(documents_embeddings=new_embeddings)
            queries = torch.randn(2, 30, 128, device="cpu")
            results = index.search(queries_embeddings=queries, top_k=45)
            for query_results in results:
                assert len(query_results) == 45
        finally:
            index.close()



class TestSingleCopyContracts:
    """CI guards for the invariants the single-copy layout promises (issue #53).

    Each test pins one contract that was validated manually during development:
    the downgrade-safe manifest schema, byte-identical shard materialization,
    crash-window recovery, mutation bookkeeping under churn, and the fixed-length
    merged header that makes in-place resizing unconditional.
    """

    def _build(self, path, n_docs=60, doc_len=60, **create_kwargs):
        index = search.FastPlaid(index=path, device="cpu")
        documents_embeddings = [
            torch.randn(doc_len, 128, device="cpu") for _ in range(n_docs)
        ]
        index.create(
            documents_embeddings=documents_embeddings,
            kmeans_niters=4,
            **create_kwargs,
        )
        return index

    def _payload_shards(self, path):
        return sorted(
            f
            for f in os.listdir(path)
            if (f.endswith(".codes.npy") or f.endswith(".residuals.npy"))
            and not f.startswith("merged_")
        )

    def test_manifest_entries_keep_the_downgrade_contract(self, test_index_path):
        """Every manifest entry carries "rows" and "mtime" keys, always.

        Older fast-plaid versions subscript entry["mtime"] directly; an entry
        without the key turns a downgrade into a KeyError instead of a clean
        re-merge. "mtime" may be None (rows live only in the merged file), but
        it must exist. This regressed once during development.
        """
        index = self._build(test_index_path, n_docs=40)
        try:
            index.update(
                documents_embeddings=[torch.randn(60, 128) for _ in range(5)]
            )
            index.delete(subset=[0, 1])
        finally:
            index.close()

        for suffix in ("codes", "residuals"):
            manifest_path = os.path.join(
                test_index_path, f"merged_{suffix}.manifest.json"
            )
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest, f"empty {suffix} manifest"
            for filename, entry in manifest.items():
                assert "rows" in entry, f"{filename}: missing rows"
                assert "mtime" in entry, (
                    f"{filename}: missing mtime key — old versions subscript "
                    f"entry['mtime'] and would crash on this manifest"
                )

    def test_materialized_shards_are_byte_identical(self, test_index_path, monkeypatch):
        """Re-materialized staging files must equal what the Rust core wrote.

        The Rust core reads staging shards with tch's npy parser; a drifting
        header format or a wrong slice offset would corrupt every update and
        delete. Compare materialized bytes against a double-copy snapshot.
        """
        import filecmp

        from fast_plaid.search import storage

        monkeypatch.setenv("FAST_PLAID_KEEP_SHARDS", "1")
        index = self._build(test_index_path, n_docs=80)
        index.close()
        originals = {
            name: os.path.join(test_index_path, name)
            for name in self._payload_shards(test_index_path)
        }
        assert originals, "expected payload shards under FAST_PLAID_KEEP_SHARDS"
        snapshot_dir = os.path.join(test_index_path, "snapshot")
        os.makedirs(snapshot_dir)
        for name, path in originals.items():
            shutil.copy2(path, os.path.join(snapshot_dir, name))

        monkeypatch.delenv("FAST_PLAID_KEEP_SHARDS")
        index = search.FastPlaid(index=test_index_path, device="cpu")
        index.close()
        assert self._payload_shards(test_index_path) == []

        with open(os.path.join(test_index_path, "metadata.json")) as f:
            num_chunks = json.load(f)["num_chunks"]
        storage.materialize_shards(test_index_path, num_chunks, "codes")
        storage.materialize_shards(test_index_path, num_chunks, "residuals")

        mismatches = [
            name
            for name in originals
            if not filecmp.cmp(
                os.path.join(test_index_path, name),
                os.path.join(snapshot_dir, name),
                shallow=False,
            )
        ]
        assert mismatches == [], (
            f"materialized shards diverge from what Rust wrote: {mismatches}"
        )

    def test_crash_after_staging_before_merge_recovers(self, test_index_path):
        """A crash after writing staging but before merging must self-heal.

        Simulated state: the last chunk exists only as a staging file, the
        merged file was never extended with its rows, and the manifest predates
        it. The next load must fold the chunk back in and return the same
        results as before the "crash".
        """
        import numpy as np

        from fast_plaid.search import storage

        index = self._build(test_index_path, n_docs=80)
        queries = torch.randn(3, 30, 128, device="cpu")
        reference = index.search(queries_embeddings=queries, top_k=10)
        index.close()

        with open(os.path.join(test_index_path, "metadata.json")) as f:
            num_chunks = json.load(f)["num_chunks"]
        last = num_chunks - 1
        last_rows = storage.shard_rows(test_index_path, last)
        assert last_rows > 0

        # Recreate the pre-crash staging file from the intact merged copy,
        # then rewind the merged files and manifests to before the merge.
        storage.materialize_shards(test_index_path, num_chunks, "codes", [last])
        storage.materialize_shards(test_index_path, num_chunks, "residuals", [last])
        for suffix, numpy_dtype in (("codes", np.int64), ("residuals", np.uint8)):
            merged_path = os.path.join(test_index_path, f"merged_{suffix}.npy")
            arr = np.load(merged_path, mmap_mode="r")
            shorter = (arr.shape[0] - last_rows, *arr.shape[1:])
            del arr
            storage.open_merged(merged_path, numpy_dtype, shorter, fresh=False)
            manifest = storage.load_manifest(test_index_path, suffix)
            manifest.pop(f"{last}.{suffix}.npy", None)
            storage.save_manifest(test_index_path, suffix, manifest)

        recovered = search.FastPlaid(index=test_index_path, device="cpu")
        try:
            results = recovered.search(queries_embeddings=queries, top_k=10)
            assert results == reference, (
                "recovery from a pre-merge crash changed search results"
            )
            assert self._payload_shards(test_index_path) == [], (
                "recovered staging must be dropped after re-merging"
            )
        finally:
            recovered.close()

    def test_crash_mid_materialization_recovers(self, test_index_path):
        """A staging file whose manifest entry was never updated is harmless.

        materialize_shards writes the file first and the manifest second; a
        crash between the two leaves a file whose recorded mtime is stale
        (None). The scan must treat it as dirty, rewrite the same bytes in
        place, and keep every later merged-only chunk readable — this is the
        offsets-equal tolerance path in _get_merged_mmap.
        """
        from fast_plaid.search import storage

        # batch_size forces several chunks; the scenario needs a chain break
        # between a staged chunk and merged-only chunks after it.
        index = self._build(test_index_path, n_docs=80, batch_size=30)
        queries = torch.randn(3, 30, 128, device="cpu")
        reference = index.search(queries_embeddings=queries, top_k=10)
        index.close()

        with open(os.path.join(test_index_path, "metadata.json")) as f:
            num_chunks = json.load(f)["num_chunks"]
        assert num_chunks >= 2, "need at least two chunks for a chain break"

        # Materialize chunk 0, then erase its mtime from the manifest: the
        # scan sees the file as dirty while chunk 1+ live only in merged.
        for suffix in ("codes", "residuals"):
            storage.materialize_shards(test_index_path, num_chunks, suffix, [0])
            manifest = storage.load_manifest(test_index_path, suffix)
            manifest[f"0.{suffix}.npy"]["mtime"] = None
            storage.save_manifest(test_index_path, suffix, manifest)

        recovered = search.FastPlaid(index=test_index_path, device="cpu")
        try:
            results = recovered.search(queries_embeddings=queries, top_k=10)
            assert results == reference, (
                "spurious chain break changed search results"
            )
            assert self._payload_shards(test_index_path) == []
        finally:
            recovered.close()

    def test_churn_keeps_bookkeeping_and_self_retrieval(self, test_index_path):
        """Cycles of update/random-delete keep ids exact and metadata honest.

        The compact version of the churn stress test: after every mutation the
        index's document count must equal the tracker's, and at the end each
        probed document must retrieve itself by its own embedding — which
        fails loudly if any position/id mapping drifted.
        """
        import random

        rng = random.Random(7)
        docs = [torch.randn(50, 128, device="cpu") for _ in range(300)]
        index = search.FastPlaid(index=test_index_path, device="cpu")
        try:
            index.create(documents_embeddings=docs[:100], kmeans_niters=4)
            live = list(range(100))
            fed = 100
            while fed < len(docs):
                batch = list(range(fed, min(fed + 100, len(docs))))
                index.update(
                    documents_embeddings=[docs[i] for i in batch],
                    start_from_scratch=0,
                )
                live.extend(batch)
                fed = batch[-1] + 1

                positions = sorted(rng.sample(range(len(live)), 50))
                index.delete(subset=positions)
                kill = set(positions)
                live = [d for p, d in enumerate(live) if p not in kill]

                with open(os.path.join(test_index_path, "metadata.json")) as f:
                    meta_docs = json.load(f)["num_documents"]
                assert meta_docs == len(live), (
                    f"bookkeeping drift: index {meta_docs} vs tracker {len(live)}"
                )

            probe_positions = rng.sample(range(len(live)), 5)
            results = index.search(
                queries_embeddings=[docs[live[p]] for p in probe_positions],
                top_k=1,
            )
            for position, res in zip(probe_positions, results):
                assert res and res[0][0] == position, (
                    f"self-retrieval failed at position {position}: {res}"
                )
        finally:
            index.close()

    def test_merged_header_is_fixed_length_across_growth(self, tmp_path):
        """The merged header never changes length, so resizing is in place.

        The old numpy-written header silently forced a full merged rewrite
        whenever the row count crossed a digit boundary. The fixed prologue
        must hold from 1 row to 20-digit row counts, and open_merged with
        fresh=False must preserve existing data bytes across a resize.
        """
        import numpy as np

        from fast_plaid.search import storage

        path = str(tmp_path / "merged_probe.npy")
        for rows in (1, 9, 10, 99, 100, 10**6, 10**12, 10**19):
            storage.write_merged_header(path, np.int64, (rows, 64))
            assert storage.has_fixed_header(path), f"header drifted at rows={rows}"

        os.remove(path)
        first = storage.open_merged(path, np.uint8, (10, 4), fresh=True)
        first[:] = np.arange(40, dtype=np.uint8).reshape(10, 4)
        first.flush()
        del first

        grown = storage.open_merged(path, np.uint8, (1000, 4), fresh=False)
        assert (
            grown[:10] == np.arange(40, dtype=np.uint8).reshape(10, 4)
        ).all(), "in-place growth clobbered existing rows"
        del grown

        loaded = np.load(path)
        assert loaded.shape == (1000, 4)
        assert (loaded[:10] == np.arange(40, dtype=np.uint8).reshape(10, 4)).all()

    def test_second_instance_sees_first_instances_update(self, test_index_path):
        """Two FastPlaid objects on one index: mutation by one is seen by the other.

        The single-copy layout mutates the merged files in place and unlinks
        staging, so the mtime-based reload of a concurrently open instance is
        the path most at risk — especially on Windows, where unlinking and
        truncating around live mmaps is what this repo historically fought.
        """
        docs = [torch.randn(50, 128, device="cpu") for _ in range(120)]
        writer = search.FastPlaid(index=test_index_path, device="cpu")
        reader = None
        try:
            writer.create(documents_embeddings=docs, kmeans_niters=4)
            reader = search.FastPlaid(index=test_index_path, device="cpu")
            queries = torch.randn(2, 30, 128, device="cpu")
            reader.search(queries_embeddings=queries, top_k=5)

            new_doc = torch.randn(50, 128, device="cpu")
            writer.update(documents_embeddings=[new_doc], start_from_scratch=0)

            results = reader.search(queries_embeddings=[new_doc], top_k=1)
            assert results and results[0][0][0] == len(docs), (
                "reader did not observe the writer's update after reload"
            )
        finally:
            if reader is not None:
                reader.close()
            writer.close()


class TestLegacyArguments:
    """Call sites written against pre-1.5.0 versions must keep working."""

    def _index(self, path) -> None:
        """Build a small searchable index at `path`."""
        index = search.FastPlaid(index=path, device="cpu")
        try:
            documents_embeddings = [
                torch.randn(60, 128, device="cpu") for _ in range(40)
            ]
            index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
        finally:
            index.close()

    @pytest.mark.parametrize("low_memory", [True, False])
    def test_low_memory_keyword_is_ignored(self, test_index_path, low_memory):
        """The removed low_memory flag is accepted as a keyword and ignored."""
        self._index(test_index_path)

        with pytest.warns(DeprecationWarning, match="low_memory"):
            index = search.FastPlaid(
                index=test_index_path, device="cpu", low_memory=low_memory
            )

        try:
            results = index.search(
                queries_embeddings=torch.randn(2, 30, 128, device="cpu"), top_k=5
            )
            assert len(results) == 2, f"Expected 2 query results, got {len(results)}"
            assert index.index_gpu_memory == "auto", (
                f"low_memory must not set placement, got {index.index_gpu_memory}"
            )
        finally:
            index.close()

    @pytest.mark.parametrize("low_memory", [True, False])
    def test_low_memory_positional_is_ignored(self, test_index_path, low_memory):
        """Pre-1.5.0 code passed low_memory as the third positional argument."""
        self._index(test_index_path)

        with pytest.warns(DeprecationWarning, match="low_memory"):
            index = search.FastPlaid(test_index_path, "cpu", low_memory)

        try:
            assert index.index_gpu_memory == "auto", (
                f"Expected 'auto' placement, got {index.index_gpu_memory}"
            )
            results = index.search(
                queries_embeddings=torch.randn(2, 30, 128, device="cpu"), top_k=5
            )
            assert len(results) == 2, f"Expected 2 query results, got {len(results)}"
        finally:
            index.close()

    def test_unknown_keywords_are_ignored(self, test_index_path):
        """Unknown keyword arguments must not raise."""
        self._index(test_index_path)

        index = search.FastPlaid(
            index=test_index_path, device="cpu", verbose=True, some_removed_option=7
        )

        try:
            results = index.search(
                queries_embeddings=torch.randn(1, 30, 128, device="cpu"), top_k=5
            )
            assert len(results) == 1, f"Expected 1 query result, got {len(results)}"
        finally:
            index.close()

    def test_legacy_integer_batch_size(self, test_index_path):
        """search(batch_size=<int>) predates batch_size='auto' and still works."""
        self._index(test_index_path)
        index = search.FastPlaid(index=test_index_path, device="cpu")

        try:
            queries_embeddings = torch.randn(2, 30, 128, device="cpu")
            auto = index.search(queries_embeddings=queries_embeddings, top_k=5)
            fixed = index.search(
                queries_embeddings=queries_embeddings, top_k=5, batch_size=2000
            )
            assert [[doc for doc, _ in q] for q in auto] == [
                [doc for doc, _ in q] for q in fixed
            ], "Fixed batch_size must not change the ranking"
        finally:
            index.close()

    def test_invalid_index_gpu_memory_still_rejected(self, test_index_path):
        """Tolerating legacy arguments must not weaken validation of real ones."""
        with pytest.raises(ValueError, match="index_gpu_memory"):
            search.FastPlaid(index=test_index_path, device="cpu", index_gpu_memory="hi")


class TestPrecomputedCentroids:
    """Tests for passing precomputed centroids to skip K-means."""

    def test_precomputed_centroids_same_results(self, test_index_path):
        """Passing centroids from a prior run produces identical search results."""
        rng = torch.Generator()
        rng.manual_seed(7)
        documents_embeddings = [torch.randn(30 + (i % 20), 128, generator=rng) for i in range(200)]
        queries = torch.randn(5, 25, 128, generator=rng)

        path_b = test_index_path + "_precomp"
        os.makedirs(path_b)

        index_a = search.FastPlaid(index=test_index_path, device="cpu")
        index_a.create(documents_embeddings=documents_embeddings, kmeans_niters=4, seed=42, n_samples_kmeans=50)
        results_a = index_a.search(queries_embeddings=queries, top_k=10)

        centroids = torch.from_numpy(
            __import__("numpy").load(os.path.join(test_index_path, "centroids.npy"))
        )
        index_a.close()

        index_b = search.FastPlaid(index=path_b, device="cpu")
        index_b.create(
            documents_embeddings=documents_embeddings,
            centroids=centroids,
            seed=42,
        )
        results_b = index_b.search(queries_embeddings=queries, top_k=10)
        index_b.close()
        shutil.rmtree(path_b)

        for q, (ra, rb) in enumerate(zip(results_a, results_b)):
            ids_a = [doc_id for doc_id, _ in ra]
            ids_b = [doc_id for doc_id, _ in rb]
            assert ids_a == ids_b, f"query {q}: centroids passthrough changed results {ids_a} != {ids_b}"


class TestBucketOverrides:
    """Tests for explicit bucket_cutoffs / bucket_weights overrides."""

    def test_explicit_buckets_creates_searchable_index(self, test_index_path):
        """An index built with explicit bucket overrides is searchable."""
        import numpy as np

        index = search.FastPlaid(index=test_index_path, device="cpu")
        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(50)]

        index.create(
            documents_embeddings=documents_embeddings,
            kmeans_niters=4,
            nbits=2,
            bucket_cutoffs=[-0.5, 0.0, 0.5],
            bucket_weights=[-0.75, -0.25, 0.25, 0.75],
        )

        queries = torch.randn(3, 30, 128, device="cpu")
        results = index.search(queries_embeddings=queries, top_k=10)

        assert len(results) == 3
        for query_results in results:
            assert len(query_results) == 10

        saved_cutoffs = np.load(os.path.join(test_index_path, "bucket_cutoffs.npy"))
        assert saved_cutoffs.shape == (3,)

    def test_bucket_override_bad_shape_raises(self, test_index_path):
        """Wrong number of cutoffs/weights is rejected."""
        index = search.FastPlaid(index=test_index_path, device="cpu")
        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(50)]

        with pytest.raises(RuntimeError):
            index.create(
                documents_embeddings=documents_embeddings,
                kmeans_niters=4,
                nbits=2,
                bucket_cutoffs=[-0.5, 0.5],
                bucket_weights=[-0.75, 0.75],
            )


class TestFastPathOverrides:
    """Tests for the full fast-path (avg_residual + cluster_threshold + buckets)."""

    def test_fast_path_matches_standard_path(self, test_index_path):
        """Full override path produces identical results to the standard path."""
        import numpy as np

        rng = torch.Generator()
        rng.manual_seed(99)
        documents_embeddings = [torch.randn(30 + (i % 15), 128, generator=rng) for i in range(100)]
        queries = torch.randn(5, 25, 128, generator=rng)

        index_a = search.FastPlaid(index=test_index_path, device="cpu")
        index_a.create(documents_embeddings=documents_embeddings, kmeans_niters=4, seed=42, n_samples_kmeans=50)
        results_a = index_a.search(queries_embeddings=queries, top_k=10)

        centroids = torch.from_numpy(np.load(os.path.join(test_index_path, "centroids.npy")))
        avg_residual = torch.from_numpy(np.load(os.path.join(test_index_path, "avg_residual.npy")))
        cluster_threshold = torch.from_numpy(np.load(os.path.join(test_index_path, "cluster_threshold.npy")))
        bucket_cutoffs_np = np.load(os.path.join(test_index_path, "bucket_cutoffs.npy"))
        bucket_weights_np = np.load(os.path.join(test_index_path, "bucket_weights.npy"))
        index_a.close()

        path_b = test_index_path + "_fast"
        os.makedirs(path_b)

        index_b = search.FastPlaid(index=path_b, device="cpu")
        index_b.create(
            documents_embeddings=documents_embeddings,
            centroids=centroids,
            seed=42,
            avg_residual=avg_residual,
            cluster_threshold=cluster_threshold,
            bucket_cutoffs=bucket_cutoffs_np.tolist(),
            bucket_weights=bucket_weights_np.tolist(),
        )
        results_b = index_b.search(queries_embeddings=queries, top_k=10)
        index_b.close()
        shutil.rmtree(path_b)

        for q, (ra, rb) in enumerate(zip(results_a, results_b)):
            ids_a = [doc_id for doc_id, _ in ra]
            ids_b = [doc_id for doc_id, _ in rb]
            assert ids_a == ids_b, (
                f"query {q}: fast-path changed results {ids_a} != {ids_b}"
            )

    def test_fast_path_partial_override_still_works(self, test_index_path):
        """Providing only bucket overrides (no avg_residual/cluster_threshold) still works."""
        index = search.FastPlaid(index=test_index_path, device="cpu")
        documents_embeddings = [torch.randn(100, 128, device="cpu") for _ in range(50)]

        index.create(
            documents_embeddings=documents_embeddings,
            kmeans_niters=4,
            nbits=2,
            bucket_cutoffs=[-0.5, 0.0, 0.5],
            bucket_weights=[-0.75, -0.25, 0.25, 0.75],
        )

        queries = torch.randn(3, 30, 128, device="cpu")
        results = index.search(queries_embeddings=queries, top_k=10)
        assert len(results) == 3


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
