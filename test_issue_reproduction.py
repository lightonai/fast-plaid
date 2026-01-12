import traceback
from datetime import date

import torch
from fast_plaid import filtering, search


def main():
    idx = search.FastPlaid("test_index")
    embedding_dim = 128

    # 2. Create initial documents with metadata
    initial_embeddings = [torch.randn(10, embedding_dim) for _ in range(3)]
    initial_metadata = [
        {"name": "Alice", "category": "A", "join_date": date(2023, 5, 17)},
        {"name": "Bob", "category": "B", "join_date": date(2021, 6, 21)},
        {"name": "Alex", "category": "A", "join_date": date(2023, 8, 1)},
    ]
    idx.create(
        documents_embeddings=initial_embeddings, metadata=initial_metadata
    )
    random_query = torch.randn(1, 10, embedding_dim)

    assert len(filtering.get(index=idx.index)) == 3, (
        "Expected 3 documents after initial creation"
    )
    assert len(idx.search(random_query, top_k=10)[0]) == 3, (
        "Expected 3 documents after initial creation"
    )

    new_embeddings = [torch.randn(10, embedding_dim) for _ in range(1)]
    new_metadata = [
        {"name": "Charlie", "category": "B", "join_date": date(2020, 3, 15)},
    ]
    idx.update(documents_embeddings=new_embeddings, metadata=new_metadata)

    assert len(filtering.get(index=idx.index)) == 4, (
        "Expected 4 documents after update"
    )
    assert len(idx.search(random_query, top_k=10)[0]) == 4, (
        "Expected 4 documents after update"
    )

    idx.delete(subset=[3])
    assert len(filtering.get(index=idx.index)) == 3, (
        "Expected 3 documents after deletion"
    )
    assert len(idx.search(random_query, top_k=10)[0]) == 3, (
        "Expected 3 documents after deletion"
    )

    idx.update(documents_embeddings=new_embeddings, metadata=new_metadata)

    assert len(filtering.get(index=idx.index)) == 4, (
        "Expected 4 documents after second update"
    )
    
    print("Metadata:", filtering.get(index=idx.index))
    results = idx.search(random_query, top_k=10)[0]
    print("Search results:", results)
    
    assert len(results) == 4, (
        f"Expected 4 documents after second update, but got {len(results)}"
    )

    print("SUCCESS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
    finally:
        import subprocess

        subprocess.run(["rm", "-rf", "test_index/"])
