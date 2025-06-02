import os
import shutil
import time

import numpy as np
import torch
from fast_plaid import evaluation, search
from pylate import models


def test_scifact():
    """Run the full evaluation pipeline for a given dataset."""
    dataset_name = "scifact"

    shutil.rmtree(dataset_name, ignore_errors=True)
    os.makedirs(dataset_name, exist_ok=True)

    index = search.FastPlaid(
        index=dataset_name,
    )

    model = models.ColBERT(
        model_name_or_path="answerdotai/answerai-colbert-small-v1",
        query_length=32,
        document_length=300,
    )

    documents, queries, qrels, documents_ids = evaluation.load_beir(
        dataset_name=dataset_name,
        split="test",
    )

    documents_embeddings = model.encode(
        [doc["text"] for doc in documents],
        batch_size=128,
        is_query=False,
    )

    queries_embeddings = model.encode(
        list(queries.values()),
        batch_size=128,
        is_query=True,
    )

    queries_embeddings = torch.Tensor(np.array(queries_embeddings))
    documents_embeddings = [torch.tensor(doc_emb) for doc_emb in documents_embeddings]
    queries_embeddings = torch.cat(tensors=[queries_embeddings], dim=0)

    start_time = time.time()
    index.create(
        documents_embeddings=documents_embeddings,
        kmeans_niters=4,
    )
    print(f"\t✅ {dataset_name} indexing: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    scores = index.search(
        queries_embeddings=queries_embeddings,
    )
    print(f"\t✅ {dataset_name} search: {time.time() - start_time:.2f} seconds")

    results = [
        [
            {"id": documents_ids[doc_id], "score": score}
            for doc_id, score in query_scores
            if documents_ids[doc_id] != query_id
        ]
        for (query_id, _), query_scores in zip(queries.items(), scores, strict=False)
    ]

    scores = evaluation.evaluate(
        scores=results,
        qrels=qrels,
        queries=list(queries.values()),
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    # Print the scores for inspection
    print("Evaluation Scores:", scores)

    # Assert that the nDCG@10 score is greater than 0.73
    assert "ndcg@10" in scores, "nDCG@10 score not found in evaluation scores"

    assert isinstance(scores["ndcg@10"], float), "nDCG@10 score is not a float"

    assert scores["ndcg@10"] > 0.73, (
        f"nDCG@10 ({scores['ndcg@10']}) is not superior to 0.73"
    )
