import os
import shutil
import time

import numpy as np
import torch
from fast_plaid import evaluation, search
from pylate import models

dataset_name = "scifact"

query_length = {
    "quora": 32,
    "climate-fever": 64,
    "nq": 32,
    "msmarco": 32,
    "hotpotqa": 32,
    "nfcorpus": 32,
    "scifact": 48,
    "trec-covid": 48,
    "fiqa": 32,
    "arguana": 64,
    "scidocs": 48,
    "dbpedia-entity": 32,
    "webis-touche2020": 32,
    "fever": 32,
}


model = models.ColBERT(
    model_name_or_path="answerdotai/answerai-colbert-small-v1",
    query_length=query_length[dataset_name],
    document_length=300,
)


shutil.rmtree(dataset_name, ignore_errors=True)
os.makedirs(dataset_name, exist_ok=True)

documents, queries, qrels, documents_ids = evaluation.load_beir(
    dataset_name=dataset_name,
    split="test",
)

documents_embeddings = model.encode(
    [document["text"] for document in documents],
    batch_size=128,
    is_query=False,
)

queries_embeddings = model.encode(
    list(queries.values()),
    batch_size=128,
    is_query=True,
)

queries_embeddings = torch.Tensor(np.array(queries_embeddings))

documents_embeddings = [
    torch.tensor(document_embedding) for document_embedding in documents_embeddings
]

queries_embeddings = torch.cat(tensors=[queries_embeddings], dim=0)

index = search.FastPlaid(
    index=dataset_name,
)

start = time.time()

index.create(
    documents_embeddings=documents_embeddings,
    kmeans_niters=4,
)

end = time.time()

index = search.FastPlaid(
    index=dataset_name,
)

print(f"\t✅ {dataset_name} indexing: {end - start:.2f} seconds")

start = time.time()
scores = index.search(
    queries_embeddings=queries_embeddings,
)
end = time.time()
print(f"\t✅ {dataset_name} search: {end - start:.2f} seconds")


results = []
for (query_id, _), query_scores in zip(queries.items(), scores, strict=False):
    results.append(
        [
            {"id": documents_ids[document_id], "score": score}
            for document_id, score in query_scores
            if documents_ids[document_id] != query_id
        ]
    )


evaluation_scores = evaluation.evaluate(
    scores=results,
    qrels=qrels,
    queries=list(queries.values()),
    metrics=[
        "map",
        "ndcg@10",
        "ndcg@100",
        "recall@10",
        "recall@100",
    ],
)

print(evaluation_scores)
