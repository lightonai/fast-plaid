import shutil
import time

import torch
from fast_plaid import evaluation, search
from pylate import models

dataset_name = "scifact"

print(f"🚀 Starting evaluation for dataset: {dataset_name}")
model = models.ColBERT(
    model_name_or_path="answerdotai/answerai-colbert-small-v1",
    query_length=42,
    document_length=300,
)

shutil.rmtree(dataset_name, ignore_errors=True)


documents, queries, qrels, documents_ids = evaluation.load_beir(
    dataset_name=dataset_name,
    split="test",
)

documents = documents[:5000]


documents_embeddings = model.encode(
    [document["text"] for document in documents],
    is_query=False,
)

queries_embeddings = model.encode(
    list(queries.values()),
    is_query=True,
)

documents_embeddings = [torch.tensor(doc_emb) for doc_emb in documents_embeddings]

index = search.FastPlaid(index=dataset_name, device="cpu")


start_index = time.time()

index.create(
    documents_embeddings=documents_embeddings,
    kmeans_niters=4,
    n_samples_kmeans=100,
)

end_index = time.time()

indexing_time = end_index - start_index

print(f"\t✅ {dataset_name} indexing: {indexing_time:.2f} seconds")
