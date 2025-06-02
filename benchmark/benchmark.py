import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
from fast_plaid import evaluation, search
from pylate import models

parser = argparse.ArgumentParser(
    description="Run Fast-PLAiD evaluation on a BEIR dataset."
)
parser.add_argument(
    "--dataset",
    type=str,
    default="scifact",
    help="Name of the dataset to process from the BEIR benchmark.",
)
args = parser.parse_args()
dataset_name = args.dataset

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

print(f"🚀 Starting evaluation for dataset: {dataset_name}")
model = models.ColBERT(
    model_name_or_path="answerdotai/answerai-colbert-small-v1",
    query_length=query_length.get(dataset_name, 32),
    document_length=300,
)

shutil.rmtree(dataset_name, ignore_errors=True)
os.makedirs(dataset_name, exist_ok=True)

print(f"📚 Loading BEIR dataset: {dataset_name}")
documents, queries, qrels, documents_ids = evaluation.load_beir(
    dataset_name=dataset_name,
    split="test",
)
num_queries = len(queries)

print(f"🧠 Encoding documents for {dataset_name}...")
documents_embeddings = model.encode(
    [document["text"] for document in documents],
    is_query=False,
)

print(f"🧠 Encoding queries for {dataset_name}...")
queries_embeddings = model.encode(
    list(queries.values()),
    is_query=True,
)

queries_embeddings = torch.Tensor(np.array(queries_embeddings))
documents_embeddings = [torch.tensor(doc_emb) for doc_emb in documents_embeddings]
queries_embeddings = torch.cat(tensors=[queries_embeddings], dim=0)

index = search.FastPlaid(index=dataset_name)
print(f"🏗️  Building index for {dataset_name}...")
start_index = time.time()
index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
end_index = time.time()
indexing_time = end_index - start_index
print(f"\t✅ {dataset_name} indexing: {indexing_time:.2f} seconds")

print(f"🔍 Searching on {dataset_name}...")
start_search = time.time()
scores = index.search(queries_embeddings=queries_embeddings)
end_search = time.time()
search_time = end_search - start_search
queries_per_second = num_queries / search_time if search_time > 0 else 0
print(
    f"\t✅ {dataset_name} search: {search_time:.2f} seconds ({queries_per_second:.2f} QPS)"
)

queries_embeddings = torch.cat(
    ([queries_embeddings] * ((100 // queries_embeddings.shape[0]) + 1))[:3000]
)

print(f"🔍 50_000 queries on {dataset_name}...")
start_search = time.time()
_ = index.search(queries_embeddings=queries_embeddings)
end_search = time.time()
heavy_search_time = end_search - start_search
queries_per_second = queries_embeddings.shape[0] / heavy_search_time
print(
    f"\t✅ {dataset_name} search: {heavy_search_time:.2f} seconds ({queries_per_second:.2f} QPS)"
)

results = []
for (query_id, _), query_scores in zip(queries.items(), scores, strict=True):
    results.append(
        [
            {"id": documents_ids[document_id], "score": score}
            for document_id, score in query_scores
            if documents_ids[document_id] != query_id
        ]
    )

print(f"📊 Calculating metrics for {dataset_name}...")
evaluation_scores = evaluation.evaluate(
    scores=results,
    qrels=qrels,
    queries=list(queries.values()),
    metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
)

print(f"\n--- 📈 Final Scores for {dataset_name} ---")
print(evaluation_scores)

output_dir = "./benchmark"
os.makedirs(output_dir, exist_ok=True)

output_data = {
    "dataset": dataset_name,
    "indexing": round(indexing_time, 3),
    "search": round(search_time, 3),
    "qps": round(queries_per_second, 2),
    "size": len(documents),
    "queries": num_queries,
    "scores": evaluation_scores,
}

output_filepath = os.path.join(output_dir, f"{dataset_name}.json")
print(f"💾 Exporting results to {output_filepath}")
with open(output_filepath, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"🎉 Finished evaluation for dataset: {dataset_name}\n")
