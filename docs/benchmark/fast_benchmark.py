import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from fast_plaid import evaluation, search
from pylate import models
from tqdm.auto import tqdm


def encode_worker(
    gpu_id: int,
    texts: list[str],
    model_name: str,
    query_length: int,
    document_length: int,
    is_query: bool,
) -> list[np.ndarray]:
    """A separate process that loads a model onto a specific GPU and encodes texts."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    import logging

    logging.getLogger("pylate.models.colbert").setLevel(logging.ERROR)

    try:
        model = models.ColBERT(
            model_name_or_path=model_name,
            query_length=query_length,
            document_length=document_length,
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"WORKER {gpu_id} FAILED to load model: {e}")
        return []

    try:
        batch_size = 512
        embeddings = []

        desc = f"GPU {gpu_id} {'Queries' if is_query else 'Docs'}"
        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), batch_size),
                desc=desc,
                position=gpu_id,
                leave=False,
            ):
                batch_texts = texts[i : i + batch_size]
                batch_embeddings = model.encode(
                    batch_texts,
                    is_query=is_query,
                )
                embeddings.extend(batch_embeddings)

    except Exception as e:
        print(f"WORKER {gpu_id} FAILED during encoding: {e}")
        del model
        torch.cuda.empty_cache()
        return []

    del model
    torch.cuda.empty_cache()
    return embeddings


def run_evaluation():
    parser = argparse.ArgumentParser(
        description="Run Fast-PLAiD evaluation on a BEIR dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco",
        help="Name of the dataset to process from the BEIR benchmark.",
    )
    args = parser.parse_args()
    dataset_name = args.dataset

    query_length_map = {
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

    MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
    QUERY_LENGTH = query_length_map.get(dataset_name, 32)
    DOC_LENGTH = 300

    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS == 0:
        print("❌ No GPUs detected. This script requires GPUs.")
        return

    print(f"🚀 Starting evaluation for dataset: {dataset_name}")
    print(f"🔥 Using {NUM_GPUS} GPUs for parallel encoding (1 process per GPU).")

    shutil.rmtree(dataset_name, ignore_errors=True)
    os.makedirs(dataset_name, exist_ok=True)
    shutil.rmtree(f"{dataset_name}_pylate", ignore_errors=True)

    print(f"📚 Loading BEIR dataset: {dataset_name}")
    documents, queries, qrels, documents_ids = evaluation.load_beir(
        dataset_name=dataset_name,
        split="dev" if "msmarco" in dataset_name else "test",
    )
    num_queries = len(queries)
    document_texts = [document["text"] for document in documents]
    query_texts = list(queries.values())

    print(f"🧠 Encoding {len(document_texts)} documents (in parallel)...")
    doc_chunks = np.array_split(document_texts, NUM_GPUS)

    doc_worker_args = []
    for gpu_id in range(NUM_GPUS):
        doc_worker_args.append(
            (
                gpu_id,
                doc_chunks[gpu_id].tolist(),
                MODEL_NAME,
                QUERY_LENGTH,
                DOC_LENGTH,
                False,  # is_query = False
            )
        )

    start_time = time.time()
    with mp.Pool(NUM_GPUS) as pool:
        doc_results_list = pool.starmap(encode_worker, doc_worker_args)

    documents_embeddings = [item for sublist in doc_results_list for item in sublist]
    end_time = time.time()
    print(f"✅ Document encoding finished in {end_time - start_time:.2f} seconds.")

    if len(documents_embeddings) != len(document_texts):
        print(
            f"❌ Error: Expected {len(document_texts)} doc embeddings, but got {len(documents_embeddings)}."
        )
        return

    print(f"🧠 Encoding {len(query_texts)} queries (in parallel)...")
    query_chunks = np.array_split(query_texts, NUM_GPUS)

    query_worker_args = []
    for gpu_id in range(NUM_GPUS):
        query_worker_args.append(
            (
                gpu_id,
                query_chunks[gpu_id].tolist(),
                MODEL_NAME,
                QUERY_LENGTH,
                DOC_LENGTH,
                True,  # is_query = True
            )
        )

    start_time = time.time()
    with mp.Pool(NUM_GPUS) as pool:
        query_results_list = pool.starmap(encode_worker, query_worker_args)

    queries_embeddings = [item for sublist in query_results_list for item in sublist]
    end_time = time.time()
    print(f"✅ Query encoding finished in {end_time - start_time:.2f} seconds.")

    if len(queries_embeddings) != len(query_texts):
        print(
            f"❌ Error: Expected {len(query_texts)} query embeddings, but got {len(queries_embeddings)}."
        )
        return

    queries_embeddings_np = torch.Tensor(np.array(queries_embeddings))
    documents_embeddings = [torch.tensor(doc_emb) for doc_emb in documents_embeddings]
    queries_embeddings = torch.cat(tensors=[queries_embeddings_np], dim=0)

    index_device = "cuda:1"
    if NUM_GPUS < 2:
        print(f"⚠️ Warning: GPU count is {NUM_GPUS}. Defaulting index to cuda:0.")
        index_device = "cuda:0"

    print(f"Setting index and search device to: {index_device}")

    index = search.FastPlaid(
        index=os.path.join("benchmark", dataset_name), device=index_device
    )
    print(f"🏗️  Building index for {dataset_name} on {index_device}...")
    start_index = time.time()

    index.create(
        documents_embeddings=documents_embeddings,
        kmeans_niters=4,
    )
    end_index = time.time()
    indexing_time = end_index - start_index
    print(f"\t✅ {dataset_name} indexing: {indexing_time:.2f} seconds")

    print(f"🔍 Searching on {dataset_name}...")
    start_search = time.time()
    # Move queries to the index device (cuda:1) for searching
    scores = index.search(
        queries_embeddings=queries_embeddings,
        top_k=20,
    )
    end_search = time.time()
    search_time = end_search - start_time
    print(f"\t✅ {dataset_name} search: {search_time:.2f} seconds")

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
        "size": len(documents),
        "queries": num_queries,
        "scores": evaluation_scores,
    }

    output_filepath = os.path.join(output_dir, f"{dataset_name}.json")
    print(f"💾 Exporting results to {output_filepath}")
    with open(output_filepath, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"🎉 Finished evaluation for dataset: {dataset_name}\n")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(
            f"Could not set 'spawn' start method. This is required for CUDA. Error: {e}"
        )

    run_evaluation()
