from collections import defaultdict


def add_duplicates(queries: list[str], scores: list[list[dict]]) -> list:
    """Replicates scores for duplicated queries.

    Args:
    ----
    queries:
        A list of queries, potentially with duplicates.
    scores:
        A list of scores corresponding to unique queries.

    """
    query_counts: defaultdict[str, int] = defaultdict(int)
    for query in queries:
        query_counts[query] += 1

    query_to_result = {}
    for i, query in enumerate(iterable=queries):
        if query not in query_to_result:
            query_to_result[query] = scores[i]

    duplicated_scores = []
    for query in queries:
        if query in query_to_result:
            duplicated_scores.append(query_to_result[query])

    return duplicated_scores


def load_beir(dataset_name: str, split: str = "test") -> tuple[list, list, dict, dict]:
    """Download and loads a BEIR dataset.

    Args:
    ----
    dataset_name:
        The name of the BEIR dataset (e.g., "scifact").
    split:
        The dataset split to load (e.g., "test").

    Examples:
    --------
    >>> from fast_plaid import evaluation

    >>> documents, queries, qrels = evaluation.load_beir(
    ...     "scifact",
    ...     split="test",
    ... )
    >>> len(documents)
    5183
    >>> len(queries)
    300
    >>> len(qrels)
    300

    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = util.download_and_unzip(
        url=f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
        out_dir="./evaluation_datasets/",
    )

    documents, queries, qrels = GenericDataLoader(data_folder=data_path).load(
        split=split
    )

    documents = [
        {
            "id": document_id,
            "text": f"{document['title']} {document['text']}".strip()
            if "title" in document
            else document["text"].strip(),
        }
        for document_id, document in documents.items()
    ]

    qrels = {
        queries[query_id]: query_documents
        for query_id, query_documents in qrels.items()
    }

    documents_ids = {index: document["id"] for index, document in enumerate(documents)}

    return documents, queries, qrels, documents_ids


def evaluate(
    scores: list[list[dict]],
    qrels: dict,
    queries: list[str],
    metrics: list | None = None,
) -> dict[str, float]:
    """Evaluate retrieval scores against ground truth relevance judgments (qrels).

    Args:
    ----
    scores:
        A list of ranked documents and their scores for each query.
    qrels:
        A dictionary mapping queries to relevant document IDs.
    queries:
        A list of the queries used for retrieval.
    metrics:
        A list of metrics to compute (e.g., ["ndcg@10", "hits@1"]).

    Examples:
    --------
    >>> from fast_plaid import evaluation

    >>> scores = [
    ...     [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.8}],
    ...     [{"id": "3", "score": 0.7}, {"id": "4", "score": 0.6}],
    ... ]

    >>> qrels = {
    ...     "query1": {"1": True, "2": True},
    ...     "query2": {"3": True, "4": True},
    ... }

    >>> queries = ["query1", "query2"]

    >>> results = evaluation.evaluate(
    ...     scores=scores,
    ...     qrels=qrels,
    ...     queries=queries,
    ...     metrics=["ndcg@10", "hits@1"],
    ... )

    """
    from ranx import Qrels, Run, evaluate

    if len(queries) > len(scores):
        scores = add_duplicates(queries=queries, scores=scores)

    qrels = Qrels(qrels=qrels)

    run_dict = {
        query: {
            match["id"]: match["score"]
            for rank, match in enumerate(iterable=query_matchs)
        }
        for query, query_matchs in zip(queries, scores, strict=False)
    }

    run = Run(run=run_dict)

    if not metrics:
        metrics = ["ndcg@10"] + [f"hits@{k}" for k in [1, 2, 3, 4, 5, 10]]

    return evaluate(
        qrels=qrels,
        run=run,
        metrics=metrics,
        make_comparable=True,
    )
