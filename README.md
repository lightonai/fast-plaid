<div align="center">
  <h1>FastPlaid</h1>
</div>

<p align="center"><img width=500 src="https://github.com/lightonai/fast-plaid/blob/6184631dd9b9609efac8ce43e3e15be2efbb5355/docs/logo.png"/></p>

<div align="center">
    <a href="https://github.com/rust-lang/rust"><img src="https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white" alt="rust"></a>
    <a href="https://github.com/pyo3"><img src="https://img.shields.io/badge/PyO₃-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white" alt="PyO₃"></a>
    <a href="https://github.com/LaurentMazare/tch-rs"><img src="https://img.shields.io/badge/tch--rs-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white" alt="tch-rs"></a>
</div>

&nbsp;

<div align="center">
    <b>FastPlaid</b> - A High-Performance Engine for Multi-Vector Search
</div>

&nbsp;

## ⭐️ Overview

Traditional vector search relies on single, fixed-size embeddings (dense vectors) for documents and queries. While powerful, this approach can lose nuanced, token-level details.

- **Multi-vector search**, used in models like [ColBERT](https://github.com/lightonai/pylate) or [ColPali](https://github.com/illuin-tech/colpali), replaces a single document or image vector with a set of per-token vectors. This enables a "late interaction" mechanism, where fine-grained similarity is calculated term-by-term to boost retrieval accuracy.

- **Higher Accuracy:** By matching at a granular, token-level, FastPlaid captures subtle relevance that single-vector models simply miss.

- **PLAID:** stands for _Per-Token Late Interaction Dense Search_.

- **Blazing Performance**: Engineered in Rust and optimized for **GPUs**.

&nbsp;

## 💻 Installation

```bash
pip install fast-plaid
```

&nbsp;

## ⚡️ Quick Start

Get started with creating an index and performing a search in just a few lines of Python.

```python
import torch

from fast_plaid import search

fast_plaid = search.FastPlaid(index="index")

embedding_dim = 128

# Index 100 documents, each with 300 tokens, each token is a 128-dim vector.
fast_plaid.create(
    documents_embeddings=[torch.randn(300, embedding_dim) for _ in range(100)]
)

# Search for 2 queries, each with 50 tokens, each token is a 128-dim vector
scores = fast_plaid.search(
    queries_embeddings=torch.randn(2, 50, embedding_dim),
    top_k=10,
)

print(scores)
```

The output will be a list of lists, where each inner list contains tuples of (document_index, similarity_score) for the top top_k results for each query:

```python
[
    [
        (20, 1334.55),
        (91, 1299.57),
        (59, 1285.78),
        (10, 1273.53),
        (62, 1267.96),
        (44, 1265.55),
        (15, 1264.42),
        (34, 1261.19),
        (19, 1261.05),
        (86, 1260.94),
    ],
    [
        (58, 1313.85),
        (75, 1313.82),
        (79, 1305.32),
        (61, 1304.45),
        (64, 1303.67),
        (68, 1302.98),
        (66, 1301.23),
        (65, 1299.78),
    ],
]
```

_FastPlaid does not support index updates. Once an index is created, it is immutable. If you need to add or remove documents, you must create a new index. FastPlaid is optimized for GPUs but is compatible with CPUs._

&nbsp;

## 📊 Benchmarks

FastPlaid significantly outperforms the original PLAID engine across various datasets, delivering comparable accuracy with faster indexing and query speeds.

```python
                                   NDCG@10  Indexing Time (s) Queries per seconds (QPS)
dataset          size   library
arguana          8674   PLAID         0.46               4.30                     56.73
                        FastPlaid     0.46               4.72            155.25 (+174%)

fiqa             57638  PLAID         0.41              17.65                     48.13
                        FastPlaid     0.41              12.62            146.62 (+205%)

nfcorpus         3633   PLAID         0.37               2.30                     78.31
                        FastPlaid     0.37               2.10            243.42 (+211%)

quora            522931 PLAID         0.88              40.01                     43.06
                        FastPlaid     0.87              11.23            281.51 (+554%)

scidocs          25657  PLAID         0.19              13.32                     57.17
                        FastPlaid     0.18              10.86            157.47 (+175%)

scifact          5183   PLAID         0.74               3.43                     67.66
                        FastPlaid     0.75               3.16            190.08 (+181%)

trec-covid       171332 PLAID         0.84              69.46                     32.09
                        FastPlaid     0.83              45.19              54.11 (+69%)

webis-touche2020 382545 PLAID         0.25             128.11                     31.94
                        FastPlaid     0.24              74.50             70.15 (+120%)
```

_All benchmarks were performed on an H100 GPU. It's important to note that PLAID relies on Just-In-Time (JIT) compilation. This means the very first execution can exhibit longer runtimes. To ensure our performance analysis is representative, we've excluded these initial JIT-affected runs from the reported results. In contrast, FastPlaid does not employ JIT compilation, so its performance on the first run is directly indicative of its typical execution speed._

&nbsp;

## 📝 Citation

FastPlaid builds upon the groundbreaking work of the original PLAID engine [Santhanam, Keshav, et al.](https://arxiv.org/abs/2205.09707).

You can cite **FastPlaid** in your work as follows:

```bibtex
@misc{fastplaid2025,
  author = {Sourty, Raphaël},
  title = {FastPlaid: A High-Performance Engine for Multi-Vector Search},
  year = {2025},
  url = {https://github.com/lightonai/fast-plaid}
}
```

And for the original PLAID research:

```bibtex
@inproceedings{santhanam2022plaid,
  title={{PLAID}: an efficient engine for late interaction retrieval},
  author={Santhanam, Keshav and Khattab, Omar and Potts, Christopher and Zaharia, Matei},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={1747--1756},
  year={2022}
}
```

&nbsp;

## 📖 FastPlaid Class

The **`FastPlaid` class** is the core component for building and querying multi-vector search indexes. It's designed for **high performance**, especially when leveraging GPUs.

### Initialization

To create an instance of `FastPlaid`, you'll provide the directory where your index will be stored and specify the device(s) for computation.

```python
class FastPlaid:
    def __init__(
        self,
        index: str,
        device: str | list[str] | None = None,
    ) -> None:
```

```
index: str
    The file path to the directory where your index will be saved or loaded from.

device: str | list[str] | None = None
    Specifies the device(s) to use for computation.
    - If None (default) and CUDA is available, it defaults to "cuda".
    - If CUDA is not available, it defaults to "cpu".
    - Can be a single device string (e.g., "cuda:0" or "cpu").
    - Can be a list of device strings (e.g., ["cuda:0", "cuda:1"]).
    - If multiple GPUs are specified and available, multiprocessing is automatically set up for parallel execution.
      Remember to include your code within an `if __name__ == "__main__":` block for proper multiprocessing behavior.
```

### Creating an Index

The **`create` method** builds the multi-vector index from your document embeddings. It uses K-means clustering to organize your data for efficient retrieval.

```python
    def create(
        self,
        documents_embeddings: list[torch.Tensor],
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        nbits: int = 4,
    ) -> "FastPlaid":
```

```
documents_embeddings: list[torch.Tensor]
    A list where each element is a PyTorch tensor representing the multi-vector embedding for a single document.
    Each document's embedding should have a shape of `(num_tokens, embedding_dimension)`.

kmeans_niters: int = 4 (optional)
    The number of iterations for the K-means algorithm used during index creation.
    This influences the quality of the initial centroid assignments.

max_points_per_centroid: int = 256 (optional)
    The maximum number of points (token embeddings) that can be assigned to a single centroid during K-means.
    This helps in balancing the clusters.

nbits: int = 4 (optional)
    The number of bits to use for product quantization.
    This parameter controls the compression of your embeddings, impacting both index size and search speed.
    Lower values mean more compression and potentially faster searches but can reduce accuracy.
```

### Searching the Index

The **`search` method** lets you query the created index with your query embeddings and retrieve the most relevant documents.

```python
    def search(
        self,
        queries_embeddings: torch.Tensor,
        top_k: int = 10,
        batch_size: int = 1 << 18,
        n_full_scores: int = 8192,
        n_ivf_probe: int = 8,
        show_progress: bool = True,
    ) -> list[list[dict]]:
```

```
queries_embeddings: torch.Tensor
    A PyTorch tensor representing the multi-vector embeddings of your queries.
    Its shape should be `(num_queries, num_tokens_per_query, embedding_dimension)`.

top_k: int = 10 (optional)
    The number of top-scoring documents to retrieve for each query.

batch_size: int = 1 << 18 (optional)
    The internal batch size used for processing queries.
    A larger batch size might improve throughput on powerful GPUs but can consume more memory.

n_full_scores: int = 8192 (optional)
    The number of candidate documents for which full (re-ranked) scores are computed.
    This is a crucial parameter for accuracy; higher values lead to more accurate results but increase computation.

n_ivf_probe: int = 8 (optional)
    The number of inverted file list "probes" to perform during the search.
    This parameter controls the number of clusters to search within the index for each query.
    Higher values improve recall but increase search time.

show_progress: bool = True (optional)
    If set to `True`, a progress bar will be displayed during the search operation.
```
