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

## PyTorch Compatibility

FastPlaid is available in multiple versions to support different PyTorch versions:

| FastPlaid Version | PyTorch Version | Installation Command                |
| ----------------- | --------------- | ----------------------------------- |
| 1.3.0.290         | 2.9.0           | `pip install fast-plaid==1.3.0.290` |
| 1.3.0.280         | 2.8.0           | `pip install fast-plaid==1.3.0.280` |
| 1.3.0.271         | 2.7.1           | `pip install fast-plaid==1.3.0.271` |
| 1.3.0.270         | 2.7.0           | `pip install fast-plaid==1.3.0.270` |

### Adding FastPlaid as a Dependency

You can add FastPlaid to your project dependencies with version ranges to ensure compatibility:

**For requirements.txt:**

```
fast-plaid>=1.3.0.270,<=1.3.0.290
```

**For pyproject.toml:**

```toml
[project]
dependencies = [
    "fast-plaid>=1.3.0.270,<=1.3.0.290"
]
```

**For setup.py:**

```python
install_requires=[
    "fast-plaid>=1.3.0.270,<=1.3.0.290"
]
```

Choose the appropriate version range based on your PyTorch requirements.

**Building from Source:**

```python
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install git+https://github.com/lightonai/fast-plaid.git
```

&nbsp;

## ⚡️ Quick Start

Get started with creating an index and performing a search in just a few lines of Python.

```python
import torch

from fast_plaid import search

fast_plaid = search.FastPlaid(index="index", device="cpu") # or "cuda" for GPU.
# On CPU, specifying device speeds up initialization.

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

## 🗂️ Update an Index

```python
import torch

from fast_plaid import search

fast_plaid = search.FastPlaid(index="index") # Load an existing index

embedding_dim = 128

fast_plaid.update(
    documents_embeddings=[torch.randn(300, embedding_dim) for _ in range(100)]
)

scores = fast_plaid.search(
    queries_embeddings=torch.randn(2, 50, embedding_dim),
    top_k=10,
)

print(scores)
```

It is highly recommended to create your initial index with a large and representative sample of your data for optimal performance and accuracy. The **`.create()` method** establishes the fundamental structure of the index by calculating centroids that are specifically tailored to the distribution of this initial dataset.

The **`.update()` method**, designed for efficiency, **does not re-compute these centroids**. Instead, it places new documents into the existing structure. If you frequently update the index with large volumes of data that have a different statistical distribution than the original set, you may experience "drift." This means the fixed centroids become less representative of the total collection, potentially leading to sub-optimal data partitioning and a gradual decline in retrieval accuracy over time. Therefore, building a robust initial index is key to its long-term health. If you find that your data distribution changes significantly, consider periodically re-creating the index with a new, representative sample to maintain optimal performance or avoid using the `.update()` method and rely on the **`.create()` method** which will delete the existing index and re-create it from scratch. To update an existing embedding, you should delete it first and then add the new version with the `.update()` method.

&nbsp;

## 🔎 Filtering

You can restrict your search to a specific subset of documents by using the `subset` parameter in the `.search()` method. This is useful for implementing metadata filtering or searching within a pre-defined collection.

The `subset` parameter accepts a list of IDs. These IDs correspond directly to the order of insertion, starting from 0. For example, if you index 100 documents with `.create()`, they will have IDs `0` through `99`. If you then add `50` more documents with `.update()`, they will be assigned the subsequent IDs `100` through `149`.

You can provide a single list of IDs to apply the same filter to all queries, or a list of lists to specify a different filter for each query.

```python
import torch
from fast_plaid import search


fast_plaid = search.FastPlaid(index="index") # Load an existing index

# Apply a single filter to all queries
# Search for the top 5 results only within documents [2, 5, 10, 15, 18]
scores = fast_plaid.search(
    queries_embeddings=torch.randn(2, 50, 128), # 2 queries
    top_k=5,
    subset=[2, 5, 10, 15, 18]
)

print(scores)

# Apply a different filter for each query
# Query 1: search within documents [0, 1, 2, 3, 4]
# Query 2: search within documents [10, 11, 12, 13, 14]
scores = fast_plaid.search(
    queries_embeddings=torch.randn(2, 50, 128), # 2 queries
    top_k=5,
    subset=[
        [0, 1, 2, 3, 4],
        [10, 11, 12, 13, 14]
    ]
)

print(scores)
```

Providing a `subset` filter can significantly speed up the search process, especially when the subset is much smaller than the total number of indexed documents. In order to increase the recall when applying drastic filtering, consider increasing the `n_ivf_probe` parameter in the `.search()` method (default: 8). It controls the number of clusters to search within the index for each query. Only clusters that contain documents from the provided subset are considered during the search.

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

## ⚖️ Settings Trade-offs

### Indexing

```python
Parameter         Default     Speed                        Accuracy                     Description
n_samples_kmeans  None        lower = faster               lower = less precise         Number of samples to compute centroids
nbits             4           lower  = faster              lower  = less precise        product quantization bits
kmeans_niters     4           higher = slower indexing     higher = better clusters     K-means iterations
```

### Search

```python
Parameter         Default     Speed               Accuracy                    Description
n_ivf_probe       8           higher = slower     higher = better recall      cluster probes per query
n_full_scores     4096        higher = slower     higher = better ranking     candidates for full scoring
```

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
    - You should specify the device to accelerate the initialization of FastPlaid index especially when using CPUs.
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
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        nbits: int = 4,
        n_samples_kmeans: int | None = None,
        batch_size: int = 25_000,
        seed: int = 42,
        use_triton_kmeans: bool | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> "FastPlaid":
```

```
documents_embeddings: list[torch.Tensor] | torch.Tensor
    A list where each element is a PyTorch tensor representing the multi-vector embedding for a single document.
    Each document's embedding should have a shape of `(num_tokens, embedding_dimension)`. Can also be a single tensor of shape `(num_documents, num_tokens, embedding_dimension)`.

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

n_samples_kmeans: int | None = None (optional)
    The number of samples to use for K-means clustering.
    If `None`, it defaults to a value based on the number of documents.
    This parameter can be adjusted to balance between speed, memory usage and
    clustering quality. If you have a large dataset, you might want to set this to a
    smaller value to speed up the indexing process and save some memory.

batch_size: int = 25_000 (optional)
    Batch size for processing embeddings during index creation.

seed: int = 42 (optional)
    Seed for the random number generator used in index creation.
    Setting this ensures reproducible results across multiple runs.

use_triton_kmeans: bool | None = None (optional)
    Whether to use the Triton-based K-means implementation.
    If `None`, it will be set to True if the device is not "cpu".
    Triton-based implementation can provide better performance on GPUs.
    Set to False to ensure perfectly reproducible results across runs.

metadata: list[dict[str, Any]] | None = None (optional)
    An optional list of metadata dictionaries corresponding to each document being indexed.
    Each dictionary can contain arbitrary key-value pairs that you want to associate with the document.
    If provided, the length of this list must match the number of documents being indexed.
    The metadata will be stored in a SQLite database within the index directory for filtering during searches.
```

### Updating the Index

The **`update` method** provides an efficient way to add new documents to an existing index without rebuilding it from scratch. This is significantly faster than calling .create() again, as it reuses the existing quantization configuration and only processes the new documents. The centroids and quantization parameters remain unchanged, **this might lead to a slight decrease in accuracy compared to a full re-indexing**. To update an existing embedding, you should delete it first and then add the new version with the `.update()` method. Warning, when using the `delete` method, the remaining documents are re-indexed to maintain a sequential order. If you delete document k, all documents with id > k will have their id decreased by 1.

```python
    def update(
        self,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
        metadata: list[dict[str, Any]] | None = None,
        batch_size: int = 25_000,
    ) -> "FastPlaid":
```

```
documents_embeddings: list[torch.Tensor]
    A list where each element is a PyTorch tensor representing the multi-vector embedding for a single document.
    Each document's embedding should have a shape of `(num_tokens, embedding_dimension)`.
    This method will add these new embeddings to the existing index.

metadata: list[dict[str, Any]] | None = None
    An optional list of metadata dictionaries corresponding to each new document being added.
    Each dictionary can contain arbitrary key-value pairs that you want to associate with the document.
    If provided, the length of this list must match the number of new documents being added.
    The metadata will be stored in a SQLite database within the index directory for filtering during searches.

batch_size: int = 25_000 (optional)
    Batch size for processing embeddings during the update.
```

### Searching the Index

The **`search` method** lets you query the created index with your query embeddings and retrieve the most relevant documents.

```python
    def search(
        self,
        queries_embeddings: torch.Tensor | list[torch.Tensor],
        top_k: int = 10,
        batch_size: int = 25_000,
        n_full_scores: int = 4096,
        n_ivf_probe: int = 8,
        show_progress: bool = True,
        subset: list[list[int]] | list[int] | None = None,
    ) -> list[list[tuple[int, float]]]:
```

```
queries_embeddings: torch.Tensor | list[torch.Tensor]
    A PyTorch tensor representing the multi-vector embeddings of your queries.
    Its shape should be `(num_queries, num_tokens_per_query, embedding_dimension)`.
    Can also be a list of tensors, each representing a separate query. All tensors in the list must have the same embedding dimension.

top_k: int = 10 (optional)
    The number of top-scoring documents to retrieve for each query.

batch_size: int = 25_000 (optional)
    The internal batch size used for processing queries.
    A larger batch size might improve throughput on powerful GPUs but can consume more memory.

n_full_scores: int = 4096 (optional)
    The number of candidate documents for which full (re-ranked) scores are computed.
    This is a crucial parameter for accuracy; higher values lead to more accurate results but increase computation.

n_ivf_probe: int = 8 (optional)
    The number of inverted file list "probes" to perform during the search.
    This parameter controls the number of clusters to search within the index for each query.
    Higher values improve recall but increase search time.

show_progress: bool = True (optional)
    If set to `True`, a progress bar will be displayed during the search operation.

subset: list[list[int]] | list[int] | None = None (optional)
    An optional list of lists of integers or a single list of integers. If provided, the search
    for each query will be restricted to the document IDs in the corresponding inner list.
    - If a single list is provided, the same filter will be applied to all queries.
    - If a list of lists is provided, each inner list corresponds to the filter for each query.
    - Document IDs correspond to the order of insertion, starting from 0.
```

### Deleting from the Index

The **`delete` method** allows to permanently remove embeddings from the index based on their insertion order IDs. If a metadata database exists, the corresponding entries will also be automatically removed. To update an existing embedding, you should delete it first and then add the new version with the `.update()` method. Warning, when using the `delete` method, the remaining documents are re-indexed to maintain a sequential order. If you delete document k, all documents with id > k will have their id decreased by 1.

```python
    def delete(
        self,
        subset: list[int],
    ) -> "FastPlaid":
```

```
subset: list[int]
    A list of embeddings IDs to delete from the index. The IDs are based on the original
    order of insertion, starting from 0. After deletion, the remaining documents are
    re-indexed to maintain a sequential order.
```

## Contributing

Any contributions to FastPlaid are welcome! If you have ideas for improvements, bug fixes, or new features, please open an issue or submit a pull request. We are particularly interested in:

- Re-computing centroids when using the `.update()` method to maintain optimal performance.
- Additional algorithms for multi-vector search.
- New search outputs formats for better integration with existing systems.

&nbsp;

## 🗂️ Built-in SQLite Filtering

FastPlaid includes a lightweight, optional, built-in metadata filtering engine powered by SQLite. When you provide the metadata parameter during .create() or .update(), FastPlaid automatically stores this information in a searchable database within your index directory.

You can then use the fast_plaid.filtering.where() function to query this database using standard SQL conditions. This function returns a list of embeddings IDs that match your criteria, which you can pass directly to the subset parameter of the .search() method to pre-filter your search.

```python
from datetime import date

import torch
from fast_plaid import filtering, search

# 1. Initialize the FastPlaid index
fast_plaid = search.FastPlaid(index="metadata_index")
embedding_dim = 128

# 2. Create initial documents with metadata
initial_embeddings = [torch.randn(10, embedding_dim) for _ in range(3)]
initial_metadata = [
    {"name": "Alice", "category": "A", "join_date": date(2023, 5, 17)},
    {"name": "Bob", "category": "B", "join_date": date(2021, 6, 21)},
    {"name": "Alex", "category": "A", "join_date": date(2023, 8, 1)},
]

fast_plaid.create(documents_embeddings=initial_embeddings, metadata=initial_metadata)

# 3. Update the index with new documents and metadata
new_embeddings = [torch.randn(10, embedding_dim) for _ in range(2)]
new_metadata = [
    {"name": "Charlie", "category": "B", "join_date": date(2020, 3, 15)},
    {
        "name": "Amanda",
        "category": "A",
        "join_date": date(2024, 1, 10),
        "status": "active",
    },
]

fast_plaid.update(documents_embeddings=new_embeddings, metadata=new_metadata)

# 4. Use filtering.where to get the corresponding rows of the FastPlaid index
# which match the SQL condition
subset = filtering.where(
    index="metadata_index",
    condition="name LIKE ? AND join_date > ?",
    parameters=("A%", "2022-12-31"),
)

# 5. Perform a search restricted to the filtered subset
query_embedding = torch.randn(1, 5, embedding_dim)
scores = fast_plaid.search(queries_embeddings=query_embedding, top_k=3, subset=subset)

print("Search results within the subset:")
print(scores)

# 5. Access to the metadata of the retrieved documents
for match in scores:
    print("Metadata of matched documents:")
    print(filtering.get(index="metadata_index", subset=[subset for subset, _ in match]))
```

You can also rely on the existing subset parameter of the .search() method to filter candidates based
on the order of insertion or rely on an external filtering system as providing the metadata parameter is optional.
