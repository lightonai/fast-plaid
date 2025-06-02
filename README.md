<div align="center">
<h1>FastPlaid\</h1>
</div>

<p align="center">
<img src="[https://img.shields.io/pypi/v/fast-plaid?style=for-the-badge](https://img.shields.io/pypi/v/fast-plaid?style=for-the-badge)" alt="PyPI version">
<img src="[https://img.shields.io/pypi/pyversions/fast-plaid?style=for-the-badge](https://img.shields.io/pypi/pyversions/fast-plaid?style=for-the-badge)" alt="Python versions">
<img src="[https://img.shields.io/badge/Made%20with-Rust-orange?style=for-the-badge\&logo=rust](https://img.shields.io/badge/Made%20with-Rust-orange?style=for-the-badge&logo=rust)" alt="Made with Rust">
</p>

A high-performance document retrieval toolkit using a ColBERT-style late interaction model, implemented in Rust with Python bindings.

-----

## ⚙️ Installation

Install the package directly from PyPI:

```bash
pip install fast-plaid
```

-----

## 🚀 Quick Start

Here's how to get started with creating an index and performing a search in just a few lines of Python.

```python
import torch
from fast_plaid import search

fast_plaid = search.FastPlaid(index="fast_plaid_index")

embedding_dim = 128

docs_embeddings = [torch.randn(50, embedding_dim) for _ in range(100)]

fast_plaid.create(documents_embeddings=docs_embeddings)

queries_embeddings = torch.randn(2, 50, embedding_dim, dtype=torch.float16)

scores = fast_plaid.search(
    queries_embeddings=queries_embeddings,
    top_k=10,
)

print(scores)
```


```python
[
    [
        (20, 1334.5103759765625),
        (91, 1299.576171875),
        (59, 1285.788818359375),
        (10, 1273.534912109375),
        (62, 1267.9666748046875),
        (44, 1265.5655517578125),
        (15, 1264.426025390625),
        (34, 1261.19775390625),
        (19, 1261.0517578125),
        (86, 1260.94140625),
    ],
    [
        (58, 1313.8587646484375),
        (75, 1313.829833984375),
        (79, 1305.322509765625),
        (59, 1299.12158203125),
        (55, 1293.456787109375),
        (44, 1288.419189453125),
        (67, 1283.658935546875),
        (60, 1283.2884521484375),
        (53, 1282.522216796875),
        (9, 1280.863037109375),
    ],
]
```
