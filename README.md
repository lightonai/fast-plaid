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
    <b>FastPlaid</b> is a high-performance search engine for multi-vector.
</div>

&nbsp;

## ⚡️ Overview

Traditional vector search relies on single, fixed-size embeddings (dense vectors) for documents and queries. While powerful, this approach can lose nuanced, token-level details.

* **Multi-vector search**, used in models like [ColBERT](https://github.com/lightonai/pylate) or [ColPali](https://github.com/illuin-tech/colpali), replaces a single document or image vector with a set of per-token vectors. This enables a "late interaction" mechanism, where fine-grained similarity is calculated term-by-term to boost retrieval accuracy.

* **Higher Accuracy:** By matching on a finer, token-level granularity, FastPlaid can capture relevance that single-vector models miss.
* **Performance:** Written in Rust, it offers blazing-fast indexing and search speeds.

* **PLAID:** stands for *Per-Token Late Interaction Dense Search* 

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

fast_plaid = search.FastPlaid(index="fast_plaid_index")

embedding_dim = 128

fast_plaid.create(
    documents_embeddings=[torch.randn(50, embedding_dim) for _ in range(100)]
)

scores = fast_plaid.search(
    queries_embeddings=torch.randn(2, 50, embedding_dim, dtype=torch.float16),
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


## Citation

FastPlaid is based on the original PLAID: [Santhanam, Keshav, et al.](https://arxiv.org/abs/2205.09707).

You can cite **FastPlaid** in your work as follows:

```bibtex
@misc{fastplaid2025,
  author = {Sourty, Raphaël},
  title = {FastPlaid: A High-Performance Engine for Multi-Vector Search},
  year = {2025},
  url = {https://github.com/lightonai/fast-plaid}
}

@inproceedings{santhanam2022plaid,
  title={{PLAID}: an efficient engine for late interaction retrieval},
  author={Santhanam, Keshav and Khattab, Omar and Potts, Christopher and Zaharia, Matei},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={1747--1756},
  year={2022}
}
```