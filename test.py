import torch
from fast_plaid import search

fast_plaid = search.FastPlaid(index="index")

torch.manual_seed(0)

embedding_dim = 128

# x = torch.nn.functional.normalize(torch.randn(100, 128, embedding_dim), dim=-1)

# Index 100 documents, each with 300 tokens, each token is a 128-dim vector.
# fast_plaid.create(documents_embeddings=x)


# x = torch.nn.functional.normalize(torch.randn(100000, 128, embedding_dim), dim=-1)

# fast_plaid.update(documents_embeddings=x)

# Search for 2 queries, each with 50 tokens, each token is a 128-dim vector
scores = fast_plaid.search(
    queries_embeddings=torch.randn(10, 32, embedding_dim),
    top_k=10,
    subset=[72000],
    n_ivf_probe=1,
)

for score in scores:
    assert len(score) == 1

print(scores)
print("hello world")
