import time
import torch
import sys
from fast_plaid import search

# The `if __name__ == "__main__":` block is essential for multi-GPU
# search, as FastPlaid uses multiprocessing to parallelize the
# work, and this ensures the code only runs in the main process.
if __name__ == "__main__":

    # 1. Check for and configure multiple GPUs
    if not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        print("Please run this on a machine with NVIDIA GPUs and CUDA.")
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"Warning: Only found {gpu_count} GPU.")
        print("This script will run, but it won't be a *multi-GPU* test.")
        print("To test multi-GPU, please run on a machine with 2+ GPUs.")
        
    # Create a list of all available CUDA devices
    devices = [f"cuda:{i}" for i in range(gpu_count)]
    print(f"Found {gpu_count} GPUs. Using devices: {devices}")

    # 2. Initialize FastPlaid with the list of devices
    # This tells FastPlaid to use all these GPUs for the search.
    fast_plaid = search.FastPlaid(
        index="multigpu_test_index", 
        device=devices[0]
    )

    # 3. Create a sample index
    embedding_dim = 128
    num_documents = 500  # A larger number of documents for a better test
    doc_tokens = 400      # Number of tokens per document
    
    print(f"Creating a test index with {num_documents} documents...")
    
    # Generate document embeddings on CPU
    documents_embeddings = [
        torch.randn(doc_tokens, embedding_dim) for _ in range(num_documents)
    ]

    fast_plaid.create(
        documents_embeddings=documents_embeddings,
    )
    print("Index creation complete.")

    # 4. Prepare a large batch of queries
    num_queries = 10000   # A large number of queries to see the speedup
    query_tokens = 30     # Number of tokens per query
    
    print(f"Preparing {num_queries} queries for search...")
    
    # Generate query embeddings on CPU
    queries_embeddings = torch.randn(num_queries, query_tokens, embedding_dim)

    # 5. Run the multi-GPU search
    print(f"Starting search on {len(devices)} GPU(s)...")
    start = time.time()

    scores = fast_plaid.search(
        queries_embeddings=queries_embeddings,
        top_k=10,
        subset=[i for i in range(50)]
    )

    end = time.time()

    print("\n--- Search Complete ---")
    print(f"Searched {num_queries} queries against {num_documents} documents.")
    print(f"Multi-GPU search completed in {end - start:.4f} seconds.")
    
    # Optional: Print a small sample of the results
    print(f"\nRetrieved scores for {len(scores)} queries.")
    print("Example scores for the first query:")
    print(scores[:100])