import json
import os
from collections.abc import Iterable

import numpy as np
import torch


def embeddings(index_path: str, subset: Iterable[int]) -> list[torch.Tensor]:
    """Reconstructs original embeddings for the specified document IDs.

    This function minimizes RAM usage by:
    1. Using memory-mapping (mmap) for large index files.
    2. Reconstructing vectors one document at a time.
    3. Accumulating results directly into the final list.

    Args:
        index_path: Path to the FastPlaid index directory.
        doc_ids: A list or range of document IDs (0-based) to reconstruct.

    Returns:
        List[torch.Tensor]: A list of reconstructed document embeddings.

    """
    # 1. Define paths
    centroids_path = os.path.join(index_path, "centroids.npy")
    codes_path = os.path.join(index_path, "merged_codes.npy")
    residuals_path = os.path.join(index_path, "merged_residuals.npy")
    bucket_weights_path = os.path.join(index_path, "bucket_weights.npy")
    metadata_path = os.path.join(index_path, "metadata.json")

    # 2. Load Metadata and Document Lengths
    # We need these to calculate the start/end offsets for random access.
    doc_lengths = []

    with open(metadata_path) as f:
        metadata = json.load(f)
        num_chunks = metadata.get("num_chunks", 0)

    for i in range(num_chunks):
        dl_path = os.path.join(index_path, f"doclens.{i}.json")
        if os.path.exists(dl_path):
            with open(dl_path) as f:
                doc_lengths.extend(json.load(f))

    # Pre-calculate offsets: offsets[i] is the starting token index for document i.
    offsets = np.concatenate(([0], np.cumsum(doc_lengths, dtype=np.int64)))
    total_docs = len(doc_lengths)

    # 3. Load Artifacts
    # Load small files into memory (centroids and weights are negligible in size).
    centroids = np.load(centroids_path)
    bucket_weights = np.load(bucket_weights_path)

    # Load large files using mmap_mode='r'.
    # This creates a reference to the file on disk WITHOUT loading it into RAM.
    # Data is only read when we explicitly slice it in the loop below.
    codes_mmap = np.load(codes_path, mmap_mode="r")
    residuals_mmap = np.load(residuals_path, mmap_mode="r")

    # 4. Determine Quantization Parameters
    embedding_dim = centroids.shape[1]
    # Calculate nbits: (96 dim * nbits) / 8 bits_per_byte = 48 bytes -> nbits = 4
    nbits = (residuals_mmap.shape[1] * 8) // embedding_dim
    powers_of_two = 2 ** np.arange(nbits)

    output_list = []

    # 5. Iterate and Reconstruct
    for doc_id in subset:
        # Safety check
        if doc_id < 0 or doc_id >= total_docs:
            continue

        start_idx = offsets[doc_id]
        end_idx = offsets[doc_id + 1]

        # Handle empty documents
        if start_idx == end_idx:
            output_list.append(torch.empty(0, embedding_dim))
            continue

        # --- Efficient Slicing (Disk I/O) ---
        # accessing the mmap array with a slice reads ONLY that specific chunk from disk.
        doc_codes = codes_mmap[start_idx:end_idx]
        doc_residuals = residuals_mmap[start_idx:end_idx]

        # --- Reconstruction ---

        # A. Retrieve Coarse Centroids
        coarse_vectors = centroids[doc_codes]

        # B. Decompress Residuals
        # Unpack bits: FastPlaid packs LSB..MSB, unpackbits produces MSB..LSB.
        bits = np.unpackbits(doc_residuals, axis=1)
        # Reshape to (num_tokens, embedding_dim, nbits)
        bits_reshaped = bits.reshape(-1, embedding_dim, nbits)

        # Compute integer indices (0-15 for 4-bit)
        bucket_indices = bits_reshaped.dot(powers_of_two).astype(np.int32)

        # Lookup weights
        residual_vectors = bucket_weights[bucket_indices]

        # C. Combine
        reconstructed = coarse_vectors + residual_vectors

        # Append as a float tensor
        output_list.append(torch.from_numpy(reconstructed).float())

    return output_list
