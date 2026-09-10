"""Tests for the host scoring-workspace budget."""

import gc
import os
import shutil

import pytest
import torch
from fast_plaid import search
from fast_plaid.search.fast_plaid import (
    CPU_STAGE_WORKSPACE_BYTES,
    device_memory_budget,
)


@pytest.fixture
def test_index_path(tmp_path):
    """Create a temporary index path for testing."""
    index_path = str(tmp_path / "budget_index")
    os.makedirs(index_path, exist_ok=True)
    yield index_path
    gc.collect()
    if os.path.exists(index_path):
        shutil.rmtree(index_path)


class TestHostBudget:
    """The host gets a real budget, not a device-shaped default."""

    def test_cpu_budget_is_the_host_workspace(self):
        """A host search must not fall through to the 4 GiB device default.

        Returning 0 here is what reached Rust as 'no budget given', and Rust
        answers that with a device-sized fallback large enough that a CPU
        search never chunks. The value matters less than its being set.
        """
        assert device_memory_budget("cpu", 0.5) == CPU_STAGE_WORKSPACE_BYTES

    def test_the_vram_fraction_does_not_reach_the_host(self):
        """The VRAM fraction must not reach the host.

        `search_memory_fraction` is a fraction of free VRAM, so on the host it
        has nothing to be a fraction of.
        """
        assert device_memory_budget("cpu", 0.1) == device_memory_budget("cpu", 0.9)

    def test_workspace_holds_enough_padded_work_to_stay_fast(self):
        """The budget is a throughput target, so it has a floor to respect.

        Chunks below roughly 100k padded cells start costing throughput. The
        exact stage's widest cell is ~2 KiB (fp16 token scores plus the
        decompression intermediates), so the budget must clear that product or
        the planner will cut chunks too small to amortize.
        """
        widest_cell_bytes = 4 * 128 + 1792
        assert 100_000 * widest_cell_bytes <= CPU_STAGE_WORKSPACE_BYTES

    def test_results_are_unchanged_by_chunking(self, test_index_path):
        """Chunking must not change the answer.

        The planner is score-preserving, so a chunked search has to agree
        exactly with one scored in a single pass.
        """
        torch.manual_seed(0)
        documents = [
            torch.randn(120 + 7 * (i % 11), 128, device="cpu") for i in range(96)
        ]
        index = search.FastPlaid(index=test_index_path, device="cpu")
        index.create(documents_embeddings=documents, kmeans_niters=4, nbits=4)

        torch.manual_seed(1)
        queries = [torch.randn(4 + 3 * (i % 5), 128, device="cpu") for i in range(8)]

        planned = index.search(queries, top_k=10, n_processes=1)
        one_chunk = index.search(queries, top_k=10, batch_size=10_000, n_processes=1)

        assert planned == one_chunk
