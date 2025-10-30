from __future__ import annotations

import glob
import math
import os
import random
from typing import Any

import torch
import torch.multiprocessing as mp
from fast_plaid import fast_plaid_rust
from fastkmeans import FastKMeans
from joblib import Parallel, delayed

from ..filtering import create, delete, update


class TorchWithCudaNotFoundError(Exception):
    """Exception raised when PyTorch with CUDA support is not found."""


def _load_torch_path(device: str) -> str:
    """Find the path to the shared library for PyTorch with CUDA."""
    search_paths = [
        os.path.join(os.path.dirname(torch.__file__), "lib", f"libtorch_{device}.so"),
        os.path.join(os.path.dirname(torch.__file__), "**", f"libtorch_{device}.so"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda.so"),
        os.path.join(os.path.dirname(torch.__file__), "**", "libtorch_cuda.dylib"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so"),
        os.path.join(os.path.dirname(torch.__file__), "**", "libtorch.so"),
        os.path.join(os.path.dirname(torch.__file__), "**", "libtorch.dylib"),
        os.path.join(os.path.dirname(torch.__file__), "lib", f"torch_{device}.dll"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "torch.dll"),
        os.path.join(os.path.dirname(torch.__file__), "lib", f"c10_{device}.dll"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "c10.dll"),
        os.path.join(os.path.dirname(torch.__file__), "**", f"torch_{device}.dll"),
        os.path.join(os.path.dirname(torch.__file__), "**", "torch.dll"),
    ]

    for path_pattern in search_paths:
        found_libs = glob.glob(path_pattern, recursive=True)
        if found_libs:
            return found_libs[0]

    error = """
    Could not find torch binary.
    Please ensure PyTorch is installed.
    """
    raise TorchWithCudaNotFoundError(error) from IndexError


def compute_kmeans(  # noqa: PLR0913
    documents_embeddings: list[torch.Tensor],
    dim: int,
    device: str,
    kmeans_niters: int,
    max_points_per_centroid: int,
    seed: int,
    n_samples_kmeans: int | None = None,
    use_triton_kmeans: bool | None = None,
) -> torch.Tensor:
    """Compute K-means centroids for document embeddings."""
    num_passages = len(documents_embeddings)

    if n_samples_kmeans is None:
        n_samples_kmeans = min(
            1 + int(16 * math.sqrt(120 * num_passages)),
            num_passages,
        )

    n_samples_kmeans = min(num_passages, n_samples_kmeans)

    sampled_pids = random.sample(
        population=range(n_samples_kmeans),
        k=n_samples_kmeans,
    )

    samples: list[torch.Tensor] = [
        documents_embeddings[pid] for pid in set(sampled_pids)
    ]

    total_tokens = sum([sample.shape[0] for sample in samples])
    num_partitions = (total_tokens / len(samples)) * len(documents_embeddings)
    num_partitions = int(2 ** math.floor(math.log2(16 * math.sqrt(num_partitions))))

    tensors = torch.cat(tensors=samples)
    if tensors.is_cuda:
        tensors = tensors.to(device="cpu", dtype=torch.float16)

    kmeans = FastKMeans(
        d=dim,
        k=min(num_partitions, total_tokens),
        niter=kmeans_niters,
        gpu=device != "cpu",
        verbose=False,
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
        use_triton=use_triton_kmeans,
    )

    kmeans.train(data=tensors.numpy())

    centroids = torch.from_numpy(
        kmeans.centroids,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    return torch.nn.functional.normalize(
        input=centroids,
        dim=-1,
    ).half()


def search_on_device(  # noqa: PLR0913
    device: str,
    queries_embeddings: list[torch.Tensor],
    batch_size: int,
    n_full_scores: int,
    top_k: int,
    n_ivf_probe: int,
    index: str,
    torch_path: str,
    show_progress: bool,
    preload_index: bool,
    subset: list[list[int]] | None = None,
) -> list[list[tuple[int, float]]]:
    """Perform a search on a single specified device."""
    search_parameters = fast_plaid_rust.SearchParameters(
        batch_size=batch_size,
        n_full_scores=n_full_scores,
        top_k=top_k,
        n_ivf_probe=n_ivf_probe,
    )

    scores = fast_plaid_rust.load_and_search(
        index=index,
        torch_path=torch_path,
        device=device,
        queries_embeddings=queries_embeddings,
        search_parameters=search_parameters,
        show_progress=show_progress,
        subset=subset,
        preload_index=preload_index,
    )

    return [
        [
            (passage_id, score)
            for score, passage_id in zip(score.scores, score.passage_ids)
        ]
        for score in scores
    ]


def cleanup_embeddings(embeddings: list[torch.Tensor] | torch.Tensor) -> list[torch.Tensor]:
    if isinstance(embeddings, torch.Tensor):
        embeddings = [
            embeddings[i] for i in range(embeddings.shape[0])
        ]
    return [
        embedding.squeeze(0) if embedding.dim() == 3 else embedding
        for embedding in embeddings
    ]


class FastPlaid:
    """A class for creating and searching a FastPlaid index.

    Args:
    ----
    index:
        Path to the directory where the index is stored or will be stored.
    device:
        The device(s) to use for computation (e.g., "cuda", ["cuda:0", "cuda:1"]).
        If None, defaults to ["cuda"].

    """

    def __init__(
        self,
        index: str,
        device: str | list[str] | None = None,
        preload_index: bool = True,
    ) -> None:
        """Initialize the FastPlaid instance."""
        self.multiple_gpus = False
        if (
            isinstance(device, list)
            and len(device) > 1
            and torch.cuda.device_count() > 1
        ):
            self.multiple_gpus = True
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method(method="spawn", force=True)

        if device is None and torch.cuda.is_available():
            self.devices = ["cuda"]
        elif not torch.cuda.is_available():
            cpu_count = os.cpu_count()
            if cpu_count is None:
                error = """
                No CPU cores available. Please check your system configuration.
                >>> import os; print(os.cpu_count())
                Returns None.
                """
                raise RuntimeError(error)
            self.devices = ["cpu"] * cpu_count
        elif isinstance(device, str):
            self.devices = [device]
        elif isinstance(device, list):
            self.devices = device
        else:
            error = "Device must be a string, a list of strings, or None."
            raise ValueError(error)

        self.torch_path = _load_torch_path(device=self.devices[0])
        self.index = index

        self.preload_index = preload_index

        if self.preload_index:
            self._load_index(
                index_path=self.index,
                torch_path=self.torch_path,
                device=self.devices[0],
            )

        if self.multiple_gpus:
            return

        fast_plaid_rust.initialize_torch(
            torch_path=self.torch_path,
        )

    @staticmethod
    def _load_index(index_path: str, torch_path: str, device: str) -> None:
        """Triggers the loading of the index.

        If the index is already in the cache, this function does nothing.
        This can be used to "warm up" the index before the first search.

        Args:
        ----
        index_path:
            Path to the index directory.
        torch_path:
            Path to the libtorch shared library.
        device:
            The device string (e.g., "cpu", "cuda:0").

        """
        if not os.path.exists(os.path.join(index_path, "metadata.json")):
            return

        # The Rust function handles both torch initialization and loading
        fast_plaid_rust.preload_index(
            index=index_path,
            torch_path=torch_path,
            device=device,
        )

    def create(  # noqa: PLR0913
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
        """Create and saves the FastPlaid index.

        Args:
        ----
        documents_embeddings:
            A list of document embedding tensors to be indexed.
        kmeans_niters:
            Number of iterations for the K-means algorithm.
        max_points_per_centroid:
            The maximum number of points per centroid for K-means.
        nbits:
            Number of bits to use for quantization (default is 4).
        n_samples_kmeans:
            Number of samples to use for K-means. If None, it will be calculated based
            on the number of documents.
        batch_size:
            Batch size for processing embeddings during index creation.
        seed:
            Optional seed for the random number generator used in index creation.
        use_triton_kmeans:
            Whether to use the Triton-based K-means implementation. If None, it will be
            set to True if the device is not "cpu".
        metadata:
            Optional list of dictionaries containing metadata for each document.

        """
        documents_embeddings = cleanup_embeddings(documents_embeddings)
        num_docs = len(documents_embeddings)

        self._prepare_index_directory(index_path=self.index)

        if metadata is not None:
            if len(metadata) != num_docs:
                error = f"""
                The length of metadata ({len(metadata)}) must match the number of
                documents_embeddings ({num_docs}).
                """
                raise ValueError(error)
            create(index=self.index, metadata=metadata)

        dim = documents_embeddings[0].shape[-1]

        print("Computing centroids of embeddings.")
        centroids = compute_kmeans(
            documents_embeddings=documents_embeddings,
            dim=dim,
            kmeans_niters=kmeans_niters,
            device=self.devices[0],
            max_points_per_centroid=max_points_per_centroid,
            n_samples_kmeans=n_samples_kmeans,
            seed=seed,
            use_triton_kmeans=use_triton_kmeans,
        )

        print("Creating FastPlaid index.")
        fast_plaid_rust.create(
            index=self.index,
            torch_path=self.torch_path,
            device=self.devices[0],
            embedding_dim=dim,
            nbits=nbits,
            embeddings=documents_embeddings,
            centroids=centroids,
            batch_size=batch_size,
            seed=seed,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.preload_index:
            self._load_index(
                index_path=self.index,
                torch_path=self.torch_path,
                device=self.devices[0],
            )

        return self

    def update(
        self,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
        metadata: list[dict[str, Any]] | None = None,
        batch_size: int = 25_000,
    ) -> "FastPlaid":
        """Update an existing FastPlaid index with new documents.

        This method adds new embeddings to the index without re-training the quantizer,
        making it much faster than re-creating the index from scratch.

        Args:
        ----
        documents_embeddings:
            A list of new document embedding tensors to add to the index.
        metadata:
            Optional list of dictionaries containing metadata for each new document.
        batch_size:
            Batch size for processing embeddings during the update.

        """
        if isinstance(documents_embeddings, torch.Tensor):
            documents_embeddings = [
                documents_embeddings[i] for i in range(documents_embeddings.shape[0])
            ]

        documents_embeddings = [
            embedding.squeeze(0) if embedding.dim() == 3 else embedding
            for embedding in documents_embeddings
        ]
        num_docs = len(documents_embeddings)

        if not os.path.exists(self.index) or not os.path.exists(
            os.path.join(self.index, "metadata.json")
        ):
            error = f"""
            Index directory '{self.index}' does not exist or is invalid.
            Please create an index first using the .create() method.
            """
            raise FileNotFoundError(error)

        if os.path.exists(os.path.join(self.index, "metadata.db")):
            if metadata is None:
                metadata = [{} for _ in range(num_docs)]

            if len(metadata) != num_docs:
                error = f"""
                The length of metadata ({len(metadata)}) must match the number of
                documents_embeddings ({num_docs}).
                """
                raise ValueError(error)
            update(index=self.index, metadata=metadata)

        fast_plaid_rust.update(
            index=self.index,
            torch_path=self.torch_path,
            device=self.devices[0],
            embeddings=documents_embeddings,
            batch_size=batch_size,
        )

        if self.preload_index:
            self._load_index(
                index_path=self.index,
                torch_path=self.torch_path,
                device=self.devices[0],
            )

        return self

    @staticmethod
    def _prepare_index_directory(index_path: str) -> None:
        """Prepare the index directory by cleaning or creating it."""
        if os.path.exists(index_path) and os.path.isdir(index_path):
            for json_file in glob.glob(os.path.join(index_path, "*.json")):
                try:
                    os.remove(json_file)
                except OSError:
                    pass

            for npy_file in glob.glob(os.path.join(index_path, "*.npy")):
                try:
                    os.remove(npy_file)
                except OSError:
                    pass
        elif not os.path.exists(index_path):
            try:
                os.makedirs(index_path)
            except OSError as e:
                raise e

    def search(  # noqa: PLR0913, C901, PLR0912, PLR0915
        self,
        queries_embeddings: torch.Tensor | list[torch.Tensor],
        top_k: int = 10,
        batch_size: int = 25000,
        n_full_scores: int = 4096,
        n_ivf_probe: int = 8,
        show_progress: bool = True,
        subset: list[list[int]] | list[int] | None = None,
    ) -> list[list[tuple[int, float]]]:
        """Search the index for the given query embeddings.

        Args:
        ----
        queries_embeddings:
            Embeddings of the queries to search for.
        top_k:
            Number of top results to return.
        batch_size:
            Internal batch size for the search, and also the size of query
            chunks for parallel processing.
        n_full_scores:
            Number of full scores to compute for re-ranking.
        n_ivf_probe:
            Number of inverted file probes to use.
        show_progress:
            Whether to show progress during the search.
        subset:
            An optional list of lists of integers. If provided, the search
            for each query will be restricted to the document IDs in the
            corresponding inner list.

        """
        queries_embeddings = cleanup_embeddings(queries_embeddings)
        num_queries = len(queries_embeddings)

        if subset is not None:
            if isinstance(subset, int):
                subset = [subset] * num_queries

            if isinstance(subset, list) and len(subset) == 0:
                subset = None

            if isinstance(subset, list) and isinstance(subset[0], int):
                subset = [subset] * num_queries  # type: ignore

        if subset is not None and len(subset) != num_queries:
            error = """
            The length of the subset must match the number of queries. You can
            provide either a single subset for all queries or a list of subsets
            with the same length as the number of queries.
            """
            raise ValueError(error)

        # Check for small query count on CPU to avoid multiprocessing overhead
        is_cpu = self.devices[0] == "cpu"
        small_query_count = num_queries <= 10

        if small_query_count and is_cpu:
            # Use single-device path directly, bypassing splitting and parallel logic
            return search_on_device(
                device=self.devices[0],
                queries_embeddings=queries_embeddings,
                batch_size=batch_size,
                n_full_scores=n_full_scores,
                top_k=top_k,
                n_ivf_probe=n_ivf_probe,
                index=self.index,
                torch_path=self.torch_path,
                show_progress=show_progress,
                preload_index=self.preload_index,
                subset=subset,  # type: ignore
            )

        # Check for parallel CPU processing (>= 10 queries, multiple CPUs)
        if is_cpu and len(self.devices) > 1:
            # Split queries based on the number of available CPUs
            num_cpus = len(self.devices)

            # Use torch.chunk to split the tensor into num_cpus
            queries_embeddings_splits = [
                queries_embeddings[i:i + num_cpus] for i in range(0, num_queries, num_cpus)
            ]

            # Filter out empty chunks that torch.chunk might create
            # if num_queries < num_cpus
            non_empty_splits = [
                split for split in queries_embeddings_splits if len(split) > 0
            ]
            num_splits = len(non_empty_splits)

            subset_splits: list[list[list[int]] | None] = (
                [None] * num_splits if subset is None else []
            )
            if subset is not None:
                current_idx = 0
                for split in non_empty_splits:
                    size = len(split)
                    subset_splits.append(subset[current_idx : current_idx + size])  # type: ignore
                    current_idx += size

            # Parallel CPU processing
            tasks = []
            for i in range(num_splits):
                device = self.devices[i]  # Use i-th CPU for i-th split
                dev_queries = non_empty_splits[i]  # Use the non-empty split
                dev_subset = subset_splits[i]

                tasks.append(
                    delayed(function=search_on_device)(
                        device=device,
                        queries_embeddings=dev_queries,
                        batch_size=batch_size,  # Keep original batch_size for inside
                        n_full_scores=n_full_scores,
                        top_k=top_k,
                        n_ivf_probe=n_ivf_probe,
                        index=self.index,
                        torch_path=self.torch_path,
                        show_progress=i == 0 and show_progress,
                        preload_index=self.preload_index,
                        subset=dev_subset,
                    )
                )

            # Use num_cpus for n_jobs to utilize all cores
            scores_per_device = Parallel(n_jobs=num_cpus)(tasks)

            scores = []
            for device_scores in scores_per_device:
                scores.extend(device_scores)

            return scores

        if not self.multiple_gpus:
            # Single device (1 GPU) processing
            return search_on_device(
                device=self.devices[0],
                queries_embeddings=queries_embeddings,
                batch_size=batch_size,
                n_full_scores=n_full_scores,
                top_k=top_k,
                n_ivf_probe=n_ivf_probe,
                index=self.index,
                torch_path=self.torch_path,
                show_progress=show_progress,
                preload_index=self.preload_index,
                subset=subset,  # type: ignore
            )

        queries_embeddings_splits = [
            queries_embeddings[i:i + len(self.devices)] for i in range(0, num_queries, len(self.devices))
        ]

        num_splits = len(queries_embeddings_splits)
        if subset is not None:
            current_idx = 0
            for split in queries_embeddings_splits:
                size = len(split)
                subset_splits.append(subset[current_idx : current_idx + size])  # type: ignore
                current_idx += size
        else:
            # Initialize subset_splits with Nones if subset is None
            subset_splits = [None] * num_splits

        # Parallel GPU processing
        args_for_starmap = []
        for i in range(num_splits):
            device = self.devices[i % len(self.devices)]  # Cycle through GPUs
            dev_queries = queries_embeddings_splits[i]
            dev_subset = subset_splits[i]  # This is now safe to access

            args_for_starmap.append(
                (
                    device,
                    dev_queries,
                    batch_size,
                    n_full_scores,
                    top_k,
                    n_ivf_probe,
                    self.index,
                    self.torch_path,
                    i == 0 and show_progress,
                    self.preload_index,
                    dev_subset,
                )
            )

        scores_devices = []

        context = mp.get_context()
        with context.Pool(processes=len(self.devices)) as pool:
            scores_devices = pool.starmap(
                func=search_on_device,
                iterable=args_for_starmap,
            )

        scores = []
        for scores_device in scores_devices:
            scores.extend(scores_device)

        return scores

    def delete(self, subset: list[int]) -> "FastPlaid":
        """Delete embeddings from an existing FastPlaid index.

        If a metadata database exists, the corresponding entries will also
        be deleted.

        Args:
        ----
        subset:
            List of embeddings to delete from the index with respect to
            the insertion order.

        """
        fast_plaid_rust.delete(
            index=self.index,
            torch_path=self.torch_path,
            device=self.devices[0],
            subset=subset,
        )

        metadata_db_path = os.path.join(self.index, "metadata.db")
        if os.path.exists(metadata_db_path):
            delete(index=self.index, subset=subset)

        if self.preload_index:
            self._load_index(
                index_path=self.index,
                torch_path=self.torch_path,
                device=self.devices[0],
            )

        return self
