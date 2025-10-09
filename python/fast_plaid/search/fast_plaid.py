from __future__ import annotations

import glob
import math
import os
import random
from typing import Any, Self

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
    queries_embeddings: torch.Tensor,
    batch_size: int,
    n_full_scores: int,
    top_k: int,
    n_ivf_probe: int,
    index: str,
    torch_path: str,
    show_progress: bool,
    subset: list[list[int]] | None = None,
) -> list[list[tuple[int, float]]]:
    """Perform a search on a single specified device."""
    search_params_obj = fast_plaid_rust.SearchParameters(
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
        search_parameters=search_params_obj,
        show_progress=show_progress,
        subset=subset,
    )

    return [
        [
            (passage_id, score)
            for score, passage_id in zip(score.scores, score.passage_ids)
        ]
        for score in scores
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
    ) -> None:
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

        if self.multiple_gpus:
            return

        fast_plaid_rust.initialize_torch(
            torch_path=self.torch_path,
        )

    def create(  # noqa: PLR0913
        self,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        nbits: int = 4,
        n_samples_kmeans: int | None = None,
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
        seed:
            Optional seed for the random number generator used in index creation.
        use_triton_kmeans:
            Whether to use the Triton-based K-means implementation. If None, it will be
            set to True if the device is not "cpu".
        metadata:
            Optional list of dictionaries containing metadata for each document.

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

        print("Computing K-means centroids...")
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
        print("Finished computing K-means centroids.")

        fast_plaid_rust.create(
            index=self.index,
            torch_path=self.torch_path,
            device=self.devices[0],
            embedding_dim=dim,
            nbits=nbits,
            embeddings=documents_embeddings,
            centroids=centroids,
            seed=seed,
        )

        return self

    def update(
        self,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
        metadata: list[dict[str, Any]] | None = None,
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

    def search(  # noqa: PLR0913, C901
        self,
        queries_embeddings: torch.Tensor | list[torch.Tensor],
        top_k: int = 10,
        batch_size: int = 1 << 18,
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
            Internal batch size for the search.
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
        if isinstance(queries_embeddings, list):
            queries_embeddings = torch.nn.utils.rnn.pad_sequence(
                sequences=[
                    embedding[0] if embedding.dim() == 3 else embedding
                    for embedding in queries_embeddings
                ],
                batch_first=True,
                padding_value=0.0,
            )

        num_queries = queries_embeddings.shape[0]

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

        if not self.multiple_gpus and len(self.devices) > 1:
            split_size = (num_queries // len(self.devices)) + 1
            queries_embeddings_splits = torch.split(
                tensor=queries_embeddings,
                split_size_or_sections=split_size,
            )

            subset_splits: list[list[list[int]] | None] = [None] * len(
                queries_embeddings_splits
            )
            if subset is not None:
                subset_splits = [
                    subset[i * split_size : (i + 1) * split_size]  # type: ignore
                    for i in range(len(queries_embeddings_splits))
                ]

            tasks = [
                delayed(function=search_on_device)(
                    device=device,
                    queries_embeddings=dev_queries,
                    batch_size=batch_size,
                    n_full_scores=n_full_scores,
                    top_k=top_k,
                    n_ivf_probe=n_ivf_probe,
                    index=self.index,
                    torch_path=self.torch_path,
                    show_progress=step == 0 and show_progress,
                    subset=dev_subset,
                )
                for step, (device, dev_queries, dev_subset) in enumerate(
                    zip(self.devices, queries_embeddings_splits, subset_splits)
                )
            ]

            scores_per_device = Parallel(n_jobs=len(self.devices))(tasks)

            scores = []
            for device_scores in scores_per_device:
                scores.extend(device_scores)

            return scores

        if not self.multiple_gpus:
            return search_on_device(
                device=self.devices[0],
                queries_embeddings=queries_embeddings,
                batch_size=batch_size,
                n_full_scores=n_full_scores,
                top_k=top_k,
                n_ivf_probe=n_ivf_probe,
                index=self.index,
                torch_path=self.torch_path,
                show_progress=True and show_progress,
                subset=subset,  # type: ignore
            )

        split_size = (num_queries // len(self.devices)) + 1
        queries_embeddings_splits = torch.split(
            tensor=queries_embeddings,
            split_size_or_sections=split_size,
        )

        subset_splits = [None] * len(queries_embeddings_splits)
        if subset is not None:
            subset_splits = [
                subset[i * split_size : (i + 1) * split_size]  # type: ignore
                for i in range(len(queries_embeddings_splits))
            ]

        args_for_starmap = [
            (
                device,
                dev_queries,
                batch_size,
                n_full_scores,
                top_k,
                n_ivf_probe,
                self.index,
                self.torch_path,
                step == 0 and show_progress,
                dev_subset,
            )
            for step, (device, dev_queries, dev_subset) in enumerate(
                zip(self.devices, queries_embeddings_splits, subset_splits)
            )
        ]

        scores_devices = []

        context = mp.get_context()
        with context.Pool(processes=len(args_for_starmap)) as pool:
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

        return self

class FastPlaidIndex:
    """A Python wrapper for the Rust FastPlaidIndex class.
    
    This class provides a clean Python interface for creating, loading, updating,
    and searching FastPlaid indexes using the underlying Rust implementation.
    
    Args:
    ----
    rust_index:
        The underlying Rust FastPlaidIndex instance.
    torch_path:
        Path to the PyTorch shared library.
    device:
        The device used by the index.
    
    """

    def __init__(
        self,
        rust_index: fast_plaid_rust.FastPlaidIndex,
        torch_path: str,
        device: str,
    ) -> None:
        self._rust_index = rust_index
        self._torch_path = torch_path
        self._device = device

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        index_path: str,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
        embedding_dim: int,
        nbits: int = 4,
        device: str | None = None,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        n_samples_kmeans: int | None = None,
        seed: int | None = None,
        use_triton_kmeans: bool | None = None,
    ) -> "FastPlaidIndex":
        """Create a new FastPlaid index.

        Args:
        ----
        index_path:
            The directory path where the new index will be saved.
        documents_embeddings:
            A list of document embedding tensors to be indexed, or a single tensor.
        embedding_dim:
            The dimensionality of the embeddings.
        nbits:
            Number of bits to use for residual quantization (default is 4).
        device:
            The compute device to use (e.g., "cpu", "cuda:0"). If None, defaults to "cuda" if available.
        kmeans_niters:
            Number of iterations for the K-means algorithm.
        max_points_per_centroid:
            The maximum number of points per centroid for K-means.
        n_samples_kmeans:
            Number of samples to use for K-means. If None, calculated automatically.
        seed:
            Optional seed for the random number generator.
        use_triton_kmeans:
            Whether to use the Triton-based K-means implementation.

        Returns:
        -------
        FastPlaidIndex:
            A new FastPlaidIndex instance ready for searching.

        """
        # Set default device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert single tensor to list
        if isinstance(documents_embeddings, torch.Tensor):
            documents_embeddings = [
                documents_embeddings[i] for i in range(documents_embeddings.shape[0])
            ]

        # Ensure embeddings are properly formatted
        documents_embeddings = [
            embedding.squeeze(0) if embedding.dim() == 3 else embedding
            for embedding in documents_embeddings
        ]

        # Prepare index directory
        FastPlaid._prepare_index_directory(index_path=index_path)

        # Get torch path
        torch_path = _load_torch_path(device=device)

        # Compute K-means centroids
        print("Computing K-means centroids...")
        centroids = compute_kmeans(
            documents_embeddings=documents_embeddings,
            dim=embedding_dim,
            kmeans_niters=kmeans_niters,
            device=device,
            max_points_per_centroid=max_points_per_centroid,
            n_samples_kmeans=n_samples_kmeans,
            seed=seed or 42,
            use_triton_kmeans=use_triton_kmeans,
        )
        print("Finished computing K-means centroids.")

        # Create the Rust index
        rust_index = fast_plaid_rust.FastPlaidIndex.create(
            index_path=index_path,
            torch_path=torch_path,
            device=device,
            embedding_dim=embedding_dim,
            nbits=nbits,
            embeddings=documents_embeddings,
            centroids=centroids,
            seed=seed,
        )

        return cls(rust_index=rust_index, torch_path=torch_path, device=device)

    @classmethod
    def load(
        cls,
        index_path: str,
        device: str | None = None,
    ) -> "FastPlaidIndex":
        """Load an existing FastPlaid index from disk.

        Args:
        ----
        index_path:
            Path to the directory containing the pre-built index.
        device:
            The compute device to use (e.g., "cpu", "cuda:0"). If None, defaults to "cuda" if available.

        Returns:
        -------
        FastPlaidIndex:
            A loaded FastPlaidIndex instance ready for searching.

        """
        # Set default device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get torch path
        torch_path = _load_torch_path(device=device)

        # Load the Rust index
        rust_index = fast_plaid_rust.FastPlaidIndex.load(
            index_path=index_path,
            torch_path=torch_path,
            device=device,
        )

        return cls(rust_index=rust_index, torch_path=torch_path, device=device)

    def update(
        self,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
    ) -> Self:
        """Update the index with new document embeddings.

        This method updates the current index in-place without creating a new instance.
        The index is updated both in memory and on disk using the stored index path.

        Args:
        ----
        documents_embeddings:
            A list of new document embedding tensors to add to the index, or a single tensor.

        """
        # Convert single tensor to list
        if isinstance(documents_embeddings, torch.Tensor):
            documents_embeddings = [
                documents_embeddings[i] for i in range(documents_embeddings.shape[0])
            ]

        # Ensure embeddings are properly formatted
        documents_embeddings = [
            embedding.squeeze(0) if embedding.dim() == 3 else embedding
            for embedding in documents_embeddings
        ]

        # Update the Rust index using the instance method (modifies in-place)
        # The index_path is now stored as an attribute in the Rust struct
        self._rust_index.update(
            embeddings=documents_embeddings,
        )
        return self

    def delete(self, subset: list[int]) -> Self:
        """Delete documents from the index.

        This method removes specified documents from the index in-place without creating 
        a new instance. The index is updated both in memory and on disk using the stored 
        index path.

        Args:
        ----
        subset:
            A list of document IDs to be removed from the index.

        """
        # Delete from the Rust index using the instance method (modifies in-place)
        # The index_path is now stored as an attribute in the Rust struct
        self._rust_index.delete(
            subset=subset,
        )
        return self

    def search(  # noqa: PLR0913
        self,
        queries_embeddings: torch.Tensor | list[torch.Tensor],
        top_k: int = 10,
        batch_size: int = 1 << 18,
        n_full_scores: int = 4096,
        n_ivf_probe: int = 8,
        show_progress: bool = True,
        subset: list[list[int]] | list[int] | None = None,
    ) -> list[list[tuple[int, float]]]:
        """Search the index for the given query embeddings.

        Args:
        ----
        queries_embeddings:
            Embeddings of the queries to search for. Can be a tensor or list of tensors.
        top_k:
            Number of top results to return.
        batch_size:
            Internal batch size for the search.
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

        Returns:
        -------
        list[list[tuple[int, float]]]:
            A list of search results, where each inner list contains tuples
            of (document_id, score) for each query.

        """
        # Convert tensor to list of tensors if needed
        if isinstance(queries_embeddings, torch.Tensor):
            if queries_embeddings.dim() == 3:
                # Batch of queries: [num_queries, seq_len, dim]
                queries_list = [queries_embeddings[i] for i in range(queries_embeddings.shape[0])]
            elif queries_embeddings.dim() == 2:
                # Single query: [seq_len, dim]
                queries_list = [queries_embeddings]
            else:
                raise ValueError(f"Expected 2D or 3D tensor, got {queries_embeddings.dim()}D")
        else:
            # Already a list of tensors
            queries_list = queries_embeddings

        # Ensure proper formatting
        queries_list = [
            query.squeeze(0) if query.dim() == 3 else query
            for query in queries_list
        ]

        num_queries = len(queries_list)

        # Handle subset formatting
        if subset is not None:
            if isinstance(subset, int):
                subset = [subset] * num_queries
            elif isinstance(subset, list) and len(subset) > 0 and isinstance(subset[0], int):
                subset = [subset] * num_queries  # type: ignore

            if len(subset) != num_queries:
                raise ValueError(
                    f"The length of the subset ({len(subset)}) must match the number of queries ({num_queries})"
                )

        # Create search parameters
        search_params = fast_plaid_rust.SearchParameters(
            batch_size=batch_size,
            n_full_scores=n_full_scores,
            top_k=top_k,
            n_ivf_probe=n_ivf_probe,
        )

        # Perform the search using the Rust implementation
        results = self._rust_index.search(
            queries_embeddings=queries_list,
            search_parameters=search_params,
            show_progress=show_progress,
            subset=subset,
        )

        # Convert results to the expected format
        formatted_results = [
            [
                (passage_id, score)
                for passage_id, score in zip(result.passage_ids, result.scores)
            ]
            for result in results
        ]

        return formatted_results

    @property
    def device(self) -> str:
        """Get the device used by the index."""
        return self._device

    @property
    def torch_path(self) -> str:
        """Get the torch library path used by the index."""
        return self._torch_path

    @property
    def index_path(self) -> str:
        """Get the directory path of the index."""
        return self._rust_index.index_path

