import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import numpy.lib.format as np_fmt
import torch
from fast_plaid import fast_plaid_rust


def _load_small_tensor(index_path: str, name: str, dtype, device: str) -> torch.Tensor:
    """Load a small tensor from a .npy file."""
    path = os.path.join(index_path, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing index file: {path}")
    return torch.from_numpy(np.load(path)).to(device=device, dtype=dtype)


def _get_merged_mmap(  # noqa: C901,PLR0913,PLR0912,PLR0915
    name_suffix: str,
    dtype: torch.dtype,
    numpy_dtype: np.dtype,
    padding_needed: int,
    device: str,
    index_path: str,
    num_chunks: int,
) -> torch.Tensor:
    """Merge multiple chunked .npy files into a single monolithic memory-mapped file.

    Incremental Persistence: Uses a sidecar manifest (.json) to track chunk
        modification times.
    Zero-Copy Skippage: If a chunk hasn't changed and its offset hasn't shifted,
        we skip I/O entirely.
    In-Place Resizing: Manually manipulates the NPY header to grow the file
        without a full rewrite.
    """
    merged_filename = f"merged_{name_suffix}.npy"
    merged_path = os.path.join(index_path, merged_filename)
    manifest_path = os.path.join(index_path, f"merged_{name_suffix}.manifest.json")

    # -------------------------------------------------------------------------
    # Phase 1: State Restoration
    # -------------------------------------------------------------------------
    # We attempt to load the previous manifest to determine which chunks are dirty.
    manifest = {}
    if os.path.exists(merged_path) and os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = {}

    # -------------------------------------------------------------------------
    # Phase 2: Change Detection & Work Plan
    # -------------------------------------------------------------------------
    total_rows_scan = 0
    cols = 0
    valid_chunks = []  # Payload: (path, size, mtime, needs_write)

    # CRITICAL: Contiguity Check.
    # If chunk N changes size, all subsequent chunks (N+1, N+2...) shift positions
    # in the monolithic array. Therefore, once the 'chain' is broken, we must
    # rewrite everything following it, regardless of mtime.
    chain_broken = False

    for i in range(num_chunks):
        filename = f"{i}.{name_suffix}.npy"
        path = os.path.join(index_path, filename)

        if os.path.exists(path):
            try:
                # Lightweight stat check before opening file
                stat = os.stat(path)  # noqa: PTH116
                current_mtime = stat.st_mtime

                # Peek at header for shape using mmap_mode="c" (copy-on-write)
                # to avoid reading data into RAM.
                shape = np.load(path, mmap_mode="c").shape

                if len(shape) > 0 and shape[0] > 0:
                    rows = shape[0]
                    total_rows_scan += rows
                    if len(shape) > 1:
                        cols = shape[1]

                    # Verify against previous state
                    prev_entry = manifest.get(filename)

                    needs_write = True
                    is_clean = (
                        prev_entry
                        and prev_entry["mtime"] == current_mtime
                        and prev_entry["rows"] == rows
                    )

                    if not chain_broken and is_clean:
                        # Happy path: File is untouched and offset is stable.
                        needs_write = False
                    else:
                        # Dirty path: Either content changed or offset shifted.
                        chain_broken = True
                        needs_write = True

                    valid_chunks.append(
                        {
                            "path": path,
                            "filename": filename,
                            "rows": rows,
                            "mtime": current_mtime,
                            "write": needs_write,
                        }
                    )
            except ValueError:
                # Handle corrupted partial chunks gracefully
                pass

    if total_rows_scan == 0:
        return torch.empty(0, device=device, dtype=dtype)

    final_rows = total_rows_scan + padding_needed
    final_shape = (final_rows, cols) if cols > 0 else (final_rows,)

    # -------------------------------------------------------------------------
    # Phase 3: Disk Allocation (Smart Resize)
    # -------------------------------------------------------------------------
    file_mode = "w+"  # Default: Destructive overwrite

    # Attempt to resize in-place using low-level header manipulation.
    # This prevents rewriting terabytes of data just because we appended 1KB.
    if os.path.exists(merged_path):
        try:
            with open(merged_path, "rb+") as f:
                version = np_fmt.read_magic(f)

                # Parse existing header to check compatibility
                if version == (1, 0):
                    shape, fortran_order, _ = np_fmt.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, fortran_order, _ = np_fmt.read_array_header_2_0(f)
                else:
                    error = "Unsupported .npy version"
                    raise ValueError(error)  # noqa: TRY301

                header_len = f.tell()
                current_cols = shape[1] if len(shape) > 1 else 0
                cols_match = (current_cols == cols) if cols > 0 else (len(shape) == 1)

                # If columns match, we can simply update the
                # header shape and truncate/extend
                if cols_match:
                    buffer = io.BytesIO()
                    header_opts = {
                        "descr": np_fmt.dtype_to_descr(np.dtype(numpy_dtype)),
                        "fortran_order": fortran_order,
                        "shape": final_shape,
                    }

                    if version == (1, 0):
                        np_fmt.write_array_header_1_0(buffer, header_opts)
                    else:
                        np_fmt.write_array_header_2_0(buffer, header_opts)

                    new_header_bytes = buffer.getvalue()

                    # Safety: Ensure new header fits in the old padding space
                    if len(new_header_bytes) == header_len:
                        f.seek(0)
                        f.write(new_header_bytes)

                        # Calculate exact byte size including data payload
                        row_size = np.dtype(numpy_dtype).itemsize * (
                            cols if cols > 0 else 1
                        )
                        total_bytes = header_len + (final_rows * row_size)

                        # OS level resize
                        f.truncate(total_bytes)
                        file_mode = "r+"  # Switch to update mode
        except (ValueError, OSError, EOFError):
            # Fallback to full rewrite if corruption or version mismatch occurs
            pass

    # -------------------------------------------------------------------------
    # Phase 4: Stream Execution
    # -------------------------------------------------------------------------
    # Open the target file as a memory-mapped array.
    # This allows us to assign slice ranges (arr[a:b] = ...) which the OS
    # translates to direct disk writes, bypassing Python memory overhead.
    output_mmap = np.lib.format.open_memmap(
        merged_path, mode=file_mode, dtype=numpy_dtype, shape=final_shape
    )

    current_idx = 0
    new_manifest = {}

    # If we are in 'w+' mode, previous data is gone; we must write everything.
    force_write_all = file_mode == "w+"

    for chunk in valid_chunks:
        n_elems = chunk["rows"]

        if force_write_all or chunk["write"]:
            # I/O Bound: Load chunk into RAM, dump to mmap
            chunk_data = np.load(chunk["path"])
            output_mmap[current_idx : current_idx + n_elems] = chunk_data
            # Explicit delete to hint GC in tight memory loops
            del chunk_data
        else:
            # CPU Bound: Skip.
            # This is the performance win. We touch nothing.
            pass

        # Update tracking for next run
        new_manifest[chunk["filename"]] = {"rows": n_elems, "mtime": chunk["mtime"]}
        current_idx += n_elems

    # Ensure OS flushes buffers to disk
    output_mmap.flush()
    del output_mmap

    # -------------------------------------------------------------------------
    # Phase 5: Finalize
    # -------------------------------------------------------------------------
    try:
        with open(manifest_path, "w") as f:
            json.dump(new_manifest, f)
    except OSError:
        pass

    # Return a read-only view for PyTorch consumption
    arr = np.load(merged_path, mmap_mode="c")
    t = torch.from_numpy(arr)
    return t.to(device=device, dtype=dtype)


def _load_index_tensors_cpu(index_path: str) -> dict[str, Any] | None:
    """Load index data into CPU tensors (Memory Mapped where applicable)."""
    metadata_path = os.path.join(index_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    num_chunks = metadata["num_chunks"]

    # Always load to CPU first to ensure single-threaded disk I/O safety
    device = "cpu"

    data = {
        "nbits": metadata["nbits"],
        "centroids": _load_small_tensor(
            index_path=index_path,
            name="centroids.npy",
            dtype=torch.float16,
            device=device,
        ),
        "avg_residual": _load_small_tensor(
            index_path=index_path,
            name="avg_residual.npy",
            dtype=torch.float16,
            device=device,
        ),
        "bucket_cutoffs": _load_small_tensor(
            index_path=index_path,
            name="bucket_cutoffs.npy",
            dtype=torch.float16,
            device=device,
        ),
        "bucket_weights": _load_small_tensor(
            index_path=index_path,
            name="bucket_weights.npy",
            dtype=torch.float16,
            device=device,
        ),
        "ivf": _load_small_tensor(
            index_path=index_path,
            name="ivf.npy",
            dtype=torch.int64,
            device=device,
        ),
        "ivf_lengths": _load_small_tensor(
            index_path=index_path,
            name="ivf_lengths.npy",
            dtype=torch.int32,
            device=device,
        ),
    }

    all_doc_lens = []
    for i in range(num_chunks):
        dl_path = os.path.join(index_path, f"doclens.{i}.json")
        if os.path.exists(dl_path):
            with open(dl_path) as f:
                chunk_lens = json.load(f)
                all_doc_lens.extend(chunk_lens)

    data["doc_lengths"] = torch.tensor(all_doc_lens, device=device, dtype=torch.int64)

    max_len = max(all_doc_lens) if all_doc_lens else 0
    last_len = all_doc_lens[-1] if all_doc_lens else 0
    padding_needed = max(0, max_len - last_len)

    data["doc_codes"] = _get_merged_mmap(
        name_suffix="codes",
        dtype=torch.int64,
        numpy_dtype=np.int64,
        padding_needed=padding_needed,
        device=device,
        index_path=index_path,
        num_chunks=num_chunks,
    )

    data["doc_residuals"] = _get_merged_mmap(
        name_suffix="residuals",
        dtype=torch.uint8,
        numpy_dtype=np.uint8,
        padding_needed=padding_needed,
        device=device,
        index_path=index_path,
        num_chunks=num_chunks,
    )

    return data


def _construct_index_from_tensors(data: dict[str, Any], device: str) -> Any:
    """Move CPU tensors to specific device and build Rust index."""
    # Transfer tensors to the target device (GPU or CPU)
    # non_blocking=True helps parallelize the transfer on GPUs
    gpu_data = {
        key: (
            val.to(device, non_blocking=True) if isinstance(val, torch.Tensor) else val
        )
        for key, val in data.items()
    }

    return fast_plaid_rust.construct_index(
        nbits=gpu_data["nbits"],
        centroids=gpu_data["centroids"],
        avg_residual=gpu_data["avg_residual"],
        bucket_cutoffs=gpu_data["bucket_cutoffs"],
        bucket_weights=gpu_data["bucket_weights"],
        ivf=gpu_data["ivf"],
        ivf_lengths=gpu_data["ivf_lengths"],
        doc_codes=gpu_data["doc_codes"],
        doc_residuals=gpu_data["doc_residuals"],
        doc_lengths=gpu_data["doc_lengths"],
        device=device,
    )


def _reload_index(
    index_path: str,
    devices: list[str],
    indices: dict[str, Any],
) -> dict[str, Any]:
    """Load or reload the index object for every configured device."""
    # Check existence first
    if not os.path.exists(os.path.join(index_path, "metadata.json")):
        for device in devices:
            indices[device] = None
        return indices

    # 1. Load raw data to CPU (RAM) sequentially.
    # This handles the safely check for disk merging/metadata.
    try:
        cpu_tensors = _load_index_tensors_cpu(index_path=index_path)
    except Exception as e:
        print(f"Critical Error loading index from disk: {e}")
        for device in devices:
            indices[device] = None
        return indices

    if cpu_tensors is None:
        for device in devices:
            indices[device] = None
        return indices

    # Helper: Provision GPU
    def _provision_gpu(device: str) -> tuple[str, Any]:
        try:
            # Constructs index by moving CPU tensors to target device
            idx = _construct_index_from_tensors(data=cpu_tensors, device=device)  # noqa: F821
            return device, idx  # noqa: TRY300
        except Exception as e:
            print(f"Warning: Failed to load index on {device}: {e}")

        return device, None

    # Scenario A: Single Device (CPU or 1 GPU)
    # Avoid ThreadPoolExecutor overhead for the simple case
    if len(devices) == 1:
        dev, idx = _provision_gpu(devices[0])
        indices[dev] = idx

    # Scenario B: Multiple Devices (Multi-GPU)
    # Use ThreadPool to saturate bandwidth by copying from RAM to all GPUs at once
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            results = executor.map(_provision_gpu, devices)
            indices = dict(results)

    # Explicitly clean up CPU reference (optional, but good for clarity)
    del cpu_tensors
    return indices
