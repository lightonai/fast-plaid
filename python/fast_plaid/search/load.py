import gc
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
    """Load a tensor from a .npy file.

    Args:
    ----
    index_path:
        The path to the index directory.
    name:
        The filename of the tensor to load.
    dtype:
        The dtype to convert the tensor to.
    device:
        The device to load the tensor to.

    """
    path = os.path.join(index_path, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing index file: {path}")
    return torch.from_numpy(np.load(path)).to(device=device, dtype=dtype)


def _get_merged_mmap(  # noqa: PLR0912
    name_suffix: str,
    dtype: torch.dtype,
    numpy_dtype: np.dtype,
    padding_needed: int,
    device: str,
    index_path: str,
    num_chunks: int,
) -> torch.Tensor:
    """Merge chunked .npy files into a single memory-mapped file.

    Uses incremental persistence with a manifest to track chunk modification times.
    Skips unchanged chunks and resizes files in-place when possible.

    Args:
    ----
    name_suffix:
        The suffix for the chunk files (e.g., "codes" or "residuals").
    dtype:
        The torch dtype for the output tensor.
    numpy_dtype:
        The numpy dtype for the memory-mapped file.
    padding_needed:
        Number of padding rows to add at the end.
    device:
        The device to load the final tensor to.
    index_path:
        The path to the index directory.
    num_chunks:
        The number of chunks to merge.

    """
    merged_filename = f"merged_{name_suffix}.npy"
    merged_path = os.path.join(index_path, merged_filename)
    manifest_path = os.path.join(index_path, f"merged_{name_suffix}.manifest.json")

    # Load previous manifest for change detection
    manifest = {}
    if os.path.exists(merged_path) and os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = {}

    # Scan chunks and detect changes
    total_rows_scan = 0
    cols = 0
    valid_chunks = []
    chain_broken = False

    for i in range(num_chunks):
        filename = f"{i}.{name_suffix}.npy"
        path = os.path.join(index_path, filename)

        if os.path.exists(path):
            try:
                stat = os.stat(path)  # noqa: PTH116
                current_mtime = stat.st_mtime
                # Use mmap_mode="c" but ensure the reference is released
                mmap_arr = np.load(path, mmap_mode="c")
                shape = mmap_arr.shape
                del mmap_arr

                if len(shape) > 0 and shape[0] > 0:
                    rows = shape[0]
                    total_rows_scan += rows
                    if len(shape) > 1:
                        cols = shape[1]

                    prev_entry = manifest.get(filename)
                    is_clean = (
                        prev_entry
                        and prev_entry["mtime"] == current_mtime
                        and prev_entry["rows"] == rows
                    )

                    if not chain_broken and is_clean:
                        needs_write = False
                    else:
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
                pass
    # Ensure all mmap handles from scanning are released before proceeding
    gc.collect()

    if total_rows_scan == 0:
        return torch.empty(0, device=device, dtype=dtype)

    final_rows = total_rows_scan + padding_needed
    final_shape = (final_rows, cols) if cols > 0 else (final_rows,)

    # Attempt in-place resize to avoid full rewrite
    file_mode = "w+"
    if os.path.exists(merged_path):
        try:
            with open(merged_path, "rb+") as f:
                version = np_fmt.read_magic(f)

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

                    if len(new_header_bytes) == header_len:
                        f.seek(0)
                        f.write(new_header_bytes)

                        row_size = np.dtype(numpy_dtype).itemsize * (
                            cols if cols > 0 else 1
                        )
                        total_bytes = header_len + (final_rows * row_size)
                        f.truncate(total_bytes)
                        file_mode = "r+"
        except (ValueError, OSError, EOFError):
            pass

    # Write chunks to memory-mapped output
    output_mmap = np.lib.format.open_memmap(
        merged_path, mode=file_mode, dtype=numpy_dtype, shape=final_shape
    )

    current_idx = 0
    new_manifest = {}
    force_write_all = file_mode == "w+"

    for chunk in valid_chunks:
        n_elems = chunk["rows"]

        if force_write_all or chunk["write"]:
            chunk_data = np.load(chunk["path"])
            output_mmap[current_idx : current_idx + n_elems] = chunk_data
            del chunk_data

        new_manifest[chunk["filename"]] = {"rows": n_elems, "mtime": chunk["mtime"]}
        current_idx += n_elems

    output_mmap.flush()
    del output_mmap
    gc.collect()

    # Save manifest and return tensor
    try:
        with open(manifest_path, "w") as f:
            json.dump(new_manifest, f)
    except OSError:
        pass

    arr = np.load(merged_path, mmap_mode="c")
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _load_index_tensors_cpu(index_path: str) -> dict[str, Any] | None:
    """Load index data into CPU tensors.

    Uses memory mapping for large tensors to avoid loading everything into RAM.

    Args:
    ----
    index_path:
        The path to the index directory.

    """
    metadata_path = os.path.join(index_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    num_chunks = metadata["num_chunks"]
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


def _construct_index_from_tensors(
    data: dict[str, Any], device: str, low_memory: bool
) -> Any:
    """Build Rust index from CPU tensors.

    Args:
    ----
    data:
        Dictionary of tensors loaded on CPU.
    device:
        The target device for the index.
    low_memory:
        If True, keeps large document tensors on CPU to save VRAM.

    """
    gpu_data = {}
    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            if low_memory and key in ["doc_codes", "doc_residuals", "doc_lengths"]:
                gpu_data[key] = val
            else:
                gpu_data[key] = val.to(device, non_blocking=True)
        else:
            gpu_data[key] = val

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
        low_memory=low_memory,
    )


def _reload_index(
    index_path: str,
    devices: list[str],
    indices: dict[str, Any],
    low_memory: bool = False,
) -> dict[str, Any]:
    """Load or reload the index for all configured devices.

    Args:
    ----
    index_path:
        The path to the index directory.
    devices:
        List of devices to load the index on.
    indices:
        Dictionary mapping devices to index objects.
    low_memory:
        If True, keeps large document tensors on CPU.

    """
    if not os.path.exists(os.path.join(index_path, "metadata.json")):
        for device in devices:
            indices[device] = None
        return indices

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

    def _provision_gpu(device: str) -> tuple[str, Any]:
        try:
            idx = _construct_index_from_tensors(
                data=cpu_tensors,  # noqa: F821
                device=device,
                low_memory=low_memory,
            )
            return device, idx  # noqa: TRY300
        except Exception as e:
            print(f"Warning: Failed to load index on {device}: {e}")
        return device, None

    if len(devices) == 1:
        dev, idx = _provision_gpu(devices[0])
        indices[dev] = idx
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            results = executor.map(_provision_gpu, devices)
            indices = dict(results)

    del cpu_tensors
    return indices


def save_list_tensors_on_disk(path: str, tensors: list[torch.Tensor]) -> None:
    """Save a list of tensors to a .npy file.

    Args:
    ----
    path:
        The file path to save to.
    tensors:
        List of tensors to save.

    """
    data_array = np.empty(len(tensors), dtype=object)
    for i, t in enumerate(tensors):
        data_array[i] = t.cpu().numpy()
    np.save(path, data_array, allow_pickle=True)
