import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from fast_plaid import fast_plaid_rust

from . import storage


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
    """Fold shard staging files into the merged file and return it memory-mapped.

    The merged file is the single durable copy of the payload. Shards recorded as
    merged in the manifest are skipped entirely (their bytes are already in place);
    staging files written by ``update``/``delete`` are appended or rewritten from
    their position onward; and once the manifest acknowledges them, staging files
    are unlinked. The merged header has a fixed length, so growing or shrinking the
    file is an in-place header patch plus truncate, never a full rewrite.

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
    manifest = {}
    if os.path.exists(merged_path):
        manifest = storage.load_manifest(index_path, name_suffix)

    merged_shape = None
    if os.path.exists(merged_path):
        probe = np.load(merged_path, mmap_mode="c")
        merged_shape = probe.shape
        del probe

    # Row offset of each chunk inside the current merged file, per the manifest.
    old_offsets = {}
    current = 0
    for i in range(num_chunks):
        entry = manifest.get(f"{i}.{name_suffix}.npy")
        if entry:
            old_offsets[i] = current
            current += entry["rows"]

    # Scan chunks: staging files on disk win; otherwise the manifest's merged rows.
    total_rows_scan = 0
    cols = 0
    valid_chunks = []
    chain_broken = False

    for i in range(num_chunks):
        filename = f"{i}.{name_suffix}.npy"
        path = os.path.join(index_path, filename)
        entry = manifest.get(filename)

        if os.path.exists(path):
            try:
                stat = os.stat(path)  # noqa: PTH116
                current_mtime = stat.st_mtime
                mmap_arr = np.load(path, mmap_mode="c")
                shape = mmap_arr.shape
                del mmap_arr
            except ValueError:
                continue

            if len(shape) == 0 or shape[0] == 0:
                continue
            rows = shape[0]
            if len(shape) > 1:
                cols = shape[1]

            is_clean = (
                entry
                and entry.get("mtime") == current_mtime
                and entry["rows"] == rows
            )
            if chain_broken or not is_clean:
                chain_broken = True
                needs_write = True
            else:
                needs_write = False

            valid_chunks.append(
                {
                    "source": "file",
                    "path": path,
                    "filename": filename,
                    "rows": rows,
                    "mtime": current_mtime,
                    "write": needs_write,
                    "offset": total_rows_scan,
                }
            )
            total_rows_scan += rows
        elif entry and entry.get("merged"):
            rows = entry["rows"]
            if rows == 0:
                valid_chunks.append(
                    {"source": "merged", "filename": filename, "rows": 0,
                     "write": False, "offset": total_rows_scan}
                )
                continue
            if chain_broken and old_offsets.get(i) != total_rows_scan:
                error = (
                    f"Chunk {filename} lives only in {merged_filename} but earlier "
                    f"chunks changed size; the index at {index_path} cannot be "
                    f"re-merged in place. Re-create the index from source embeddings."
                )
                raise RuntimeError(error)
            valid_chunks.append(
                {"source": "merged", "filename": filename, "rows": rows,
                 "write": False, "offset": total_rows_scan}
            )
            total_rows_scan += rows
            if merged_shape is not None and len(merged_shape) > 1:
                cols = merged_shape[1]
    gc.collect()

    # Manifest lost but the merged file survives: the doclens files define the
    # merged layout exactly, so rebuild the manifest instead of reporting an
    # empty index.
    if total_rows_scan == 0 and merged_shape is not None:
        doclens_rows = [
            storage.shard_rows(index_path, i) for i in range(num_chunks)
        ]
        data_rows = sum(doclens_rows)
        if 0 < data_rows <= merged_shape[0]:
            valid_chunks = []
            offset = 0
            for i, rows in enumerate(doclens_rows):
                valid_chunks.append(
                    {"source": "merged", "filename": f"{i}.{name_suffix}.npy",
                     "rows": rows, "write": False, "offset": offset}
                )
                offset += rows
            total_rows_scan = data_rows
            if len(merged_shape) > 1:
                cols = merged_shape[1]

    if total_rows_scan == 0:
        return torch.empty(0, device=device, dtype=dtype)

    final_rows = total_rows_scan + padding_needed
    final_shape = (final_rows, cols) if cols > 0 else (final_rows,)

    any_merged_source = any(
        c["source"] == "merged" and c["rows"] > 0 for c in valid_chunks
    )
    fresh = merged_shape is None or not storage.has_fixed_header(merged_path)
    if merged_shape is not None:
        old_cols = merged_shape[1] if len(merged_shape) > 1 else 0
        if old_cols != (max(0, cols)):
            fresh = True

    # Read-only load: nothing dirty, same final size. Skip every write so
    # loading an index never touches its files (concurrent readers, mtime
    # watchers, and backup tools all rely on that).
    unchanged = (
        not fresh
        and merged_shape[0] == final_rows
        and not any(c["source"] == "file" and c["write"] for c in valid_chunks)
    )
    if unchanged:
        new_manifest = {}
        for chunk in valid_chunks:
            entry = {"rows": chunk["rows"], "merged": True}
            entry["mtime"] = chunk["mtime"] if chunk["source"] == "file" else None
            new_manifest[chunk["filename"]] = entry
        if new_manifest != manifest:
            storage.save_manifest(index_path, name_suffix, new_manifest)
        storage.drop_shard_payloads(index_path, name_suffix, num_chunks)
        arr = np.load(merged_path, mmap_mode="c")
        return torch.from_numpy(arr).to(device=device, dtype=dtype)
    if fresh and any_merged_source:
        error = (
            f"{merged_filename} at {index_path} must be rewritten but some rows "
            f"exist only inside it. Re-create the index from source embeddings."
        )
        raise RuntimeError(error)

    output_mmap = storage.open_merged(
        merged_path, numpy_dtype, final_shape, fresh=fresh
    )

    new_manifest = {}
    for chunk in valid_chunks:
        n_elems = chunk["rows"]
        if chunk["source"] == "file" and (fresh or chunk["write"]):
            chunk_data = np.load(chunk["path"])
            output_mmap[chunk["offset"] : chunk["offset"] + n_elems] = chunk_data
            del chunk_data
        entry = {"rows": n_elems, "merged": True}
        entry["mtime"] = chunk["mtime"] if chunk["source"] == "file" else None
        new_manifest[chunk["filename"]] = entry

    output_mmap.flush()
    del output_mmap
    gc.collect()

    storage.save_manifest(index_path, name_suffix, new_manifest)
    storage.drop_shard_payloads(index_path, name_suffix, num_chunks)

    arr = np.load(merged_path, mmap_mode="c")
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _mmap_frozen_tensor(
    name_suffix: str,
    dtype: torch.dtype,
    device: str,
    index_path: str,
) -> torch.Tensor:
    """Load a pre-existing merged_*.npy directly via mmap, with no chunk scan.

    Used when the index is marked ``frozen``: the per-shard files have been
    dropped, so the merged file is the sole source of truth.
    """
    merged_path = os.path.join(index_path, f"merged_{name_suffix}.npy")
    if not os.path.exists(merged_path):
        error = (
            f"Frozen index is missing {merged_path}. "
            f"The merged file is required when per-shard files have been dropped."
        )
        raise FileNotFoundError(error)
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
    # A frozen flag with mutable=true is the single-copy layout, which still needs
    # the merge scan to fold staging shards; only an immutable index may skip it.
    frozen = metadata.get("frozen", False) and not metadata.get("mutable", False)
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
    }

    ivf_path = os.path.join(index_path, "ivf.npy")
    ivf_lengths_path = os.path.join(index_path, "ivf_lengths.npy")
    if os.path.exists(ivf_path) and os.path.exists(ivf_lengths_path):
        data["ivf"] = _load_small_tensor(
            index_path=index_path,
            name="ivf.npy",
            dtype=torch.int64,
            device=device,
        )
        data["ivf_lengths"] = _load_small_tensor(
            index_path=index_path,
            name="ivf_lengths.npy",
            dtype=torch.int32,
            device=device,
        )
    else:
        data["ivf"] = None
        data["ivf_lengths"] = None

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

    if frozen:
        data["doc_codes"] = _mmap_frozen_tensor(
            name_suffix="codes",
            dtype=torch.int64,
            device=device,
            index_path=index_path,
        )
        data["doc_residuals"] = _mmap_frozen_tensor(
            name_suffix="residuals",
            dtype=torch.uint8,
            device=device,
            index_path=index_path,
        )
    else:
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
        if not storage.keep_shards():
            storage.mark_single_copy(index_path)

    return data


def _resolve_index_gpu_memory(
    data: dict[str, Any],
    device: str,
    index_gpu_memory: str,
    index_memory_fraction: float,
) -> str:
    """Resolve 'auto' to the highest tier within `index_memory_fraction` of VRAM."""
    if index_gpu_memory != "auto":
        return index_gpu_memory
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "low"

    free, total = torch.cuda.mem_get_info(torch.device(device))
    codes_bytes = data["doc_codes"].numel() * 8  # stored as int64
    residuals_bytes = data["doc_residuals"].numel()  # uint8
    headroom = int((1.0 - index_memory_fraction) * total)

    if free - (codes_bytes + residuals_bytes) >= headroom:
        return "high"
    if free - codes_bytes >= headroom:
        return "medium"
    return "low"


def _construct_index_from_tensors(
    data: dict[str, Any],
    device: str,
    index_gpu_memory: str,
    index_memory_fraction: float,
) -> Any:
    """Build Rust index from CPU tensors.

    Args:
    ----
    data:
        Dictionary of tensors loaded on CPU.
    device:
        The target device for the index.
    index_gpu_memory:
        GPU placement tier for the large document tensors; 'auto' adapts per device.
    index_memory_fraction:
        Fraction of device memory 'auto' placement may fill.

    """
    index_gpu_memory = _resolve_index_gpu_memory(
        data, device, index_gpu_memory, index_memory_fraction
    )

    gpu_data: dict[str, Any] = {}
    for key, val in data.items():
        if val is None:
            gpu_data[key] = None
        elif isinstance(val, torch.Tensor):
            if key in ["doc_codes", "doc_residuals", "doc_lengths"]:
                # Tier placement happens on the Rust side; hand these over on CPU.
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
        index_gpu_memory=index_gpu_memory,
    )


def _reload_index(
    index_path: str,
    devices: list[str],
    indices: dict[str, Any],
    index_gpu_memory: str = "auto",
    index_memory_fraction: float = 0.7,
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
    index_gpu_memory:
        GPU placement tier for the large document tensors; 'auto' adapts per device.
    index_memory_fraction:
        Fraction of device memory 'auto' placement may fill.

    """
    # Work on a copy: callers refresh their own dict from the return value
    # (e.g. `indices_dict.clear(); indices_dict.update(new_indices)`), which
    # empties the result too if it aliases the input.
    indices = dict(indices)

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
                index_gpu_memory=index_gpu_memory,
                index_memory_fraction=index_memory_fraction,
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
