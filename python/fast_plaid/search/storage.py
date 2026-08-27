"""Single-copy storage for codes and residuals.

The merged ``merged_codes.npy`` / ``merged_residuals.npy`` files are the only durable
copy of the index payload. Per-shard ``{i}.codes.npy`` / ``{i}.residuals.npy`` files
are transient staging: the Rust core writes them on ``update()`` / ``delete()``, the
next merge folds them into the merged file, and they are then unlinked. Before calling
into Rust (which reads shards), the affected shards are re-materialized from the
merged file. ``doclens.{i}.json`` / ``{i}.metadata.json`` remain on disk and define
the logical shard layout.

The manifest (``merged_*.manifest.json``) records, per shard, either the mtime of a
staging file that has not been folded in yet, or ``{"merged": true}`` meaning the
rows live only in the merged file. Every mutation keeps at least one durable source
for every row (staging files survive until the manifest acknowledging them is
written), so a crash at any point leaves a recoverable index.

Set ``FAST_PLAID_KEEP_SHARDS=1`` to keep shard payloads on disk (the pre-1.7
double-storage layout).
"""

import json
import os

import numpy as np

# Total bytes of the npy prologue (magic + header) for merged files. Generous enough
# that any row count up to 20 digits reformats to the same length, so the in-place
# resize in the merge never falls back to a full rewrite because the shape field
# crossed a digit boundary.
MERGED_HEADER_LEN = 192


def keep_shards() -> bool:
    """Whether shard payloads should be kept on disk after merging (opt-out flag)."""
    return os.environ.get("FAST_PLAID_KEEP_SHARDS", "").strip() == "1"


def _header_bytes(numpy_dtype: np.dtype, shape: tuple) -> bytes:
    """Build a fixed-length npy v1.0 prologue for a merged file."""
    descr = np.lib.format.dtype_to_descr(np.dtype(numpy_dtype))
    if len(shape) == 1:
        shape_str = f"({shape[0]},)"
    else:
        shape_str = "(" + ", ".join(str(s) for s in shape) + ")"
    body = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}"
    pad = MERGED_HEADER_LEN - 10 - len(body) - 1
    if pad < 0:
        error = f"Merged npy header too large for fixed length: {body!r}"
        raise ValueError(error)
    header = (body + " " * pad + "\n").encode("ascii")
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header


def write_merged_header(path: str, numpy_dtype: np.dtype, shape: tuple) -> None:
    """Write or patch the fixed-length header of a merged file in place.

    Creates the file if missing. Because the header length is constant, patching the
    shape is a single small write at offset 0 and the data offset never moves.
    """
    prologue = _header_bytes(numpy_dtype, shape)
    mode = "rb+" if os.path.exists(path) else "wb+"
    with open(path, mode) as f:
        f.write(prologue)


def has_fixed_header(path: str) -> bool:
    """Whether a merged file already uses the fixed-length header.

    Legacy merged files were written by numpy with 64-byte-aligned headers; those are
    upgraded by one full rewrite (all shard payloads are still on disk at that point).
    """
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic[:6] != b"\x93NUMPY":
            return False
        header_len = int.from_bytes(f.read(2), "little")
        return 10 + header_len == MERGED_HEADER_LEN


def open_merged(
    path: str, numpy_dtype: np.dtype, shape: tuple, fresh: bool
) -> np.memmap:
    """Open a merged file for writing at its final shape, resizing in place.

    ``fresh`` truncates and rewrites everything; otherwise existing data bytes are
    preserved and only the header and file length are adjusted.
    """
    row_bytes = np.dtype(numpy_dtype).itemsize
    for dim in shape[1:]:
        row_bytes *= dim
    total = MERGED_HEADER_LEN + shape[0] * row_bytes
    if fresh and os.path.exists(path):
        os.remove(path)
    write_merged_header(path, numpy_dtype, shape)
    with open(path, "rb+") as f:
        f.truncate(total)
    return np.memmap(
        path, dtype=numpy_dtype, mode="r+", offset=MERGED_HEADER_LEN, shape=shape
    )


def load_manifest(index_path: str, name_suffix: str) -> dict:
    """Load the merge manifest, or an empty dict when absent or unreadable."""
    manifest_path = os.path.join(index_path, f"merged_{name_suffix}.manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_manifest(index_path: str, name_suffix: str, manifest: dict) -> None:
    """Atomically replace the merge manifest."""
    manifest_path = os.path.join(index_path, f"merged_{name_suffix}.manifest.json")
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, manifest_path)  # noqa: PTH105


def shard_rows(index_path: str, chunk_idx: int) -> int:
    """Count the embedding rows of a logical shard from its doclens file."""
    doclens_path = os.path.join(index_path, f"doclens.{chunk_idx}.json")
    if not os.path.exists(doclens_path):
        return 0
    with open(doclens_path) as f:
        return int(sum(json.load(f)))


def merged_offsets(
    index_path: str, name_suffix: str, num_chunks: int
) -> list[int] | None:
    """Row offset of each shard inside the merged file, from the manifest.

    Returns ``None`` when any shard in range is not recorded as merged — offsets are
    only meaningful for rows that actually live in the merged file.
    """
    manifest = load_manifest(index_path, name_suffix)
    offsets = []
    current = 0
    for i in range(num_chunks):
        entry = manifest.get(f"{i}.{name_suffix}.npy")
        if not entry or not entry.get("merged"):
            return None
        offsets.append(current)
        current += entry["rows"]
    return offsets


def write_tch_npy(path: str, arr: np.ndarray) -> None:
    """Write a .npy file byte-identical to ``tch::Tensor::write_npy``.

    Staging shards are read back by the Rust core, which expects tch's header
    formatting: Rust-style shape tuples ``(R,C,)``, explicit ``<`` byte order even
    for 1-byte types, and the prologue padded to a multiple of 16.
    """
    arr = np.ascontiguousarray(arr)
    kind_size = arr.dtype.kind + str(arr.dtype.itemsize)
    descr = f"<{kind_size}"

    if arr.ndim == 1:
        shape_str = f"({arr.shape[0]},)"
    else:
        shape_str = "(" + ",".join(str(s) for s in arr.shape) + ",)"

    header_body = (
        f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}"
    )

    prologue_fixed = 10  # magic(6) + version(2) + header_len_field(2)
    body_with_newline = len(header_body) + 1
    target = ((prologue_fixed + body_with_newline + 15) // 16) * 16
    pad = target - prologue_fixed - body_with_newline

    header = header_body + (" " * pad) + "\n"
    header_bytes = header.encode("ascii")

    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00")
        f.write(len(header_bytes).to_bytes(2, "little"))
        f.write(header_bytes)
        f.write(arr.tobytes())


def materialize_shards(
    index_path: str,
    num_chunks: int,
    name_suffix: str,
    chunk_indices: list[int] | None = None,
) -> None:
    """Recreate missing shard payload files from the merged file.

    Called before Rust ``update`` / ``delete``, which read shards directly. Rows are
    sliced from the merged file at the offsets the manifest records, written in tch
    format, and the manifest entry switches from merged to a file mtime — so the
    merge scan sees a clean, unchanged chain and skips rewriting them.
    """
    manifest = load_manifest(index_path, name_suffix)
    merged_path = os.path.join(index_path, f"merged_{name_suffix}.npy")
    wanted = range(num_chunks) if chunk_indices is None else chunk_indices

    to_create = []
    offset = 0
    for i in range(num_chunks):
        filename = f"{i}.{name_suffix}.npy"
        entry = manifest.get(filename)
        # An index frozen by an older version has no manifest; the merged file's
        # layout is the doclens order by construction, so rows derive from there.
        rows = entry["rows"] if entry else shard_rows(index_path, i)
        shard_path = os.path.join(index_path, filename)
        if i in wanted and not os.path.exists(shard_path):
            if not os.path.exists(merged_path):
                merged_name = os.path.basename(merged_path)  # noqa: PTH119
                error = (
                    f"Shard {filename} is missing and there is no "
                    f"{merged_name}; the index at {index_path} is incomplete."
                )
                raise FileNotFoundError(error)
            to_create.append((i, filename, shard_path, offset, rows))
        offset += rows

    if not to_create:
        return

    merged = np.load(merged_path, mmap_mode="r")
    changed = False
    for _, filename, shard_path, start, rows in to_create:
        write_tch_npy(shard_path, np.ascontiguousarray(merged[start : start + rows]))
        manifest[filename] = {
            "rows": rows,
            "mtime": os.stat(shard_path).st_mtime,  # noqa: PTH116
            "merged": True,  # rows are still in the merged file, byte-identical
        }
        changed = True
    del merged

    if changed:
        save_manifest(index_path, name_suffix, manifest)


def drop_shard_payloads(index_path: str, name_suffix: str, num_chunks: int) -> None:
    """Unlink shard payload files whose rows the manifest records as merged.

    Runs only after the merge and its manifest write succeeded, so every unlinked
    byte already exists in the merged file. Ordering makes a crash harmless: a file
    that survives an unlink attempt is re-dropped on the next load.
    """
    if keep_shards():
        return
    manifest = load_manifest(index_path, name_suffix)
    changed = False
    for i in range(num_chunks):
        filename = f"{i}.{name_suffix}.npy"
        entry = manifest.get(filename)
        if not entry or not entry.get("merged"):
            continue
        shard_path = os.path.join(index_path, filename)
        if os.path.exists(shard_path):
            try:
                os.remove(shard_path)
            except OSError:
                continue
        # Keep the key with a null value: older versions subscript
        # entry["mtime"] directly and must see a mismatch, not a KeyError.
        if entry.get("mtime") is not None:
            entry["mtime"] = None
            changed = True
    if changed:
        save_manifest(index_path, name_suffix, manifest)


def mark_single_copy(index_path: str) -> None:
    """Stamp metadata.json so older fast-plaid versions open this index read-only.

    ``"frozen": true`` makes pre-1.7 versions take their frozen loading path (merged
    file only — exactly what exists) and refuse mutation, instead of silently seeing
    an empty index. ``"mutable": true`` tells 1.7+ that mutation is allowed because
    shards can be re-materialized on demand. ``unfreeze()`` on an old version fully
    converts the index back to the double-copy layout.
    """
    meta_path = os.path.join(index_path, "metadata.json")
    if not os.path.exists(meta_path):
        return
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("frozen") and meta.get("mutable"):
        return
    # A legacy frozen index stays immutable: its owner asked for that explicitly.
    if meta.get("frozen") and "mutable" not in meta:
        return
    meta["frozen"] = True
    meta["mutable"] = True
    tmp_path = meta_path + ".tmp"
    # newline="\n" prevents Windows text-mode translation; Rust's serde_json
    # always uses \n regardless of platform.
    with open(tmp_path, "w", newline="\n") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_path, meta_path)  # noqa: PTH105


def is_mutable(index_path: str) -> bool:
    """Whether mutation is allowed: not frozen, or frozen-but-single-copy."""
    meta_path = os.path.join(index_path, "metadata.json")
    if not os.path.exists(meta_path):
        return True
    with open(meta_path) as f:
        meta = json.load(f)
    if not meta.get("frozen", False):
        return True
    return bool(meta.get("mutable", False))
