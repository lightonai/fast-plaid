use anyhow::Result;
use serde::Deserialize;
use tch::{Device, Kind, Tensor};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;

use crate::search::tensor::StridedTensor;
use crate::utils::errors::anyhow_to_pyerr;
use crate::utils::residual_codec::ResidualCodec;

/// Parses a Python-style device string into a `tch::Device`.
///
/// Supports "cpu", "cuda", and specific GPU indices like "cuda:1".
pub fn get_device(device: &str) -> Result<Device, PyErr> {
    match device.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)),
        s if s.starts_with("cuda:") => {
            let parts: Vec<&str> = s.split(':').collect();
            if parts.len() == 2 {
                parts[1].parse::<usize>().map(Device::Cuda).map_err(|_| {
                    PyValueError::new_err(format!("Invalid CUDA device index: '{}'", parts[1]))
                })
            } else {
                Err(PyValueError::new_err(
                    "Invalid CUDA device format. Expected 'cuda:N'.",
                ))
            }
        },
        _ => Err(PyValueError::new_err(format!(
            "Unsupported device string: '{}'",
            device
        ))),
    }
}

#[derive(Deserialize, Debug)]
pub struct Metadata {
    pub num_chunks: usize,
    pub nbits: i64,
}

/// The core struct holding all immutable data required for search operations.
///
/// This struct is designed to be shared across threads. It contains the
/// quantization codec (centroids, weights) and the document index structures
/// (IVF lists, compressed codes, and residuals).
pub struct LoadedIndex {
    pub codec: ResidualCodec,
    pub ivf_index_strided: StridedTensor,
    pub doc_codes_strided: StridedTensor,
    pub doc_residuals_strided: StridedTensor,
    pub nbits: i64,
}

unsafe impl Send for LoadedIndex {}
unsafe impl Sync for LoadedIndex {}

// ----------------------------------------------------------------------------
// PyLoadedIndex Wrapper
// ----------------------------------------------------------------------------

/// A wrapper around the Rust `LoadedIndex` struct that can be held by Python.
///
/// This wrapper allows the Python runtime to manage the lifetime of the
/// underlying Rust index structure. When the Python object is garbage collected,
/// the Rust memory is freed.
#[pyclass]
pub struct PyLoadedIndex {
    pub inner: LoadedIndex,
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/// Ensures the tensor is on the target device and kind without copying if not necessary.
///
/// This is critical for memory-mapped tensors. Calling `to_device` blindly on a
/// CPU mmap tensor will force a load into RAM, even if the target is also CPU.
fn ensure_tensor(t: PyTensor, device: Device, kind: Kind) -> Tensor {
    // PyTensor derefs to Tensor. We take a shallow reference first.
    let mut res: Tensor = t.shallow_clone();

    // Only convert device if different (avoids copy/move overhead)
    if res.device() != device {
        res = res.to_device(device);
    }

    // Only convert kind if different (avoids casting overhead)
    if res.kind() != kind {
        res = res.to_kind(kind);
    }

    res
}

/// Validates that a memory-mapped tensor has sufficient padding for strided access.
///
/// `StridedTensor` creates a sliding window view over a flat array. For the very last
/// element in the array, the window (stride) extends beyond the start of that element.
/// If the physical file ends exactly at the last element's data, the stride view
/// would access invalid memory.
///
///
///
/// This function calculates if the underlying storage has enough "ghost" padding
/// at the end to allow for a zero-copy view. If not, it returns an error advising
/// the user to pad the file in Python.
///
/// # Arguments
///
/// * `data_tensor` - The flat data tensor (possibly memory mapped).
/// * `lengths_tensor` - The lengths of the sequences stored in `data_tensor`.
/// * `tensor_name` - A label for the error message.
fn check_mmap_padding(
    data_tensor: &Tensor,
    lengths_tensor: &Tensor,
    tensor_name: &str,
) -> PyResult<()> {
    let num_docs = lengths_tensor.size()[0];
    if num_docs == 0 {
        return Ok(());
    }

    // Move computations to CPU for scalar extraction to avoid sync overhead if on GPU
    let lengths_cpu = if lengths_tensor.device() != Device::Cpu {
        lengths_tensor.to_device(Device::Cpu)
    } else {
        lengths_tensor.shallow_clone()
    };

    let max_len = lengths_cpu.max().int64_value(&[]);
    let last_len = lengths_cpu.int64_value(&[num_docs - 1]);

    // Calculate the logical end of the data
    let total_len = lengths_cpu.sum(Kind::Int64).int64_value(&[]);

    // Logic:
    // The view for the *last* element starts at: `total_len - last_len`.
    // The view requires a width of `max_len` (the stride size).
    // Therefore, valid memory must exist up to: `(total_len - last_len) + max_len`.
    let start_of_last = total_len - last_len;
    let required_size = start_of_last + max_len;

    let current_size = data_tensor.size()[0];

    if current_size < required_size {
        let missing = required_size - current_size;
        return Err(PyValueError::new_err(format!(
            "Memory Map Error: The '{}' tensor is too small for zero-copy strided access. \
            It requires {} elements of padding at the end. \
            Please pad the .npy file in Python (e.g. using .resize() or during creation) \
            to avoid loading the entire file into RAM.",
            tensor_name, missing
        )));
    }

    Ok(())
}

// ----------------------------------------------------------------------------
// Index Construction
// ----------------------------------------------------------------------------

/// Constructs the internal Index object from raw tensors loaded in Python.
///
/// This function acts as the bridge between Python's file loading and Rust's
/// search engine. It organizes the raw tensors into a `LoadedIndex` struct.
///
///
///
/// # Key Behavior
/// - **Zero-Copy Optimization**: If `device` is "cpu", large tensors (codes, residuals)
///   are assumed to be memory-mapped. The function verifies padding and uses them
///   directly without allocation.
/// - **Codec Handling**: Small tensors (centroids, weights) are loaded into RAM/VRAM
///   immediately for fast lookup during decompression.
///
/// # Arguments
///
/// * `nbits` - The quantization bit-width (e.g., 2 or 4).
/// * `centroids` - The coarse centroids (float16).
/// * `avg_residual` - The average residual vector (float16).
/// * `bucket_cutoffs` - Quantization bucket boundaries (float16).
/// * `bucket_weights` - Quantization bucket values (float16).
/// * `ivf` - The Inverted File index structure (int64).
/// * `ivf_lengths` - Lengths of the IVF lists (int32).
/// * `doc_codes` - The compressed document codes (int64).
/// * `doc_residuals` - The compressed document residuals (uint8).
/// * `doc_lengths` - The true lengths of documents (int64).
/// * `device` - The target device string.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn construct_index(
    _py: Python<'_>,
    nbits: i64,
    centroids: PyTensor,
    avg_residual: PyTensor,
    bucket_cutoffs: PyTensor,
    bucket_weights: PyTensor,
    ivf: PyTensor,
    ivf_lengths: PyTensor,
    doc_codes: PyTensor,
    doc_residuals: PyTensor,
    doc_lengths: PyTensor,
    device: String,
) -> PyResult<PyLoadedIndex> {
    let device_tch = get_device(&device)?;

    // 1. Codec Loading
    // These tensors are small (MBs), so we ensure they are on the correct device/kind immediately.
    let codec = ResidualCodec::load(
        nbits,
        ensure_tensor(centroids, device_tch, Kind::Half),
        ensure_tensor(avg_residual, device_tch, Kind::Half),
        Some(ensure_tensor(bucket_cutoffs, device_tch, Kind::Half)),
        Some(ensure_tensor(bucket_weights, device_tch, Kind::Half)),
        device_tch,
    )
    .map_err(anyhow_to_pyerr)?;

    // 2. IVF Index
    // Standard strided tensor construction for the inverted file lists.
    let ivf_index_strided = StridedTensor::new(
        ensure_tensor(ivf, device_tch, Kind::Int64),
        ensure_tensor(ivf_lengths, device_tch, Kind::Int),
        device_tch,
    );

    // 3. Document Data
    // These are the large tensors (GBs). We must handle them carefully.

    // Lengths are needed for checking padding and structure.
    let doc_lens_t = ensure_tensor(doc_lengths, device_tch, Kind::Int64);
    let doc_codes_t = ensure_tensor(doc_codes, device_tch, Kind::Int64);
    let doc_residuals_t = ensure_tensor(doc_residuals, device_tch, Kind::Uint8);

    // We validate padding here to prevent `StridedTensor` from triggering a massive data copy
    // to add padding in RAM, which would defeat the purpose of mmap.
    if device_tch == Device::Cpu {
        check_mmap_padding(&doc_codes_t, &doc_lens_t, "doc_codes")?;
        check_mmap_padding(&doc_residuals_t, &doc_lens_t, "doc_residuals")?;
    }

    let doc_codes_strided = StridedTensor::new(doc_codes_t, doc_lens_t.shallow_clone(), device_tch);

    let doc_residuals_strided = StridedTensor::new(doc_residuals_t, doc_lens_t, device_tch);

    let loaded_index = LoadedIndex {
        codec,
        ivf_index_strided,
        doc_codes_strided,
        doc_residuals_strided,
        nbits,
    };

    Ok(PyLoadedIndex {
        inner: loaded_index,
    })
}
