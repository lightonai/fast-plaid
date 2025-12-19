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

/// A wrapper around the Rust `LoadedIndex` struct that can be held by Python.
///
/// This wrapper allows the Python runtime to manage the lifetime of the
/// underlying Rust index structure. When the Python object is garbage collected,
/// the Rust memory is freed.
#[pyclass]
pub struct PyLoadedIndex {
    pub inner: LoadedIndex,
}

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
/// - **Low Memory Mode**: If `low_memory` is true, the large document tensors are strictly
///   kept on the CPU, even if the target `device` is CUDA.
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
/// * `device` - The target device string (e.g. "cuda:0").
/// * `low_memory` - If true, keeps document data on CPU.
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
    low_memory: bool,
) -> PyResult<PyLoadedIndex> {
    let main_device = get_device(&device)?;

    // Force document tensors to CPU in low memory mode
    let storage_device = if low_memory { Device::Cpu } else { main_device };

    // Load codec (small tensors)
    let codec = ResidualCodec::load(
        nbits,
        ensure_tensor(centroids, main_device, Kind::Half),
        ensure_tensor(avg_residual, main_device, Kind::Half),
        Some(ensure_tensor(bucket_cutoffs, main_device, Kind::Half)),
        Some(ensure_tensor(bucket_weights, main_device, Kind::Half)),
        main_device,
    )
    .map_err(anyhow_to_pyerr)?;

    // Build IVF index
    let ivf_index_strided = StridedTensor::new(
        ensure_tensor(ivf, main_device, Kind::Int64),
        ensure_tensor(ivf_lengths, main_device, Kind::Int),
        main_device,
    );

    // Load document data (large tensors, may stay on CPU in low memory mode)
    let doc_lens_t = ensure_tensor(doc_lengths, storage_device, Kind::Int64);
    let doc_codes_t = ensure_tensor(doc_codes, storage_device, Kind::Int64);
    let doc_residuals_t = ensure_tensor(doc_residuals, storage_device, Kind::Uint8);

    let doc_codes_strided =
        StridedTensor::new(doc_codes_t, doc_lens_t.shallow_clone(), storage_device);

    let doc_residuals_strided = StridedTensor::new(doc_residuals_t, doc_lens_t, storage_device);

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
