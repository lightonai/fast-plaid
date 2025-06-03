use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::search::tensor::StridedTensor;
use crate::utils::residual_codec::ResidualCodec;

/// Holds metadata, typically loaded from a configuration or header file.
///
/// This struct is designed to be deserialized from a format like JSON or TOML,
/// capturing key properties about an associated dataset, such as its structure
/// or encoding scheme.
#[derive(Deserialize, Debug)]
pub struct Metadata {
    /// The number of chunks or segments the data is divided into.
    pub num_chunks: usize,
    /// The number of bits used for data representation, often for quantization.
    pub nbits: i64,
}

/// Represents a loaded search index and its associated data tensors.
///
/// This structure holds all the necessary components for performing a search,
/// including the quantization codec and strided tensors for efficient data access.
pub struct LoadedIndex {
    /// The residual codec used for encoding and decoding vector residuals.
    pub codec: ResidualCodec,
    /// A strided tensor containing the IVF (Inverted File) index.
    pub ivf_index_strided: StridedTensor,
    /// A strided tensor of document codes.
    pub doc_codes_strided: StridedTensor,
    /// A strided tensor of document residuals.
    pub doc_residuals_strided: StridedTensor,
    /// The number of bits used for quantization in the codec.
    pub nbits: i64,
}

/// Loads an index from a specified directory and prepares it for searching.
///
/// This function reads index metadata, loads tensor data from `.npy` files,
/// and assembles them into a `LoadedIndex` struct. It handles loading tensors
/// onto the specified compute device and dynamically loads the required Torch library.
///
/// # Arguments
///
/// * `index_dir_path_str` - The file path to the directory containing the index files.
/// * `device` - The `tch::Device` (e.g., `Device::Cuda(0)` or `Device::Cpu`) onto which the tensors will be loaded.
///
/// # Returns
///
/// A `Result` containing the fully loaded `LoadedIndex` on success, or an error
/// if any part of the loading process fails (e.g., file not found, parsing error).
///
/// # Errors
///
/// This function will return an error if:
/// - The Torch library path cannot be converted to a C-style string.
/// - The `metadata.json` file is missing or cannot be parsed.
/// - Any of the required `.npy` or `.json` files are missing or corrupt.
/// - Tensors cannot be loaded onto the specified device.
pub fn load_index(index_dir_path_str: &str, device: Device) -> Result<LoadedIndex> {
    let index_dir_path = Path::new(index_dir_path_str);
    let metadata_content_raw = std::fs::read(index_dir_path.join("metadata.json"))
        .map_err(|e| anyhow!("Failed to read metadata.json: {}", e))?;
    let app_metadata: Metadata = serde_json::from_slice(&metadata_content_raw)
        .map_err(|e| anyhow!("Failed to parse metadata.json: {}", e))?;

    let nbits_metadata: i64 = app_metadata.nbits;
    let num_chunks_metadata: usize = app_metadata.num_chunks;

    let centroids_initial_data = Tensor::read_npy(index_dir_path.join("centroids.npy"))?
        .to_kind(Kind::Half)
        .to_device(device);
    let avg_residual_initial_data = Tensor::read_npy(index_dir_path.join("avg_residual.npy"))?
        .to_kind(Kind::Half)
        .to_device(device);
    let bucket_cutoffs_initial_data = Tensor::read_npy(index_dir_path.join("bucket_cutoffs.npy"))?
        .to_kind(Kind::Half)
        .to_device(device);
    let bucket_weights_initial_data = Tensor::read_npy(index_dir_path.join("bucket_weights.npy"))?
        .to_kind(Kind::Half)
        .to_device(device);

    let index_dimension = centroids_initial_data.size()[1];

    let codec = ResidualCodec::load(
        nbits_metadata,
        centroids_initial_data,
        avg_residual_initial_data,
        Some(bucket_cutoffs_initial_data),
        Some(bucket_weights_initial_data),
        device,
    )?;

    let ivf_data = Tensor::read_npy(index_dir_path.join("ivf.npy"))?
        .to_kind(Kind::Int64)
        .to_device(device);
    let ivf_lengths = Tensor::read_npy(index_dir_path.join("ivf_lengths.npy"))?
        .to_kind(Kind::Int64)
        .to_device(device);
    let ivf_index_strided = StridedTensor::new(ivf_data, ivf_lengths, device);

    let mut all_doc_lens_vec: Vec<i64> = Vec::new();
    for chunk_idx in 0..num_chunks_metadata {
        let chunk_doclens_path = index_dir_path.join(format!("doclens.{}.json", chunk_idx));
        let doclens_file = File::open(&chunk_doclens_path)
            .map_err(|e| anyhow!("Unable to open {:?}: {}", chunk_doclens_path, e))?;
        let doclens_reader = BufReader::new(doclens_file);
        let chunk_doc_lens: Vec<i64> = serde_json::from_reader(doclens_reader)
            .map_err(|e| anyhow!("Failed to parse JSON from {:?}: {}", chunk_doclens_path, e))?;
        all_doc_lens_vec.extend(chunk_doc_lens);
    }

    let all_doc_lengths = Tensor::from_slice(&all_doc_lens_vec)
        .to_kind(Kind::Int64)
        .to_device(device);

    let total_embs_from_doclens = if all_doc_lengths.numel() > 0 {
        all_doc_lengths.sum(Kind::Int64).int64_value(&[])
    } else {
        0
    };

    let full_codes_preallocated = Tensor::empty(&[total_embs_from_doclens], (Kind::Int64, device));
    let residual_element_size = (index_dimension * nbits_metadata) / 8;
    let full_residuals_preallocated = Tensor::empty(
        &[total_embs_from_doclens, residual_element_size],
        (Kind::Uint8, device),
    );

    let mut current_write_offset = 0i64;
    for chunk_idx in 0..num_chunks_metadata {
        let chunk_codes_path = index_dir_path.join(format!("{}.codes.npy", chunk_idx));
        let chunk_residuals_path = index_dir_path.join(format!("{}.residuals.npy", chunk_idx));

        let chunk_codes_tensor = Tensor::read_npy(&chunk_codes_path)
            .map_err(|e| anyhow!("Failed to read codes {:?}: {}", chunk_codes_path, e))?
            .to_kind(Kind::Int64)
            .to_device(device);
        let chunk_residuals_tensor = Tensor::read_npy(&chunk_residuals_path)
            .map_err(|e| anyhow!("Failed to read residuals {:?}: {}", chunk_residuals_path, e))?
            .to_kind(Kind::Uint8)
            .to_device(device);

        let num_elements_in_chunk = chunk_codes_tensor.size()[0];

        if num_elements_in_chunk > 0 {
            full_codes_preallocated
                .narrow(0, current_write_offset, num_elements_in_chunk)
                .copy_(&chunk_codes_tensor);
            full_residuals_preallocated
                .narrow(0, current_write_offset, num_elements_in_chunk)
                .copy_(&chunk_residuals_tensor);
            current_write_offset += num_elements_in_chunk;
        }
    }

    let final_codes = full_codes_preallocated.narrow(0, 0, current_write_offset);
    let final_residuals = full_residuals_preallocated.narrow(0, 0, current_write_offset);

    let doc_codes_strided =
        StridedTensor::new(final_codes, all_doc_lengths.shallow_clone(), device);
    let doc_residuals_strided =
        StridedTensor::new(final_residuals, all_doc_lengths.shallow_clone(), device);

    Ok(LoadedIndex {
        codec,
        ivf_index_strided,
        doc_codes_strided,
        doc_residuals_strided,
        nbits: nbits_metadata,
    })
}
