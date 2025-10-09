// rust/index/delete.rs

use anyhow::Result;
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::index::create::optimize_ivf;
use crate::search::load::LoadedIndex;

/// Deletes documents from an existing FastPlaid index.
///
/// This function removes specified documents by rewriting the index chunks
/// they belong to and then rebuilding the IVF index.
///
/// # Arguments
///
/// * `subset` - A slice of document IDs to be removed from the index.
/// * `idx_path` - The directory path of the index to modify.
/// * `device` - The `tch::Device` (e.g., CPU or CUDA) on which to perform computations.
///
/// # Returns
///
/// A `Result` indicating success or failure.
pub fn delete_from_index(subset: &[i64], idx_path: &str, device: Device) -> Result<()> {
    let _grad_guard = tch::no_grad_guard();
    let idx_path_obj = Path::new(idx_path);

    // Load main metadata
    let main_meta_path = idx_path_obj.join("metadata.json");
    let main_meta_file = File::open(&main_meta_path)?;
    let main_meta: serde_json::Value = serde_json::from_reader(BufReader::new(main_meta_file))?;
    let num_chunks = main_meta["num_chunks"].as_u64().unwrap() as usize;
    let nbits = main_meta["nbits"].as_i64().unwrap();
    let est_total_embs = main_meta["num_partitions"].as_i64().unwrap();

    let ids_to_delete_set: HashSet<i64> = subset.iter().cloned().collect();
    let mut current_doc_offset = 0;
    let mut total_embs = 0;

    for chunk_idx in 0..num_chunks {
        let doclens_path = idx_path_obj.join(format!("doclens.{}.json", chunk_idx));
        let doclens_file = File::open(&doclens_path)?;
        let doclens: Vec<i64> = serde_json::from_reader(BufReader::new(doclens_file))?;

        let mut new_doclens = Vec::new();
        let mut embs_to_keep_mask = Vec::new();
        let mut _embs_in_chunk = 0;

        for (i, &len) in doclens.iter().enumerate() {
            let doc_id = current_doc_offset + i as i64;
            _embs_in_chunk += len;
            if !ids_to_delete_set.contains(&doc_id) {
                new_doclens.push(len);
                for _ in 0..len {
                    embs_to_keep_mask.push(true);
                }
            } else {
                for _ in 0..len {
                    embs_to_keep_mask.push(false);
                }
            }
        }

        if new_doclens.len() < doclens.len() {
            // Rewrite doclens
            let new_doclens_file = File::create(&doclens_path)?;
            serde_json::to_writer(BufWriter::new(new_doclens_file), &new_doclens)?;

            let embs_to_keep_tensor = Tensor::from_slice(&embs_to_keep_mask).to_device(device);

            // Rewrite codes
            let codes_path = idx_path_obj.join(format!("{}.codes.npy", chunk_idx));
            let codes = Tensor::read_npy(&codes_path)?.to_device(device);
            let new_codes = codes.masked_select(&embs_to_keep_tensor);
            new_codes.write_npy(&codes_path)?;

            // Rewrite residuals
            let residuals_path = idx_path_obj.join(format!("{}.residuals.npy", chunk_idx));
            let residuals = Tensor::read_npy(&residuals_path)?.to_device(device);
            let new_residuals = residuals.masked_select(&embs_to_keep_tensor.unsqueeze(-1));
            let new_residuals_shape = [-1, residuals.size()[1]];
            new_residuals.reshape(&new_residuals_shape).write_npy(&residuals_path)?;

            // Update metadata
            let chunk_meta_path = idx_path_obj.join(format!("{}.metadata.json", chunk_idx));
            let chunk_meta_file = File::open(&chunk_meta_path)?;
            let mut chunk_meta: serde_json::Value =
                serde_json::from_reader(BufReader::new(chunk_meta_file))?;
            chunk_meta["num_passages"] = serde_json::json!(new_doclens.len());
            chunk_meta["num_embeddings"] = serde_json::json!(new_codes.size()[0]);
            let new_chunk_meta_file = File::create(&chunk_meta_path)?;
            serde_json::to_writer_pretty(BufWriter::new(new_chunk_meta_file), &chunk_meta)?;
        }
        total_embs += new_doclens.iter().sum::<i64>();
        current_doc_offset += doclens.len() as i64;
    }

    // Recreate IVF
    let all_codes = Tensor::zeros(&[total_embs], (Kind::Int64, device));
    let mut current_emb_offset = 0;
    for chk_idx in 0..num_chunks {
        let codes_fpath_for_global = idx_path_obj.join(format!("{}.codes.npy", chk_idx));
        let codes_from_file = Tensor::read_npy(&codes_fpath_for_global)?.to_device(device);
        let codes_in_chk_count = codes_from_file.size()[0];
        all_codes
            .narrow(0, current_emb_offset, codes_in_chk_count)
            .copy_(&codes_from_file);
        current_emb_offset += codes_in_chk_count;
    }

    let (sorted_codes, sorted_indices) = all_codes.sort(0, false);
    let code_counts = sorted_codes.bincount::<Tensor>(None, est_total_embs);
    let (opt_ivf, opt_ivf_lens) = optimize_ivf(&sorted_indices, &code_counts, idx_path, device)?;

    opt_ivf.write_npy(&idx_path_obj.join("ivf.npy"))?;
    opt_ivf_lens.write_npy(&idx_path_obj.join("ivf_lengths.npy"))?;

    // Update main metadata
    let doclens_re = regex::Regex::new(r"doclens\.(\d+)\.json")?;
    let mut total_passages = 0;
    for entry in fs::read_dir(idx_path)? {
        let entry = entry?;
        let fname = entry.file_name();
        if let Some(fname_str) = fname.to_str() {
            if doclens_re.is_match(fname_str) {
                let file = File::open(entry.path())?;
                let doclens: Vec<i64> = serde_json::from_reader(BufReader::new(file))?;
                total_passages += doclens.len();
            }
        }
    }

    let final_avg_doclen = if total_passages > 0 {
        total_embs as f64 / total_passages as f64
    } else {
        0.0
    };

    let final_meta_json = json!({
        "num_chunks": num_chunks,
        "nbits": nbits,
        "num_partitions": est_total_embs,
        "num_embeddings": total_embs,
        "avg_doclen": final_avg_doclen,
    });

    let final_meta_file = fs::File::create(&main_meta_path)?;
    serde_json::to_writer_pretty(BufWriter::new(final_meta_file), &final_meta_json)?;

    Ok(())
}

/// Deletes documents from a loaded index in-memory.
///
/// This function takes an already loaded index and removes specified documents
/// by modifying the existing LoadedIndex structure directly. This is more efficient
/// than reloading the entire index from disk after deletion.
///
/// # Arguments
///
/// * `loaded_index` - A mutable reference to the existing LoadedIndex to update
/// * `subset` - A slice of document IDs to be removed from the index
/// * `idx_path` - The directory path where the index files will be updated on disk
/// * `device` - The `tch::Device` on which to perform computations
///
/// # Returns
///
/// A `Result` indicating success or failure of the deletion operation
pub fn delete_from_loaded_index(
    loaded_index: &mut LoadedIndex,
    subset: &[i64],
    idx_path: &str,
    device: Device,
) -> Result<()> {
    let _grad_guard = tch::no_grad_guard();
    let idx_path_obj = Path::new(idx_path);

    // Load main metadata
    let main_meta_path = idx_path_obj.join("metadata.json");
    let main_meta_file = File::open(&main_meta_path)?;
    let main_meta: serde_json::Value = serde_json::from_reader(BufReader::new(main_meta_file))?;
    let num_chunks = main_meta["num_chunks"].as_u64().unwrap() as usize;
    let nbits = loaded_index.nbits;
    let est_total_embs = main_meta["num_partitions"].as_i64().unwrap();

    let ids_to_delete_set: HashSet<i64> = subset.iter().cloned().collect();
    let mut current_doc_offset = 0;
    let mut total_embs = 0;

    // Process each chunk - similar to delete_from_index but update disk files
    for chunk_idx in 0..num_chunks {
        let doclens_path = idx_path_obj.join(format!("doclens.{}.json", chunk_idx));
        let doclens_file = File::open(&doclens_path)?;
        let doclens: Vec<i64> = serde_json::from_reader(BufReader::new(doclens_file))?;

        let mut new_doclens = Vec::new();
        let mut embs_to_keep_mask = Vec::new();
        let mut _embs_in_chunk = 0;

        for (i, &len) in doclens.iter().enumerate() {
            let doc_id = current_doc_offset + i as i64;
            _embs_in_chunk += len;
            if !ids_to_delete_set.contains(&doc_id) {
                new_doclens.push(len);
                for _ in 0..len {
                    embs_to_keep_mask.push(true);
                }
            } else {
                for _ in 0..len {
                    embs_to_keep_mask.push(false);
                }
            }
        }

        if new_doclens.len() < doclens.len() {
            // Rewrite doclens
            let new_doclens_file = File::create(&doclens_path)?;
            serde_json::to_writer(BufWriter::new(new_doclens_file), &new_doclens)?;

            let embs_to_keep_tensor = Tensor::from_slice(&embs_to_keep_mask).to_device(device);

            // Rewrite codes
            let codes_path = idx_path_obj.join(format!("{}.codes.npy", chunk_idx));
            let codes = Tensor::read_npy(&codes_path)?.to_device(device);
            let new_codes = codes.masked_select(&embs_to_keep_tensor);
            new_codes.to_device(tch::Device::Cpu).write_npy(&codes_path)?;

            // Rewrite residuals
            let residuals_path = idx_path_obj.join(format!("{}.residuals.npy", chunk_idx));
            let residuals = Tensor::read_npy(&residuals_path)?.to_device(device);
            let new_residuals = residuals.masked_select(&embs_to_keep_tensor.unsqueeze(-1));
            let new_residuals_shape = [-1, residuals.size()[1]];
            new_residuals.reshape(&new_residuals_shape).to_device(tch::Device::Cpu).write_npy(&residuals_path)?;

            // Update metadata
            let chunk_meta_path = idx_path_obj.join(format!("{}.metadata.json", chunk_idx));
            let chunk_meta_file = File::open(&chunk_meta_path)?;
            let mut chunk_meta: serde_json::Value =
                serde_json::from_reader(BufReader::new(chunk_meta_file))?;
            chunk_meta["num_passages"] = serde_json::json!(new_doclens.len());
            chunk_meta["num_embeddings"] = serde_json::json!(new_codes.size()[0]);
            let new_chunk_meta_file = File::create(&chunk_meta_path)?;
            serde_json::to_writer_pretty(BufWriter::new(new_chunk_meta_file), &chunk_meta)?;
        }
        total_embs += new_doclens.iter().sum::<i64>();
        current_doc_offset += doclens.len() as i64;
    }

    // Recreate IVF on disk
    let all_codes = Tensor::zeros(&[total_embs], (Kind::Int64, device));
    let mut current_emb_offset = 0;
    for chk_idx in 0..num_chunks {
        let codes_fpath_for_global = idx_path_obj.join(format!("{}.codes.npy", chk_idx));
        let codes_from_file = Tensor::read_npy(&codes_fpath_for_global)?.to_device(device);
        let codes_in_chk_count = codes_from_file.size()[0];
        all_codes
            .narrow(0, current_emb_offset, codes_in_chk_count)
            .copy_(&codes_from_file);
        current_emb_offset += codes_in_chk_count;
    }

    let (sorted_codes, sorted_indices) = all_codes.sort(0, false);
    let code_counts = sorted_codes.bincount::<Tensor>(None, est_total_embs);
    let (opt_ivf, opt_ivf_lens) = optimize_ivf(&sorted_indices, &code_counts, idx_path, device)?;

    opt_ivf.to_device(tch::Device::Cpu).write_npy(&idx_path_obj.join("ivf.npy"))?;
    opt_ivf_lens.to_device(tch::Device::Cpu).write_npy(&idx_path_obj.join("ivf_lengths.npy"))?;

    // Update main metadata
    let doclens_re = regex::Regex::new(r"doclens\.(\d+)\.json")?;
    let mut total_passages = 0;
    for entry in fs::read_dir(idx_path)? {
        let entry = entry?;
        let fname = entry.file_name();
        if let Some(fname_str) = fname.to_str() {
            if doclens_re.is_match(fname_str) {
                let file = File::open(entry.path())?;
                let doclens: Vec<i64> = serde_json::from_reader(BufReader::new(file))?;
                total_passages += doclens.len();
            }
        }
    }

    let final_avg_doclen = if total_passages > 0 {
        total_embs as f64 / total_passages as f64
    } else {
        0.0
    };

    let final_meta_json = json!({
        "num_chunks": num_chunks,
        "nbits": nbits,
        "num_partitions": est_total_embs,
        "num_embeddings": total_embs,
        "avg_doclen": final_avg_doclen,
    });

    let final_meta_file = fs::File::create(&main_meta_path)?;
    serde_json::to_writer_pretty(BufWriter::new(final_meta_file), &final_meta_json)?;

    // Update the in-memory loaded index with the new data
    // Load the updated IVF data
    let updated_ivf_data = Tensor::read_npy(&idx_path_obj.join("ivf.npy"))?
        .to_kind(Kind::Int64)
        .to_device(device);
    let updated_ivf_lengths = Tensor::read_npy(&idx_path_obj.join("ivf_lengths.npy"))?
        .to_kind(Kind::Int64)
        .to_device(device);
    
    // Update the IVF index in the loaded index
    use crate::search::tensor::StridedTensor;
    loaded_index.ivf_index_strided = StridedTensor::new(updated_ivf_data, updated_ivf_lengths, device);

    // Load all document lengths after deletion
    let mut all_doc_lens_vec: Vec<i64> = Vec::new();
    for chk_idx in 0..num_chunks {
        let chunk_doclens_path = idx_path_obj.join(format!("doclens.{}.json", chk_idx));
        let doclens_file = File::open(&chunk_doclens_path)?;
        let doclens_reader = BufReader::new(doclens_file);
        let chunk_doc_lens: Vec<i64> = serde_json::from_reader(doclens_reader)?;
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

    // Rebuild the codes and residuals tensors with the filtered data
    let embedding_dim = loaded_index.codec.centroids.size()[1];
    let full_codes_preallocated = Tensor::empty(&[total_embs_from_doclens], (Kind::Int64, device));
    let residual_element_size = (embedding_dim * nbits) / 8;
    let full_residuals_preallocated = Tensor::empty(
        &[total_embs_from_doclens, residual_element_size],
        (Kind::Uint8, device),
    );

    let mut current_write_offset = 0i64;
    for chk_idx in 0..num_chunks {
        let chunk_codes_path = idx_path_obj.join(format!("{}.codes.npy", chk_idx));
        let chunk_residuals_path = idx_path_obj.join(format!("{}.residuals.npy", chk_idx));

        let chunk_codes_tensor = Tensor::read_npy(&chunk_codes_path)?
            .to_kind(Kind::Int64)
            .to_device(device);
        let chunk_residuals_tensor = Tensor::read_npy(&chunk_residuals_path)?
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

    // Update the strided tensors in the loaded index
    loaded_index.doc_codes_strided = StridedTensor::new(final_codes, all_doc_lengths.shallow_clone(), device);
    loaded_index.doc_residuals_strided = StridedTensor::new(final_residuals, all_doc_lengths.shallow_clone(), device);

    Ok(())
}