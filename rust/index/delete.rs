use anyhow::Result;
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::index::create::optimize_ivf;

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
    let est_num_embeddings = main_meta["num_partitions"].as_i64().unwrap();

    let mut final_num_documents: usize = 0;

    let ids_to_delete_set: HashSet<i64> = subset.iter().cloned().collect();
    let mut current_doc_offset = 0;
    let mut num_embeddings = 0;

    for chunk_idx in 0..num_chunks {
        let doclens_path = idx_path_obj.join(format!("doclens.{}.json", chunk_idx));
        let doclens_file = File::open(&doclens_path)?;
        let doclens: Vec<i64> = serde_json::from_reader(BufReader::new(doclens_file))?;

        let mut new_doclens = Vec::new();
        let mut embs_to_keep_mask = Vec::new();

        for (i, &len) in doclens.iter().enumerate() {
            let doc_id = current_doc_offset + i as i64;
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

        final_num_documents += new_doclens.len();

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
            new_residuals
                .reshape(&new_residuals_shape)
                .write_npy(&residuals_path)?;

            // Update metadata
            let chunk_meta_path = idx_path_obj.join(format!("{}.metadata.json", chunk_idx));
            let chunk_meta_file = File::open(&chunk_meta_path)?;
            let mut chunk_meta: serde_json::Value =
                serde_json::from_reader(BufReader::new(chunk_meta_file))?;
            chunk_meta["num_documents"] = serde_json::json!(new_doclens.len());
            chunk_meta["num_embeddings"] = serde_json::json!(new_codes.size()[0]);
            let new_chunk_meta_file = File::create(&chunk_meta_path)?;
            serde_json::to_writer_pretty(BufWriter::new(new_chunk_meta_file), &chunk_meta)?;
        }
        num_embeddings += new_doclens.iter().sum::<i64>();
        current_doc_offset += doclens.len() as i64;
    }

    // Recreate IVF
    let all_codes = Tensor::zeros(&[num_embeddings], (Kind::Int64, device));
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
    let code_counts = sorted_codes.bincount::<Tensor>(None, est_num_embeddings);
    let (opt_ivf, opt_ivf_lens) = optimize_ivf(&sorted_indices, &code_counts, idx_path, device)?;

    opt_ivf.write_npy(&idx_path_obj.join("ivf.npy"))?;
    opt_ivf_lens.write_npy(&idx_path_obj.join("ivf_lengths.npy"))?;

    // Update main metadata
    let final_avg_doclen = if final_num_documents > 0 {
        num_embeddings as f64 / final_num_documents as f64
    } else {
        0.0
    };

    let final_meta_json = json!({
        "num_chunks": num_chunks,
        "nbits": nbits,
        "num_partitions": est_num_embeddings,
        "num_embeddings": num_embeddings,
        "avg_doclen": final_avg_doclen,
        "num_documents": final_num_documents,
    });

    let final_meta_file = fs::File::create(&main_meta_path)?;
    serde_json::to_writer_pretty(BufWriter::new(final_meta_file), &final_meta_json)?;

    Ok(())
}
