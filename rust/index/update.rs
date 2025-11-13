use anyhow::{anyhow, Context, Result};
use serde_json;
use serde_json::json;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::index::create::{compress_into_codes, optimize_ivf, packbits, Metadata};
use crate::search::load::LoadedIndex;
use crate::search::tensor::StridedTensor;
use crate::utils::residual_codec::ResidualCodec;

const DEFAULT_PROC_CHUNK_SIZE: usize = 25_000;

/// Updates an existing compressed index with a new collection of document embeddings.
///
/// This function loads the configuration and codec from an existing index,
/// processes the new documents into new chunks, and then rebuilds the
/// Inverted File (IVF) by merging the codes from old and new chunks.
/// This avoids retraining the codec and re-quantizing existing documents.
///
/// # Arguments
///
/// * `documents_embeddings` - A vector of tensors, where each tensor represents the embeddings for a single new document to be added.
/// * `idx_path` - The directory path where the existing index is located and will be updated.
/// * `device` - The `tch::Device` (e.g., CPU or CUDA) on which to perform computations.
///
/// # Returns
///
/// A `Result` indicating success or failure. On success, the index in `idx_path`
/// will be updated to include the new documents.
pub fn update_index(
    documents_embeddings: &Vec<Tensor>,
    idx_path: &str,
    device: Device,
    batch_size: i64,
    codec: &ResidualCodec,
) -> Result<LoadedIndex> {
    let _grad_guard = tch::no_grad_guard();
    let idx_path_obj = Path::new(idx_path);

    // 1. Load Initial Metadata
    let main_meta_path = idx_path_obj.join("metadata.json");
    let main_meta_file = File::open(&main_meta_path)
        .with_context(|| format!("Failed to open main metadata file: {:?}", main_meta_path))?;
    let main_meta: serde_json::Value = serde_json::from_reader(BufReader::new(main_meta_file))
        .context("Failed to parse main metadata JSON")?;

    let nbits = main_meta["nbits"]
        .as_i64()
        .context("Missing 'nbits' in metadata")?;
    let start_chunk_idx = main_meta["num_chunks"]
        .as_u64()
        .context("Missing 'num_chunks' in metadata")? as usize;
    let est_total_embs = main_meta["num_partitions"]
        .as_i64()
        .context("Missing 'num_partitions' in metadata")?;
    let embedding_dim = codec.centroids.size()[1];

    let b_cutoffs = codec
        .bucket_cutoffs
        .as_ref()
        .ok_or_else(|| anyhow!("Codec missing bucket_cutoffs"))?;

    // 2. Process New Documents (Chunking & Quantization)
    let n_new_docs = documents_embeddings.len();
    let proc_chunk_sz = DEFAULT_PROC_CHUNK_SIZE.min(1 + n_new_docs);
    let n_new_chunks = (n_new_docs as f64 / proc_chunk_sz as f64).ceil() as usize;

    for i in 0..n_new_chunks {
        let chk_idx = start_chunk_idx + i;
        let chk_offset = i * proc_chunk_sz;
        let chk_end_offset = (chk_offset + proc_chunk_sz).min(n_new_docs);

        let chk_embs_vec: Vec<Tensor> = documents_embeddings[chk_offset..chk_end_offset]
            .iter()
            .map(|t| t.shallow_clone())
            .collect();
        let chk_doclens: Vec<i64> = chk_embs_vec.iter().map(|e| e.size()[0]).collect();
        let chk_embs_tensor = Tensor::cat(&chk_embs_vec, 0)
            .to_kind(Kind::Half)
            .to_device(device);

        let mut chk_codes_list: Vec<Tensor> = Vec::new();
        let mut chk_res_list: Vec<Tensor> = Vec::new();

        for emb_batch in chk_embs_tensor.split(batch_size, 0) {
            let code_batch = compress_into_codes(&emb_batch, &codec.centroids, batch_size);
            chk_codes_list.push(code_batch.shallow_clone());

            let mut recon_centroids_batches: Vec<Tensor> = Vec::new();
            for sub_code_batch in code_batch.split(batch_size, 0) {
                recon_centroids_batches.push(codec.centroids.index_select(0, &sub_code_batch));
            }
            let recon_centroids = Tensor::cat(&recon_centroids_batches, 0);

            let mut res_batch = &emb_batch - &recon_centroids;
            res_batch = Tensor::bucketize(&res_batch, b_cutoffs, true, false);

            let mut res_shape = res_batch.size();
            res_shape.push(nbits);
            res_batch = res_batch.unsqueeze(-1).expand(&res_shape, false);
            res_batch = res_batch.bitwise_right_shift(&codec.bit_helper);
            let ones = Tensor::ones_like(&res_batch).to_device(device);
            res_batch = res_batch.bitwise_and_tensor(&ones);

            let res_flat = res_batch.flatten(0, -1);
            let res_packed = packbits(&res_flat);

            let shape = [res_batch.size()[0], embedding_dim / 8 * nbits];
            chk_res_list.push(res_packed.reshape(&shape));
        }

        let chk_codes = Tensor::cat(&chk_codes_list, 0);
        let chk_residuals = Tensor::cat(&chk_res_list, 0);

        chk_codes
            .to_device(Device::Cpu)
            .write_npy(&idx_path_obj.join(&format!("{}.codes.npy", chk_idx)))?;
        chk_residuals
            .to_device(Device::Cpu)
            .write_npy(&idx_path_obj.join(&format!("{}.residuals.npy", chk_idx)))?;

        let dl_file = File::create(idx_path_obj.join(format!("doclens.{}.json", chk_idx)))?;
        serde_json::to_writer(BufWriter::new(dl_file), &chk_doclens)?;

        let chk_meta = Metadata {
            num_passages: chk_doclens.len(),
            num_embeddings: chk_codes.size()[0] as usize,
        };
        let meta_f_w = File::create(idx_path_obj.join(format!("{}.metadata.json", chk_idx)))?;
        serde_json::to_writer(BufWriter::new(meta_f_w), &chk_meta)?;
    }

    // 3. Update Global Metadata & Collect Offsets
    let new_total_chunks = start_chunk_idx + n_new_chunks;
    let mut current_emb_offset = 0;
    let mut chk_emb_offsets: Vec<usize> = Vec::with_capacity(new_total_chunks);
    let mut all_doc_lens_vec: Vec<i64> = Vec::new();

    for chk_idx in 0..new_total_chunks {
        let chk_meta_fpath = idx_path_obj.join(format!("{}.metadata.json", chk_idx));
        let meta_f_r = File::open(&chk_meta_fpath)?;
        let mut json_val: serde_json::Value = serde_json::from_reader(BufReader::new(meta_f_r))?;

        if let Some(meta_obj) = json_val.as_object_mut() {
            meta_obj.insert("embedding_offset".to_string(), json!(current_emb_offset));
            chk_emb_offsets.push(current_emb_offset);

            let embs_in_chk = meta_obj["num_embeddings"].as_u64().unwrap() as usize;
            current_emb_offset += embs_in_chk;

            let meta_f_w_updated = File::create(&chk_meta_fpath)?;
            serde_json::to_writer_pretty(BufWriter::new(meta_f_w_updated), &json_val)?;
        }

        let doclens_path = idx_path_obj.join(format!("doclens.{}.json", chk_idx));
        let file = File::open(&doclens_path)?;
        let doclens: Vec<i64> = serde_json::from_reader(BufReader::new(file))?;
        all_doc_lens_vec.extend(doclens);
    }

    let total_num_embs = current_emb_offset;

    // 4. Aggregate Codes and Residuals into Global Tensors
    let all_codes = Tensor::zeros(&[total_num_embs as i64], (Kind::Int64, device));

    let residual_dim = embedding_dim / 8 * nbits;
    let all_residuals = Tensor::zeros(
        &[total_num_embs as i64, residual_dim],
        (Kind::Uint8, device),
    );

    for chk_idx in 0..new_total_chunks {
        let chk_offset_global = chk_emb_offsets[chk_idx];

        let codes_fpath = idx_path_obj.join(format!("{}.codes.npy", chk_idx));
        let codes_from_file = Tensor::read_npy(&codes_fpath)?.to_device(device);
        let count = codes_from_file.size()[0];

        all_codes
            .narrow(0, chk_offset_global as i64, count)
            .copy_(&codes_from_file);

        let res_fpath = idx_path_obj.join(format!("{}.residuals.npy", chk_idx));
        let res_from_file = Tensor::read_npy(&res_fpath)?.to_device(device);

        all_residuals
            .narrow(0, chk_offset_global as i64, count)
            .copy_(&res_from_file);
    }

    // 5. Optimize IVF
    let (sorted_codes, sorted_indices) = all_codes.sort(0, false);
    let code_counts = sorted_codes.bincount::<Tensor>(None, est_total_embs);

    let (opt_ivf, opt_ivf_lens) = optimize_ivf(&sorted_indices, &code_counts, idx_path, device)
        .context("Failed to optimize IVF during update")?;

    opt_ivf
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64)
        .write_npy(&idx_path_obj.join("ivf.npy"))?;
    opt_ivf_lens
        .to_device(Device::Cpu)
        .write_npy(&idx_path_obj.join("ivf_lengths.npy"))?;

    let final_avg_doclen = if !all_doc_lens_vec.is_empty() {
        total_num_embs as f64 / all_doc_lens_vec.len() as f64
    } else {
        0.0
    };

    let final_meta_json = json!({
        "num_chunks": new_total_chunks,
        "nbits": nbits,
        "num_partitions": est_total_embs,
        "num_embeddings": total_num_embs,
        "avg_doclen": final_avg_doclen,
    });
    let final_meta_file = fs::File::create(&main_meta_path)?;
    serde_json::to_writer_pretty(BufWriter::new(final_meta_file), &final_meta_json)?;

    let all_doc_lengths_tensor = Tensor::from_slice(&all_doc_lens_vec)
        .to_kind(Kind::Int64)
        .to_device(device);

    let doc_codes_strided =
        StridedTensor::new(all_codes, all_doc_lengths_tensor.shallow_clone(), device);

    let doc_residuals_strided = StridedTensor::new(all_residuals, all_doc_lengths_tensor, device);

    let ivf_index_strided = StridedTensor::new(opt_ivf, opt_ivf_lens, device);

    drop(sorted_codes);
    drop(sorted_indices);

    Ok(LoadedIndex {
        codec: codec.clone(),
        ivf_index_strided,
        doc_codes_strided,
        doc_residuals_strided,
        nbits: nbits,
    })
}
