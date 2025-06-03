use anyhow::{anyhow, Context, Result};
use rand::prelude::SliceRandom;
use regex::Regex;
use serde::Serialize;
use serde_json;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::utils::residual_codec::ResidualCodec;

/// Holds metadata for a chunk of the index, including the number of
/// passages and the total number of embeddings.
#[derive(Serialize)]
struct Metadata {
    num_passages: usize,
    num_embeddings: usize,
}

/// Optimizes an Inverted File (IVF) index by removing duplicate passage IDs (PIDs)
/// from each inverted list.
///
/// This function maps each embedding in the IVF to its original passage ID and then,
/// for each list in the IVF, retains only the unique PIDs. This is useful for retrieval
/// tasks where scoring each passage once per query is sufficient.
///
/// # Arguments
///
/// * `ivf` - A 1D tensor containing the indices of embeddings, forming the concatenated inverted lists.
/// * `ivf_lens` - A 1D tensor where each element specifies the length of the corresponding inverted list in `ivf`.
/// * `idx_path` - The path to the directory containing the index files, specifically the `doclens.*.json` files.
/// * `device` - The `tch::Device` on which to perform tensor operations.
///
/// # Returns
///
/// A `Result` containing a tuple of two tensors:
/// * The new, optimized IVF tensor with unique PIDs per list.
/// * A tensor with the new lengths of each optimized inverted list.
pub fn optimize_ivf(
    ivf: &Tensor,
    ivf_lens: &Tensor,
    idx_path: &str,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    let mut doclen_files: BTreeMap<i64, String> = BTreeMap::new();
    let doclen_re =
        Regex::new(r"doclens\.(\d+)\.json").context("Failed to compile regex for doclens files")?;

    for dir_entry_res in
        fs::read_dir(idx_path).with_context(|| format!("Failed to read directory: {}", idx_path))?
    {
        let dir_entry =
            dir_entry_res.with_context(|| format!("Failed to read entry in {}", idx_path))?;
        let fname = dir_entry.file_name();
        if let Some(fname_str) = fname.to_str() {
            if let Some(caps) = doclen_re.captures(fname_str) {
                if let Some(id_cap) = caps.get(1) {
                    let id = id_cap
                        .as_str()
                        .parse::<i64>()
                        .with_context(|| format!("Failed to parse chunk ID from {}", fname_str))?;
                    doclen_files.insert(id, dir_entry.path().to_str().unwrap().to_string());
                }
            }
        }
    }

    let mut all_doclens: Vec<i64> = Vec::new();
    for (_id, fpath) in doclen_files {
        let file = fs::File::open(&fpath)
            .with_context(|| format!("Failed to open doclens file: {}", fpath))?;
        let reader = BufReader::new(file);
        let chunk_doclens: Vec<i64> = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse JSON from {}", fpath))?;
        all_doclens.extend(chunk_doclens);
    }

    let total_embs: i64 = all_doclens.iter().sum();

    let mut emb_to_pid_vec: Vec<i64> = Vec::with_capacity(total_embs as usize);
    let mut pid_counter: i64 = 0;
    for &doc_len in &all_doclens {
        for _ in 0..doc_len {
            emb_to_pid_vec.push(pid_counter);
        }
        pid_counter += 1;
    }

    let emb_to_pid = Tensor::from_slice(&emb_to_pid_vec)
        .to_device(device)
        .to_kind(Kind::Int64);

    let pids_in_ivf = emb_to_pid.index_select(0, ivf);
    let mut unique_pids_list: Vec<Tensor> = Vec::new();
    let mut new_ivf_lens_vec: Vec<i64> = Vec::new();
    let ivf_lens_vec: Vec<i64> = Vec::<i64>::try_from(ivf_lens)?;
    let mut ivf_offset: i64 = 0;

    for &len in &ivf_lens_vec {
        let pids_seg = pids_in_ivf.narrow(0, ivf_offset, len);
        let (unique_pids, _, _) = pids_seg.unique_dim(0, true, false, false);
        unique_pids_list.push(unique_pids.copy());
        new_ivf_lens_vec.push(unique_pids.size1().unwrap_or(0));
        ivf_offset += len;
    }

    let pids_in_ivf = Tensor::cat(&unique_pids_list, 0);
    let new_ivf_lens = Tensor::from_slice(&new_ivf_lens_vec)
        .to_device(device)
        .to_kind(Kind::Int64);

    Ok((pids_in_ivf, new_ivf_lens))
}

/// Compresses embeddings into codes by finding the nearest centroid.
///
/// This function performs vector quantization by computing the matrix multiplication
/// between centroids and embedding batches, then finding the index of the maximum
/// value (i.e., the closest centroid) for each embedding.
///
/// # Arguments
///
/// * `embs` - A tensor of embeddings to be compressed, with shape `[num_embeddings, dim]`.
/// * `centroids` - A tensor of centroids, with shape `[num_centroids, dim]`.
///
/// # Returns
///
/// A 1D tensor of codes (indices of the nearest centroids).
fn compress_into_codes(embs: &Tensor, centroids: &Tensor) -> Tensor {
    let mut codes = Vec::new();
    let batch_sz = (1 << 29) / centroids.size()[0] as i64;
    for mut emb_batch in embs.split(batch_sz, 0) {
        codes.push(centroids.matmul(&emb_batch.t_()).argmax(0, false));
    }
    Tensor::cat(&codes, 0)
}

/// Packs a tensor of bits (0s or 1s) into a tensor of `Uint8` bytes.
///
/// The function reshapes the input tensor into rows of 8 bits and computes
/// their byte representation using a weighted sum.
///
/// # Arguments
///
/// * `res` - A 1D tensor containing bit values (0s or 1s).
///
/// # Returns
///
/// A 1D tensor of `Uint8` bytes.
pub fn packbits(res: &Tensor) -> Tensor {
    let bits_mat = res.reshape(&[-1, 8]).to_kind(Kind::Half);
    let weights = Tensor::from_slice(&[128i64, 64, 32, 16, 8, 4, 2, 1])
        .to_device(res.device())
        .to_kind(Kind::Half);
    let packed = bits_mat.matmul(&weights).to_kind(Kind::Uint8);
    packed
}

/// Creates a compressed index from a collection of document embeddings.
///
/// This function orchestrates the end-to-end process of building a quantized
/// index. It trains a `ResidualCodec` on a sample of the embeddings,
/// then processes all embeddings in chunks to generate codes and quantized
/// residuals. Finally, it builds and optimizes an IVF index from the codes.
///
/// # Arguments
///
/// * `documents_embeddings` - A vector of tensors, where each tensor represents the embeddings for a single document.
/// * `idx_path` - The directory path where the generated index files will be stored.
/// * `embedding_dim` - The dimensionality of the embeddings.
/// * `nbits` - The number of bits to use for residual quantization.
/// * `device` - The `tch::Device` (e.g., CPU or CUDA) on which to perform computations.
/// * `centroids` - The initial centroids for the quantization codec.
///
/// # Returns
///
/// A `Result` indicating success or failure. On success, the `idx_path`
/// directory will contain all the necessary index files.
pub fn create_index(
    documents_embeddings: &Vec<Tensor>,
    idx_path: &str,
    embedding_dim: i64,
    nbits: i64,
    device: Device,
    centroids: Tensor,
) -> Result<()> {
    let _grad_guard = tch::no_grad_guard();

    let n_docs = documents_embeddings.len();
    let n_chunks = (n_docs as f64 / 25_000f64.min(1.0 + n_docs as f64)).ceil() as usize;

    let n_passages = documents_embeddings.len();

    let sample_k_float = 16.0 * (120.0 * n_passages as f64).sqrt();
    let k = (1.0 + sample_k_float).min(n_passages as f64) as usize;

    let mut rng = rand::rng();
    let mut passage_indices: Vec<u32> = (0..n_passages as u32).collect();
    passage_indices.shuffle(&mut rng);
    let sample_pids: Vec<u32> = passage_indices.into_iter().take(k).collect();

    let mut sample_tensors_vec: Vec<&Tensor> = Vec::with_capacity(k);
    let avg_doc_len = documents_embeddings
        .iter()
        .map(|t| t.size()[0] as f64)
        .sum::<f64>()
        / n_docs as f64;

    for &pid in &sample_pids {
        sample_tensors_vec.push(&documents_embeddings[pid as usize]);
    }

    let sample_embs = Tensor::cat(&sample_tensors_vec, 0)
        .to_kind(Kind::Half)
        .to_device(device);

    let mut est_total_embs_f64 = (n_passages as f64) * avg_doc_len;
    est_total_embs_f64 = (16.0 * est_total_embs_f64.sqrt()).log2().floor();
    let est_total_embs = 2f64.powf(est_total_embs_f64) as i64;

    let plan_fpath = Path::new(idx_path).join("plan.json");
    let plan_data = json!({ "nbits": nbits, "num_chunks": n_chunks });
    let mut plan_file = File::create(plan_fpath)?;
    writeln!(plan_file, "{}", serde_json::to_string_pretty(&plan_data)?)?;

    let total_samples = sample_embs.size()[0] as f64;
    let heldout_sz = (0.05 * total_samples).min(50_000f64).round() as i64;
    let sample_splits =
        sample_embs.split_with_sizes(&[total_samples as i64 - heldout_sz, heldout_sz], 0);

    let heldout_samples = sample_splits[1].shallow_clone();

    let initial_codec = ResidualCodec::load(
        nbits,
        centroids.copy(),
        Tensor::zeros(&[embedding_dim], (Kind::Half, device)),
        None,
        None,
        device,
    )?;

    let heldout_codes = compress_into_codes(&heldout_samples, &initial_codec.centroids);

    let mut recon_embs_vec = Vec::new();
    for code_batch_idxs in heldout_codes.split((1 << 20) as i64, 0) {
        recon_embs_vec.push(initial_codec.centroids.index_select(0, &code_batch_idxs));
    }
    let heldout_recon_embs = Tensor::cat(&recon_embs_vec, 0);

    let heldout_res_raw = (&heldout_samples - &heldout_recon_embs).to_kind(Kind::Float); // Here float on purpose
    let avg_res_per_dim = heldout_res_raw
        .abs()
        .mean_dim(Some(&[0i64][..]), false, Kind::Float) // Here float on purpose
        .to_device(device);

    let n_options = 2_i32.pow(nbits as u32);
    let quantiles_base =
        Tensor::arange_start(0, n_options.into(), (Kind::Float, device)) * (1.0 / n_options as f64); // Here float on purpose

    let cutoff_quantiles = quantiles_base.narrow(0, 1, n_options as i64 - 1);
    let weight_quantiles = &quantiles_base + (0.5 / n_options as f64);

    let b_cutoffs = heldout_res_raw.quantile(&cutoff_quantiles, None, false, "linear");
    let b_weights = heldout_res_raw.quantile(&weight_quantiles, None, false, "linear");

    let final_codec = ResidualCodec::load(
        nbits,
        initial_codec.centroids.copy(),
        avg_res_per_dim,
        Some(b_cutoffs.copy()),
        Some(b_weights.copy()),
        device,
    )?;

    let centroids_fpath = Path::new(idx_path).join("centroids.npy");
    final_codec
        .centroids
        .to_device(Device::Cpu)
        .write_npy(&centroids_fpath)?;

    let cutoffs_fpath = Path::new(idx_path).join("bucket_cutoffs.npy");
    b_cutoffs.to_device(Device::Cpu).write_npy(&cutoffs_fpath)?;

    let weights_fpath = Path::new(idx_path).join("bucket_weights.npy");
    b_weights.to_device(Device::Cpu).write_npy(&weights_fpath)?;

    let avg_res_fpath = Path::new(idx_path).join("avg_residual.npy");
    final_codec
        .avg_residual
        .to_device(Device::Cpu)
        .write_npy(&avg_res_fpath)?;

    let proc_chunk_sz = 25_000.min(1 + n_passages);

    for chk_idx in 0..n_chunks {
        let chk_offset = chk_idx * proc_chunk_sz;
        let chk_end_offset = (chk_offset + proc_chunk_sz).min(n_passages);

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

        for emb_batch in chk_embs_tensor.split(1 << 18, 0) {
            let code_batch = compress_into_codes(&emb_batch, &final_codec.centroids);
            chk_codes_list.push(code_batch.shallow_clone());

            let mut recon_centroids_batches: Vec<Tensor> = Vec::new();
            for sub_code_batch in code_batch.split(1 << 20, 0) {
                recon_centroids_batches
                    .push(final_codec.centroids.index_select(0, &sub_code_batch));
            }
            let recon_centroids = Tensor::cat(&recon_centroids_batches, 0);

            let mut res_batch = &emb_batch - &recon_centroids;
            res_batch = Tensor::bucketize(&res_batch, &b_cutoffs, true, false);

            let mut res_shape = res_batch.size();
            res_shape.push(nbits);
            res_batch = res_batch.unsqueeze(-1).expand(&res_shape, false);
            res_batch = res_batch.bitwise_right_shift(&final_codec.bit_helper);
            let ones = Tensor::ones_like(&res_batch).to_device(device);
            res_batch = res_batch.bitwise_and_tensor(&ones);

            let res_flat = res_batch.flatten(0, -1);

            let res_packed = packbits(&res_flat);

            let shape = [res_batch.size()[0], embedding_dim / 8 * nbits];
            chk_res_list.push(res_packed.reshape(&shape));
        }

        let chk_codes = Tensor::cat(&chk_codes_list, 0);
        let chk_residuals = Tensor::cat(&chk_res_list, 0);

        let chk_codes_fpath = Path::new(idx_path).join(&format!("{}.codes.npy", chk_idx));
        chk_codes
            .to_device(Device::Cpu)
            .write_npy(&chk_codes_fpath)?;

        let chk_res_fpath = Path::new(idx_path).join(&format!("{}.residuals.npy", chk_idx));
        chk_residuals
            .to_device(Device::Cpu)
            .write_npy(&chk_res_fpath)?;

        let chk_doclens_fpath = Path::new(idx_path).join(format!("doclens.{}.json", chk_idx));
        let dl_file = File::create(chk_doclens_fpath)?;
        let buf_writer = BufWriter::new(dl_file);
        serde_json::to_writer(buf_writer, &chk_doclens)?;

        let chk_meta = Metadata {
            num_passages: chk_doclens.len(),
            num_embeddings: chk_codes.size()[0] as usize,
        };
        let chk_meta_fpath = Path::new(idx_path).join(format!("{}.metadata.json", chk_idx));
        let meta_f_w = File::create(chk_meta_fpath)?;
        let buf_writer_meta = BufWriter::new(meta_f_w);
        serde_json::to_writer(buf_writer_meta, &chk_meta)?;
    }

    let mut current_emb_offset: usize = 0;
    let mut chk_emb_offsets: Vec<usize> = Vec::new();

    for chk_idx in 0..n_chunks {
        let chk_meta_fpath = Path::new(idx_path).join(format!("{}.metadata.json", chk_idx));
        let meta_f_r = File::open(&chk_meta_fpath)?;
        let buf_reader = BufReader::new(meta_f_r);
        let mut json_val: serde_json::Value = serde_json::from_reader(buf_reader)?;

        if let Some(meta_obj) = json_val.as_object_mut() {
            meta_obj.insert("embedding_offset".to_string(), json!(current_emb_offset));
            chk_emb_offsets.push(current_emb_offset);

            let embs_in_chk = meta_obj
                .get("num_embeddings")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    anyhow!(
                        "'num_embeddings' not found or invalid in {}",
                        chk_meta_fpath.display()
                    )
                })? as usize;
            current_emb_offset += embs_in_chk;

            let meta_f_w_updated = File::create(&chk_meta_fpath)?;
            let writer_updated = BufWriter::new(meta_f_w_updated);
            serde_json::to_writer_pretty(writer_updated, &json_val)?;
        } else {
            return Err(anyhow!(
                "Metadata in {} is not a JSON object",
                chk_meta_fpath.display()
            ));
        }
    }

    let total_num_embs = current_emb_offset;
    let all_codes = Tensor::zeros(&[total_num_embs as i64], (Kind::Int64, device));

    for chk_idx in 0..n_chunks {
        let chk_offset_global = chk_emb_offsets[chk_idx];
        let codes_fpath_for_global = Path::new(idx_path).join(format!("{}.codes.npy", chk_idx));
        let codes_from_file = Tensor::read_npy(&codes_fpath_for_global)?.to_device(device);
        let codes_in_chk_count = codes_from_file.size()[0];
        all_codes
            .narrow(0, chk_offset_global as i64, codes_in_chk_count)
            .copy_(&codes_from_file);
    }

    let (sorted_codes, sorted_indices) = all_codes.sort(0, false);
    let code_counts = sorted_codes.bincount::<Tensor>(None, est_total_embs);

    let (opt_ivf, opt_ivf_lens) = optimize_ivf(&sorted_indices, &code_counts, idx_path, device)
        .context("Failed to optimize IVF")?;

    let opt_ivf_fpath = Path::new(idx_path).join("ivf.npy");
    opt_ivf
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64)
        .write_npy(&opt_ivf_fpath)?;
    let opt_ivf_lens_fpath = Path::new(idx_path).join("ivf_lengths.npy");
    opt_ivf_lens
        .to_device(Device::Cpu)
        .write_npy(&opt_ivf_lens_fpath)?;

    let final_meta_fpath = Path::new(idx_path).join("metadata.json");
    let final_num_docs = documents_embeddings.len();
    let final_avg_doclen = if final_num_docs > 0 {
        total_num_embs as f64 / final_num_docs as f64
    } else {
        0.0
    };

    let final_meta_json = json!({
        "num_chunks": n_chunks,
        "nbits": nbits,
        "num_partitions": est_total_embs,
        "num_embeddings": total_num_embs,
        "avg_doclen": final_avg_doclen,
    });
    let final_meta_file = fs::File::create(&final_meta_fpath)?;
    let final_writer = BufWriter::new(final_meta_file);
    serde_json::to_writer_pretty(final_writer, &final_meta_json)?;

    Ok(())
}
