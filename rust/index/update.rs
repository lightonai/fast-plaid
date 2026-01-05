use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressIterator};
use pyo3_tch::PyTensor;
use serde_json;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::index::create::{compress_into_codes, packbits, Metadata};
use crate::search::load::LoadedIndex;
use crate::search::tensor::scalar_quantile_kthvalue;

/// The default batch size for processing chunks of documents (I/O Buffering).
const DEFAULT_PROC_CHUNK_SIZE: usize = 25_000;

/// Updates an existing FastPlaid index with new documents.
///
/// # Arguments
///
/// * `documents_embeddings` - List of tensors, one per document.
/// * `idx_path` - Directory where index files are stored.
/// * `device` - Computation device (CPU/GPU).
/// * `batch_size` - Size for micro-batching operations.
/// * `index` - The loaded index structure containing centroids/codecs.
/// * `update_threshold` - Whether to update the residual quantile threshold.
pub fn update_index(
    documents_embeddings: &Vec<PyTensor>,
    idx_path: &str,
    device: Device,
    batch_size: i64,
    index: &LoadedIndex,
    update_threshold: bool,
) -> Result<()> {
    let _grad_guard = tch::no_grad_guard();
    let idx_path_obj = Path::new(idx_path);

    // Load global metadata
    let main_meta_path = idx_path_obj.join("metadata.json");
    let main_meta_file = File::open(&main_meta_path)
        .with_context(|| format!("Failed to open main metadata file: {:?}", main_meta_path))?;
    let main_meta: serde_json::Value = serde_json::from_reader(BufReader::new(main_meta_file))
        .context("Failed to parse main metadata JSON")?;

    let num_existing_chunks = main_meta["num_chunks"]
        .as_u64()
        .context("Missing 'num_chunks' in metadata")? as usize;

    let old_num_documents = main_meta
        .get("num_documents")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| index.doc_codes_strided.element_lengths.size()[0]);

    let est_num_embeddings = main_meta["num_partitions"]
        .as_i64()
        .context("Missing 'num_partitions' in metadata")?;

    let old_total_embeddings_count = main_meta["num_embeddings"].as_i64().unwrap_or(0);

    let embedding_dim = index.codec.centroids.size()[1];
    let num_centroids = index.codec.centroids.size()[0];
    let nbits = index.nbits;
    let b_cutoffs = index
        .codec
        .bucket_cutoffs
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Codec missing bucket_cutoffs"))?;

    // Determine start chunk for appending
    let mut start_chunk_idx = num_existing_chunks;
    let mut append_to_last = false;
    let mut current_emb_offset = old_total_embeddings_count as usize;

    if start_chunk_idx > 0 {
        let last_idx = start_chunk_idx - 1;
        let last_meta_path = idx_path_obj.join(format!("{}.metadata.json", last_idx));

        if last_meta_path.exists() {
            let f = File::open(&last_meta_path)?;
            let last_meta_json: serde_json::Value = serde_json::from_reader(BufReader::new(f))?;

            if let Some(nd) = last_meta_json.get("num_documents").and_then(|x| x.as_u64()) {
                if nd < 2000 {
                    start_chunk_idx = last_idx;
                    append_to_last = true;
                    if let Some(off) = last_meta_json
                        .get("embedding_offset")
                        .and_then(|x| x.as_u64())
                    {
                        current_emb_offset = off as usize;
                    } else {
                        let embs_in_last = last_meta_json
                            .get("num_embeddings")
                            .and_then(|x| x.as_u64())
                            .unwrap_or(0);
                        current_emb_offset =
                            (old_total_embeddings_count as u64 - embs_in_last) as usize;
                    }
                }
            }
        }
    }

    // Process new documents
    let num_new_documents = documents_embeddings.len();
    let proc_chunk_sz = DEFAULT_PROC_CHUNK_SIZE.min(1 + num_new_documents);
    let n_new_chunks = (num_new_documents as f64 / proc_chunk_sz as f64).ceil() as usize;

    let mut new_codes_accumulated: Vec<Tensor> = Vec::new();
    let mut new_doclens_accumulated: Vec<i64> = Vec::new();
    let mut all_residual_norms: Vec<Tensor> = Vec::new();

    let bar = ProgressBar::new(n_new_chunks.try_into().unwrap());
    bar.set_message("Updating index...");

    let process_batch =
        |batch_tensor: Tensor| -> Result<(Tensor, Tensor, Option<Tensor>)> {
            let mut codes_list = Vec::new();
            let mut packed_list = Vec::new();
            let mut norms_list = Vec::new();

            let safe_batch_size = if device == Device::Cpu {
                batch_size
            } else {
                let target_bytes: i64 = 4 * 1024 * 1024 * 1024;
                (target_bytes / (num_centroids * 2)).min(batch_size)
            };

            let split_embs = batch_tensor.split(safe_batch_size, 0);

            for micro_batch in split_embs.into_iter() {
                let codes = compress_into_codes(&micro_batch, &index.codec.centroids);

                let reconstructed = index.codec.centroids.index_select(0, &codes);
                let mut res = &micro_batch - &reconstructed;

                if update_threshold {
                    let n = res.to_kind(Kind::Float).norm_scalaropt_dim(2, &[1], false);
                    norms_list.push(n.to_device(Device::Cpu));
                }

                res = Tensor::bucketize(&res, b_cutoffs, true, false);
                let mut res_shape = res.size();
                res_shape.push(nbits);
                res = res.unsqueeze(-1).expand(&res_shape, false);
                res = res.bitwise_right_shift(&index.codec.bit_helper);
                let ones = Tensor::ones_like(&res).to_device(device);
                res = res.bitwise_and_tensor(&ones);

                let res_flat = res.flatten(0, -1);
                let packed = packbits(&res_flat);
                let shape = [res.size()[0], embedding_dim / 8 * nbits];

                codes_list.push(codes.to_device(Device::Cpu));
                packed_list.push(packed.reshape(&shape).to_device(Device::Cpu));

                drop(reconstructed);
            }

            let final_codes = Tensor::cat(&codes_list, 0);
            let final_packed = Tensor::cat(&packed_list, 0);
            let final_norms = if update_threshold && !norms_list.is_empty() {
                Some(Tensor::cat(&norms_list, 0))
            } else {
                None
            };

            Ok((final_codes, final_packed, final_norms))
        };

    for i in (0..n_new_chunks).progress_with(bar) {
        let global_chunk_idx = start_chunk_idx + i;
        let chk_offset = i * proc_chunk_sz;
        let chk_end_offset = (chk_offset + proc_chunk_sz).min(num_new_documents);

        let mut chk_codes_list: Vec<Tensor> = Vec::new();
        let mut chk_res_list: Vec<Tensor> = Vec::new();
        let mut chk_doclens: Vec<i64> = Vec::new();

        let mut batch_acc: Vec<Tensor> = Vec::new();
        let mut current_rows: i64 = 0;

        for doc_tensor in documents_embeddings[chk_offset..chk_end_offset].iter() {
            let doc_len = doc_tensor.size()[0];
            chk_doclens.push(doc_len);

            let t = if doc_tensor.kind() == Kind::Half {
                doc_tensor.shallow_clone()
            } else {
                doc_tensor.to_kind(Kind::Half)
            };

            batch_acc.push(t);
            current_rows += doc_len;

            if current_rows >= batch_size {
                let big_batch = Tensor::cat(&batch_acc, 0).to_device(device);

                batch_acc.clear();
                current_rows = 0;

                let (c, r, n) = process_batch(big_batch)?;
                chk_codes_list.push(c);
                chk_res_list.push(r);
                if let Some(norms) = n {
                    all_residual_norms.push(norms);
                }
            }
        }

        if !batch_acc.is_empty() {
            let big_batch = Tensor::cat(&batch_acc, 0).to_device(device);
            batch_acc.clear();

            let (c, r, n) = process_batch(big_batch)?;
            chk_codes_list.push(c);
            chk_res_list.push(r);
            if let Some(norms) = n {
                all_residual_norms.push(norms);
            }
        }

        let mut chk_codes = Tensor::cat(&chk_codes_list, 0);
        let mut chk_residuals = Tensor::cat(&chk_res_list, 0);

        new_codes_accumulated.push(chk_codes.shallow_clone());
        new_doclens_accumulated.extend(&chk_doclens);

        if i == 0 && append_to_last {
            let old_codes_path = idx_path_obj.join(&format!("{}.codes.npy", global_chunk_idx));
            let old_res_path = idx_path_obj.join(&format!("{}.residuals.npy", global_chunk_idx));
            let old_dl_path = idx_path_obj.join(format!("doclens.{}.json", global_chunk_idx));

            if old_codes_path.exists() {
                let old_codes = Tensor::read_npy(&old_codes_path)?.to_device(Device::Cpu);
                let old_residuals = Tensor::read_npy(&old_res_path)?.to_device(Device::Cpu);
                let old_dl_file = File::open(&old_dl_path)?;
                let old_doclens: Vec<i64> = serde_json::from_reader(BufReader::new(old_dl_file))?;

                chk_codes = Tensor::cat(&[old_codes, chk_codes], 0);
                chk_residuals = Tensor::cat(&[old_residuals, chk_residuals], 0);
                let mut combined = old_doclens;
                combined.extend(chk_doclens);
                chk_doclens = combined;
            }
        }

        chk_codes.write_npy(&idx_path_obj.join(&format!("{}.codes.npy", global_chunk_idx)))?;
        chk_residuals
            .write_npy(&idx_path_obj.join(&format!("{}.residuals.npy", global_chunk_idx)))?;

        let dl_file =
            File::create(idx_path_obj.join(format!("doclens.{}.json", global_chunk_idx)))?;
        serde_json::to_writer(BufWriter::new(dl_file), &chk_doclens)?;

        let chk_meta = Metadata {
            num_documents: chk_doclens.len(),
            num_embeddings: chk_codes.size()[0] as usize,
        };
        let meta_f_w =
            File::create(idx_path_obj.join(format!("{}.metadata.json", global_chunk_idx)))?;
        serde_json::to_writer(BufWriter::new(meta_f_w), &chk_meta)?;

        drop(chk_codes);
        drop(chk_residuals);
        drop(chk_codes_list);
        drop(chk_res_list);
    }

    // Update quantile threshold
    if update_threshold && !all_residual_norms.is_empty() {
        let new_norms = Tensor::cat(&all_residual_norms, 0).to_device(Device::Cpu);
        let new_count = new_norms.size()[0];
        let new_threshold_tensor = scalar_quantile_kthvalue(&new_norms, 0.75);
        let new_threshold_val: f64 = f64::try_from(&new_threshold_tensor).unwrap_or(0.0);

        let thresh_fpath = idx_path_obj.join("cluster_threshold.npy");
        let final_threshold_val = if thresh_fpath.exists() {
            let old_threshold_tensor = Tensor::read_npy(&thresh_fpath)?.to_device(Device::Cpu);
            let old_threshold_val: f64 = f64::try_from(&old_threshold_tensor).unwrap_or(0.0);
            let total_count = old_total_embeddings_count + new_count;
            ((old_threshold_val * old_total_embeddings_count as f64)
                + (new_threshold_val * new_count as f64))
                / total_count as f64
        } else {
            new_threshold_val
        };
        Tensor::from(final_threshold_val)
            .to_device(Device::Cpu)
            .write_npy(&thresh_fpath)?;
        drop(new_norms);
    }
    drop(all_residual_norms);

    // Update global metadata and offsets
    let new_total_chunks = start_chunk_idx + n_new_chunks;
    for chk_idx in start_chunk_idx..new_total_chunks {
        let chk_meta_fpath = idx_path_obj.join(format!("{}.metadata.json", chk_idx));
        let meta_f_r = File::open(&chk_meta_fpath)?;
        let mut json_val: serde_json::Value = serde_json::from_reader(BufReader::new(meta_f_r))?;
        if let Some(meta_obj) = json_val.as_object_mut() {
            meta_obj.insert("embedding_offset".to_string(), json!(current_emb_offset));
            let embs_in_chk = meta_obj["num_embeddings"].as_u64().unwrap() as usize;
            current_emb_offset += embs_in_chk;
            let meta_f_w_updated = File::create(&chk_meta_fpath)?;
            serde_json::to_writer_pretty(BufWriter::new(meta_f_w_updated), &json_val)?;
        }
    }

    // Generate new partial IVF
    let new_codes_flat = Tensor::cat(&new_codes_accumulated, 0).to_device(Device::Cpu);
    drop(new_codes_accumulated);
    let new_codes_vec: Vec<i64> = Vec::<i64>::try_from(&new_codes_flat)?;

    let mut partition_pids_map: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut pid_counter = old_num_documents;
    let mut code_idx = 0;
    for &doc_len in &new_doclens_accumulated {
        for _ in 0..doc_len {
            let code = new_codes_vec[code_idx];
            partition_pids_map
                .entry(code)
                .or_insert_with(Vec::new)
                .push(pid_counter);
            code_idx += 1;
        }
        pid_counter += 1;
    }

    let mut new_partition_data: HashMap<usize, Tensor> = HashMap::new();
    for (code, mut pids) in partition_pids_map.into_iter() {
        if code >= 0 && code < est_num_embeddings {
            pids.sort_unstable();
            pids.dedup();
            let unique_pids = Tensor::from_slice(&pids).to_device(Device::Cpu);
            new_partition_data.insert(code as usize, unique_pids);
        }
    }

    // Merge with old IVF
    let old_ivf_flat = &index.ivf_index_strided.underlying_data;
    let old_ivf_lengths = &index.ivf_index_strided.element_lengths;
    let old_ivf_lengths_cpu = old_ivf_lengths.to_device(Device::Cpu);
    let old_lengths_vec: Vec<i64> = Vec::<i64>::try_from(&old_ivf_lengths_cpu)?;
    let num_partitions = est_num_embeddings as usize;

    if new_partition_data.is_empty() {
        return Ok(());
    }

    let mut old_ivf_offsets = Vec::with_capacity(old_lengths_vec.len());
    let mut curr = 0;
    for &l in &old_lengths_vec {
        old_ivf_offsets.push(curr);
        curr += l;
    }

    let mut final_ivf_parts: Vec<Tensor> = Vec::new();
    let mut final_lengths_vec: Vec<i64> = Vec::with_capacity(num_partitions);

    let mut i = 0;
    while i < num_partitions {
        if let Some(new_pids) = new_partition_data.get(&i) {
            let old_len = if i < old_lengths_vec.len() {
                old_lengths_vec[i]
            } else {
                0
            };
            let new_len = new_pids.size()[0];
            if old_len > 0 {
                let start = old_ivf_offsets[i];
                final_ivf_parts.push(
                    old_ivf_flat
                        .narrow(0, start, old_len)
                        .to_device(Device::Cpu),
                );
            }
            final_ivf_parts.push(new_pids.shallow_clone());
            final_lengths_vec.push(old_len + new_len);
            i += 1;
        } else {
            let range_start = i;
            while i < num_partitions && !new_partition_data.contains_key(&i) {
                i += 1;
            }
            let range_end = i;

            let flat_start = if range_start < old_ivf_offsets.len() {
                old_ivf_offsets[range_start]
            } else {
                0
            };
            let flat_end = if range_end < old_ivf_offsets.len() {
                old_ivf_offsets[range_end]
            } else if range_end > 0 && range_end - 1 < old_lengths_vec.len() {
                old_ivf_offsets[range_end - 1] + old_lengths_vec[range_end - 1]
            } else {
                flat_start
            };

            let flat_length = flat_end - flat_start;
            if flat_length > 0 {
                final_ivf_parts.push(
                    old_ivf_flat
                        .narrow(0, flat_start, flat_length)
                        .to_device(Device::Cpu),
                );
            }

            for j in range_start..range_end {
                let old_len = if j < old_lengths_vec.len() {
                    old_lengths_vec[j]
                } else {
                    0
                };
                final_lengths_vec.push(old_len);
            }
        }
    }

    let final_ivf_tensor = Tensor::cat(&final_ivf_parts, 0);
    let final_lengths_tensor = Tensor::from_slice(&final_lengths_vec)
        .to_device(Device::Cpu)
        .to_kind(Kind::Int);

    // Write updated global files
    final_ivf_tensor
        .to_kind(Kind::Int64)
        .write_npy(&idx_path_obj.join("ivf.npy"))?;
    final_lengths_tensor.write_npy(&idx_path_obj.join("ivf_lengths.npy"))?;

    // Update global metadata
    let new_tokens_count: i64 = new_doclens_accumulated.iter().sum();
    let num_embeddings = old_total_embeddings_count + new_tokens_count;
    let total_num_documents = old_num_documents as usize + num_new_documents;

    let final_avg_doclen = if total_num_documents > 0 {
        let old_avg = main_meta["avg_doclen"].as_f64().unwrap_or(0.0);
        let old_sum = old_avg * (old_num_documents as f64);
        (old_sum + new_tokens_count as f64) / (total_num_documents as f64)
    } else {
        0.0
    };

    let final_meta_json = json!({
        "num_chunks": new_total_chunks,
        "nbits": nbits,
        "num_partitions": est_num_embeddings,
        "num_embeddings": num_embeddings,
        "num_documents": total_num_documents,
        "avg_doclen": final_avg_doclen,
    });
    let final_meta_file = fs::File::create(&main_meta_path)?;
    serde_json::to_writer_pretty(BufWriter::new(final_meta_file), &final_meta_json)?;

    Ok(())
}
