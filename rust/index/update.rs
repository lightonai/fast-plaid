use anyhow::{Context, Result};
use serde_json;
use serde_json::json;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::index::create::{compress_into_codes, packbits, Metadata};
use crate::search::load::LoadedIndex;

/// The default batch size for processing chunks of documents to avoid OOM errors.
const DEFAULT_PROC_CHUNK_SIZE: usize = 25_000;

/// Updates an existing FastPlaid index with new documents.
///
/// This function implements an **incremental update** strategy:
/// 1. **Quantization:** New documents are compressed using the *existing* Codec (centroids and
///    bucket cutoffs) from the loaded index. This ensures compatibility without requiring
///    expensive re-clustering.
/// 2. **Persistence:** The new compressed codes and residuals are saved to disk as new
///    files (chunks), appended to the existing file sequence.
/// 3. **IVF Merging:** A new partial Inverted File (IVF) is generated for the new documents
///    and then merged with the existing global IVF to create a unified search structure.
///
/// # Arguments
///
/// * `documents_embeddings` - A vector of tensors, where each tensor represents the
///   multi-vector embeddings (shape: `[num_tokens, dim]`) for a single new document.
/// * `idx_path` - The file system path to the existing index directory. Files here will
///   be modified or added.
/// * `device` - The `tch::Device` (CPU or CUDA) used for tensor operations (quantization).
/// * `batch_size` - The internal batch size used during the vector quantization step.
/// * `index` - A reference to the currently loaded `LoadedIndex`. This is crucial as it
///   provides the existing centroids for quantization and the current IVF structure for merging.
///
/// # Returns
///
/// Returns `Ok(())` on success. Fails if metadata is missing, file I/O errors occur,
/// or if dimensions mismatch.
pub fn update_index(
    documents_embeddings: &Vec<Tensor>,
    idx_path: &str,
    device: Device,
    batch_size: i64,
    index: &LoadedIndex,
) -> Result<()> {
    // Disable gradient calculation to save memory during inference/indexing.
    let _grad_guard = tch::no_grad_guard();
    let idx_path_obj = Path::new(idx_path);

    // -------------------------------------------------------------------------
    // 1. Load Global Metadata
    // -------------------------------------------------------------------------
    // We need to know the current state of the index (how many chunks exist,
    // total number of embeddings) to assign correct offsets to the new data.
    let main_meta_path = idx_path_obj.join("metadata.json");
    let main_meta_file = File::open(&main_meta_path)
        .with_context(|| format!("Failed to open main metadata file: {:?}", main_meta_path))?;
    let main_meta: serde_json::Value = serde_json::from_reader(BufReader::new(main_meta_file))
        .context("Failed to parse main metadata JSON")?;

    let start_chunk_idx = main_meta["num_chunks"]
        .as_u64()
        .context("Missing 'num_chunks' in metadata")? as usize;

    // Determine the starting Passage ID (PID) for the new documents.
    // If the metadata doesn't explicitly store `num_passages`, we attempt to infer it
    // from the loaded index structure.
    let old_num_passages = main_meta
        .get("num_passages")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| {
            // Fallback: Use the length of the document codes length tensor if available.
            index.doc_codes_strided.element_lengths.size()[0]
        });

    let est_total_embs = main_meta["num_partitions"]
        .as_i64()
        .context("Missing 'num_partitions' in metadata")?;

    // Retrieve quantization parameters from the loaded index.
    let embedding_dim = index.codec.centroids.size()[1];
    let nbits = index.nbits;
    let b_cutoffs = index
        .codec
        .bucket_cutoffs
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Codec missing bucket_cutoffs"))?;

    // -------------------------------------------------------------------------
    // 2. Process New Documents (Chunking & Quantization)
    // -------------------------------------------------------------------------
    // To handle large updates without exhausting RAM/VRAM, we process documents in
    // manageable chunks. Each chunk is quantized and immediately written to disk.
    let n_new_docs = documents_embeddings.len();
    let proc_chunk_sz = DEFAULT_PROC_CHUNK_SIZE.min(1 + n_new_docs);
    let n_new_chunks = (n_new_docs as f64 / proc_chunk_sz as f64).ceil() as usize;

    let mut new_codes_accumulated: Vec<Tensor> = Vec::new();
    let mut new_doclens_accumulated: Vec<i64> = Vec::new();

    for i in 0..n_new_chunks {
        let chk_idx = start_chunk_idx + i;
        let chk_offset = i * proc_chunk_sz;
        let chk_end_offset = (chk_offset + proc_chunk_sz).min(n_new_docs);

        // Prepare batch tensors
        let chk_embs_vec: Vec<Tensor> = documents_embeddings[chk_offset..chk_end_offset]
            .iter()
            .map(|t| t.shallow_clone().to(device))
            .collect();
        let chk_doclens: Vec<i64> = chk_embs_vec.iter().map(|e| e.size()[0]).collect();
        new_doclens_accumulated.extend(&chk_doclens);

        let chk_embs_tensor = Tensor::cat(&chk_embs_vec, 0)
            .to_kind(Kind::Half)
            .to_device(device);

        // -- Quantization Loop --
        let mut chk_codes_list: Vec<Tensor> = Vec::new();
        let mut chk_res_list: Vec<Tensor> = Vec::new();

        for emb_batch in chk_embs_tensor.split(batch_size, 0) {
            // A. Assign Centroids (Codes)
            let code_batch = compress_into_codes(&emb_batch, &index.codec.centroids, batch_size);
            chk_codes_list.push(code_batch.shallow_clone());

            // B. Compute Residuals
            let mut recon_centroids_batches: Vec<Tensor> = Vec::new();
            for sub_code_batch in code_batch.split(batch_size, 0) {
                recon_centroids_batches
                    .push(index.codec.centroids.index_select(0, &sub_code_batch));
            }
            let recon_centroids = Tensor::cat(&recon_centroids_batches, 0);
            let mut res_batch = &emb_batch - &recon_centroids;

            // C. Quantize & Pack Residuals
            res_batch = Tensor::bucketize(&res_batch, b_cutoffs, true, false);
            let mut res_shape = res_batch.size();
            res_shape.push(nbits);

            // Expand bits for packing
            res_batch = res_batch.unsqueeze(-1).expand(&res_shape, false);
            res_batch = res_batch.bitwise_right_shift(&index.codec.bit_helper);
            let ones = Tensor::ones_like(&res_batch).to_device(device);
            res_batch = res_batch.bitwise_and_tensor(&ones);

            let res_flat = res_batch.flatten(0, -1);
            let res_packed = packbits(&res_flat);

            let shape = [res_batch.size()[0], embedding_dim / 8 * nbits];
            chk_res_list.push(res_packed.reshape(&shape));
        }

        let chk_codes = Tensor::cat(&chk_codes_list, 0);
        let chk_residuals = Tensor::cat(&chk_res_list, 0);

        // Cache codes in memory: we need them later to build the IVF index.
        new_codes_accumulated.push(chk_codes.shallow_clone());

        // Write this chunk's data to disk immediately.
        chk_codes
            .to_device(Device::Cpu)
            .write_npy(&idx_path_obj.join(&format!("{}.codes.npy", chk_idx)))?;
        chk_residuals
            .to_device(Device::Cpu)
            .write_npy(&idx_path_obj.join(&format!("{}.residuals.npy", chk_idx)))?;

        // Save chunk-specific metadata
        let dl_file = File::create(idx_path_obj.join(format!("doclens.{}.json", chk_idx)))?;
        serde_json::to_writer(BufWriter::new(dl_file), &chk_doclens)?;

        let chk_meta = Metadata {
            num_passages: chk_doclens.len(),
            num_embeddings: chk_codes.size()[0] as usize,
        };
        let meta_f_w = File::create(idx_path_obj.join(format!("{}.metadata.json", chk_idx)))?;
        serde_json::to_writer(BufWriter::new(meta_f_w), &chk_meta)?;
    }

    // -------------------------------------------------------------------------
    // 3. Update Global Metadata & Link Offsets
    // -------------------------------------------------------------------------
    // We iterate through the newly created metadata files to inject the correct
    // global embedding offsets. This allows the searcher to map a global ID
    // back to a specific file chunk.
    let new_total_chunks = start_chunk_idx + n_new_chunks;
    let old_total_embs = main_meta["num_embeddings"].as_u64().unwrap_or(0) as usize;
    let mut current_emb_offset = old_total_embs;

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
    let total_num_embs = current_emb_offset;

    // -------------------------------------------------------------------------
    // 4. Generate New Partial IVF
    // -------------------------------------------------------------------------
    // We need to determine which centroid every new token belongs to.
    // This allows us to append the new document IDs (PIDs) to the inverted lists.

    // Flatten all new codes
    let new_codes_flat = Tensor::cat(&new_codes_accumulated, 0).to_device(device);

    // Generate PIDs for the new tokens, ensuring they continue sequentially from old data.
    let mut new_pids_vec: Vec<i64> = Vec::with_capacity(new_codes_flat.size()[0] as usize);
    let mut pid_counter = old_num_passages;
    for &doc_len in &new_doclens_accumulated {
        for _ in 0..doc_len {
            new_pids_vec.push(pid_counter);
        }
        pid_counter += 1;
    }
    let new_pids = Tensor::from_slice(&new_pids_vec).to_device(device);

    // Group PIDs by their centroid assignment.
    // 1. Sort codes to group identical centroids together.
    let (sorted_new_codes, sorted_indices) = new_codes_flat.sort(0, false);
    // 2. Count how many tokens fall into each centroid.
    let new_ivf_counts = sorted_new_codes.bincount::<Tensor>(None, est_total_embs);
    // 3. Reorder PIDs to match the sorted codes.
    let new_ivf_pids_sorted = new_pids.index_select(0, &sorted_indices);

    // Optimize: Reduce redundancy. If a document appears multiple times in the same
    // centroid, we only need to store its PID once per centroid.
    let new_counts_vec: Vec<i64> = Vec::<i64>::try_from(&new_ivf_counts)?;
    let mut new_optimized_pids: Vec<Tensor> = Vec::new();
    let mut new_optimized_lengths: Vec<i64> = Vec::new();
    let mut offset = 0;

    for &count in &new_counts_vec {
        if count > 0 {
            let segment = new_ivf_pids_sorted.narrow(0, offset, count);
            // unique_dim returns (unique_elements, inverse_indices, counts)
            let (unique_pids, _, _) = segment.unique_dim(0, true, false, false);
            new_optimized_pids.push(unique_pids.shallow_clone());
            new_optimized_lengths.push(unique_pids.size()[0]);
        } else {
            // Placeholder: length 0 indicates this centroid has no new docs.
            new_optimized_lengths.push(0);
        }
        offset += count;
    }

    // -------------------------------------------------------------------------
    // 5. Merge with Old IVF
    // -------------------------------------------------------------------------
    // We now combine the existing inverted lists (from `index.ivf_index_strided`)
    // with our newly computed partial IVF lists.
    //
    //
    // The diagram above illustrates how we iterate through every centroid (partition),
    // taking the existing list of PIDs and appending the new list of PIDs, creating
    // a new, longer unified list.

    let old_ivf_flat = &index.ivf_index_strided.underlying_data;
    let old_ivf_lengths = &index.ivf_index_strided.element_lengths;

    let old_ivf_lengths_cpu = old_ivf_lengths.to_device(Device::Cpu);
    let old_lengths_vec: Vec<i64> = Vec::<i64>::try_from(&old_ivf_lengths_cpu)?;

    // Pre-calculate offsets for the old flattened IVF tensor.
    let mut old_ivf_offsets = Vec::with_capacity(old_lengths_vec.len());
    let mut curr = 0;
    for &l in &old_lengths_vec {
        old_ivf_offsets.push(curr);
        curr += l;
    }

    let mut final_ivf_parts: Vec<Tensor> = Vec::new();
    let mut final_lengths_vec: Vec<i64> = Vec::new();
    let num_partitions = est_total_embs as usize;

    // We use an iterator for `new_optimized_pids` because it only contains tensors
    // for centroids that actually received new data (count > 0).
    let mut new_pids_iter = new_optimized_pids.into_iter();

    for i in 0..num_partitions {
        let old_len = if i < old_lengths_vec.len() {
            old_lengths_vec[i]
        } else {
            0
        };
        let new_len = if i < new_optimized_lengths.len() {
            new_optimized_lengths[i]
        } else {
            0
        };

        // Append old data if it exists
        if old_len > 0 {
            let start = old_ivf_offsets[i];
            final_ivf_parts.push(old_ivf_flat.narrow(0, start, old_len));
        }

        // Append new data if it exists
        if new_len > 0 {
            if let Some(t) = new_pids_iter.next() {
                final_ivf_parts.push(t);
            }
        }
        final_lengths_vec.push(old_len + new_len);
    }

    // Concatenate all parts into a single flattened IVF tensor.
    let final_ivf_tensor = Tensor::cat(&final_ivf_parts, 0);
    let final_lengths_tensor = Tensor::from_slice(&final_lengths_vec)
        .to_device(Device::Cpu)
        .to_kind(Kind::Int); // IVF lengths are typically Int32 or Int64.

    // -------------------------------------------------------------------------
    // 6. Write Updated Global Files
    // -------------------------------------------------------------------------
    final_ivf_tensor
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64)
        .write_npy(&idx_path_obj.join("ivf.npy"))?;

    final_lengths_tensor
        .to_device(Device::Cpu)
        .write_npy(&idx_path_obj.join("ivf_lengths.npy"))?;

    // -------------------------------------------------------------------------
    // 7. Update Global Metadata
    // -------------------------------------------------------------------------
    // Calculate new average document length.
    let final_avg_doclen = if !new_doclens_accumulated.is_empty() || old_num_passages > 0 {
        let old_avg = main_meta["avg_doclen"].as_f64().unwrap_or(0.0);
        let old_sum = old_avg * (old_num_passages as f64);
        let new_sum: i64 = new_doclens_accumulated.iter().sum();
        (old_sum + new_sum as f64) / ((old_num_passages as usize + n_new_docs) as f64)
    } else {
        0.0
    };

    let final_meta_json = json!({
        "num_chunks": new_total_chunks,
        "nbits": nbits,
        "num_partitions": est_total_embs,
        "num_embeddings": total_num_embs,
        "num_passages": old_num_passages as usize + n_new_docs,
        "avg_doclen": final_avg_doclen,
    });

    let final_meta_file = fs::File::create(&main_meta_path)?;
    serde_json::to_writer_pretty(BufWriter::new(final_meta_file), &final_meta_json)?;

    Ok(())
}
