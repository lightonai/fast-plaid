use anyhow::{anyhow, bail, Result};
use indicatif::{ProgressBar, ProgressIterator};
use pyo3::prelude::*;
use serde::Serialize;
use tch::{Device, IndexOp, Kind, Tensor};

use crate::search::load::LoadedIndex;
use crate::search::padding::direct_pad_sequences;
use crate::search::tensor::StridedTensor;
use crate::utils::residual_codec::ResidualCodec;

/// Decompresses residual vectors from a packed, quantized format.
///
/// This function reconstructs full embedding vectors by combining coarse centroids with
/// fine-grained, quantized residuals. The residuals are packed with multiple codes per byte
/// (determined by `nbits`) and are unpacked using a series of lookup tables. This is a
/// typical operation in multi-stage vector quantization schemes designed to reduce
/// memory footprint.
///
/// The process involves:
/// 1. Unpacking `nbits` codes from each byte in `packed_residuals` using a bit-reversal map.
/// 2. Performing a series of indexed lookups to translate these codes into quantization bucket weights.
/// 3. Selecting the coarse centroids corresponding to the input `codes`.
/// 4. Adding the retrieved bucket weights (the decompressed residuals) to the coarse centroids.
///
/// # Preconditions
///
/// This function assumes specific dimensional relationships and will not work correctly if they
/// are not met. The caller must ensure:
/// - `(emb_dim * nbits)` is perfectly divisible by 8.
/// - 8 is perfectly divisible by `nbits`.
/// - The first dimension of `packed_residuals` matches the first dimension of `codes`.
/// - The second dimension of `packed_residuals` is `(emb_dim * nbits) / 8`.
///
/// # Arguments
///
/// * `packed_residuals` - The tensor of compressed residuals, where multiple codes are packed into each byte.
/// * `bucket_weights` - The codebook containing the fine-grained quantization vectors.
/// * `byte_reversed_bits_map` - A lookup table to efficiently unpack `nbits` codes from a byte.
/// * `bucket_weight_indices_lookup` - An intermediate table to map unpacked codes to `bucket_weights` indices.
/// * `codes` - Indices used to select the initial coarse centroids for each embedding.
/// * `centroids` - The codebook of coarse centroids.
/// * `emb_dim` - The dimensionality of the final, decompressed embedding vectors.
/// * `nbits` - The number of bits used for each sub-quantizer code within the packed residuals.
///
/// # Returns
///
/// A `Tensor` of shape `[num_embeddings, emb_dim]` containing the fully decompressed embeddings.
pub fn decompress_residuals(
    packed_residuals: &Tensor,
    bucket_weights: &Tensor,
    byte_reversed_bits_map: &Tensor,
    bucket_weight_indices_lookup: &Tensor,
    codes: &Tensor,
    centroids: &Tensor,
    emb_dim: i64,
    nbits: i64,
) -> Tensor {
    let num_embeddings = codes.size()[0];

    const BITS_PER_PACKED_UNIT: i64 = 8;
    let packed_dim = (emb_dim * nbits) / BITS_PER_PACKED_UNIT;
    let codes_per_packed_unit = BITS_PER_PACKED_UNIT / nbits;

    let retrieved_centroids = centroids.index_select(0, codes);
    let reshaped_centroids =
        retrieved_centroids.view([num_embeddings, packed_dim, codes_per_packed_unit]);

    let flat_packed_residuals_u8 = packed_residuals.flatten(0, -1);
    let flat_packed_residuals_indices = flat_packed_residuals_u8.to_kind(Kind::Int64);

    let flat_reversed_bits = byte_reversed_bits_map.index_select(0, &flat_packed_residuals_indices);
    let reshaped_reversed_bits = flat_reversed_bits.view([num_embeddings, packed_dim]);

    let flat_reversed_bits_for_lookup = reshaped_reversed_bits.flatten(0, -1);

    let flat_selected_bucket_indices =
        bucket_weight_indices_lookup.index_select(0, &flat_reversed_bits_for_lookup);
    let reshaped_selected_bucket_indices =
        flat_selected_bucket_indices.view([num_embeddings, packed_dim, codes_per_packed_unit]);

    let flat_bucket_indices_for_weights = reshaped_selected_bucket_indices.flatten(0, -1);

    let flat_gathered_weights = bucket_weights.index_select(0, &flat_bucket_indices_for_weights);
    let reshaped_gathered_weights =
        flat_gathered_weights.view([num_embeddings, packed_dim, codes_per_packed_unit]);

    let output_contributions_sum = reshaped_gathered_weights + reshaped_centroids;
    let decompressed_embeddings = output_contributions_sum.view([num_embeddings, emb_dim]);

    decompressed_embeddings
}

/// Represents the results of a single search query.
///
/// This struct is designed to be exposed to Python via `PyO3` and is also
/// serializable. It encapsulates the retrieved passage IDs and their
/// corresponding scores for a specific query.
#[pyclass]
#[derive(Serialize, Debug)]
pub struct QueryResult {
    /// The unique identifier for the query that produced these results.
    #[pyo3(get)]
    pub query_id: usize,
    /// A vector of document or passage identifiers, ranked by relevance.
    #[pyo3(get)]
    pub passage_ids: Vec<i64>,
    /// A vector of relevance scores corresponding to each passage in `passage_ids`.
    #[pyo3(get)]
    pub scores: Vec<f32>,
}

/// Search configuration parameters, exposed to Python.
#[pyclass]
#[derive(Clone, Debug)]
pub struct SearchParameters {
    /// Number of queries per batch.
    #[pyo3(get, set)]
    pub batch_size: usize,
    /// Number of documents to re-rank with exact scores.
    #[pyo3(get, set)]
    pub n_full_scores: usize,
    /// Number of final results to return per query.
    #[pyo3(get, set)]
    pub top_k: usize,
    /// Number of IVF cells to probe during the initial search.
    #[pyo3(get, set)]
    pub n_ivf_probe: usize,
}

#[pymethods]
impl SearchParameters {
    /// Creates a new `SearchParameters` instance from Python.
    #[new]
    fn new(batch_size: usize, n_full_scores: usize, top_k: usize, n_ivf_probe: usize) -> Self {
        Self {
            batch_size,
            n_full_scores,
            top_k,
            n_ivf_probe,
        }
    }
}

/// Processes a batch of queries against the loaded index.
///
/// This function iterates through query embeddings, executes the core search logic for each,
/// and collects the results, displaying a progress bar.
///
/// # Arguments
///
/// * `queries` - A 3D tensor of query embeddings with shape `[num_queries, tokens_per_query, dim]`.
/// * `index` - The `LoadedIndex` containing all necessary index components.
/// * `params` - `SearchParameters` for search configuration.
/// * `device` - The `tch::Device` for computation.
///
/// # Returns
///
/// A `Result` with a `Vec<QueryResult>`. Individual search failures result in an empty
/// `QueryResult` for that specific query, ensuring the operation doesn't halt.
pub fn search_index(
    queries: &Tensor,
    index: &LoadedIndex,
    params: &SearchParameters,
    device: Device,
    show_progress: bool,
) -> Result<Vec<QueryResult>> {
    let [num_queries, _, query_dim] = queries.size()[..] else {
        bail!(
            "Expected a 3D tensor for queries, but got shape {:?}",
            queries.size()
        );
    };

    let search_closure = |idx| {
        let query_embedding = queries.i(idx).to(device);

        let (passage_ids, scores) = search(
            &query_embedding,
            &index.ivf_index_strided,
            &index.codec,
            query_dim,
            &index.doc_codes_strided,
            &index.doc_residuals_strided,
            params.n_ivf_probe as i64,
            params.batch_size as i64,
            params.n_full_scores as i64,
            index.nbits,
            params.top_k,
            device,
        )
        .unwrap_or_default();

        QueryResult {
            query_id: idx as usize,
            passage_ids,
            scores,
        }
    };

    let results = if show_progress {
        let bar = ProgressBar::new(num_queries.try_into().unwrap());
        (0..num_queries)
            .progress_with(bar)
            .map(search_closure)
            .collect()
    } else {
        (0..num_queries).map(search_closure).collect()
    };

    Ok(results)
}

/// Reduces token-level similarity scores into a final document score using the ColBERT MaxSim strategy.
///
/// This function implements the core reduction step of the ColBERT model's scoring mechanism.
/// It first finds the maximum similarity score for each document token across all query tokens,
/// effectively ignoring padded tokens in the document. Then, it sums these maximum scores to
/// produce a single relevance score for each query-document pair in the batch.
///
/// # Arguments
///
/// * `token_scores` - A 3D `Tensor` of shape `[batch_size, query_length, doc_length]`
///   containing the token-level similarity scores.
/// * `attention_mask` - A 2D `Tensor` of shape `[batch_size, doc_length]` where `true`
///   indicates a valid token and `false` indicates a padded token.
///
/// # Returns
///
/// A 1D `Tensor` of shape `[batch_size]`, where each element is the final aggregated
/// ColBERT score for a query-document pair.
pub fn colbert_score_reduce(token_scores: &Tensor, attention_mask: &Tensor) -> Tensor {
    let scores_shape = token_scores.size();

    // Expand the document attention mask to match the shape of the token scores.
    let expanded_mask = attention_mask.unsqueeze(-1).expand(&scores_shape, true);

    // Invert the mask to identify padding positions.
    let padding_mask = expanded_mask.logical_not();

    // Nullify scores at padded positions by filling them with a large negative number.
    let masked_scores = token_scores.masked_fill(&padding_mask, -9999.0);

    // For each document token, find the maximum similarity score across all query tokens (MaxSim).
    let (max_scores_per_token, _) = masked_scores.max_dim(1, false);

    // Sum the MaxSim scores for all tokens in each document to get the final score.
    max_scores_per_token.sum_dim_intlist(-1, false, Kind::Float)
}

/// Performs a multi-stage search for a query against a quantized document index.
///
/// This function implements a multi-step search process common in efficient vector retrieval systems:
/// 1.  **IVF Probing**: Identifies a set of candidate documents by selecting the nearest Inverted File (IVF) cells.
/// 2.  **Approximate Scoring**: Computes fast, approximate scores for the candidate documents using their quantized codes.
/// 3.  **Re-ranking**: Filters the candidates based on approximate scores, then decompressesthe residuals for a smaller subset and computes exact scores.
/// 4.  **Top-K Selection**: Returns the highest-scoring documents.
///
/// # Arguments
/// * `query_embeddings` - A tensor containing the query embeddings.
/// * `ivf_index_strided` - A strided tensor representing the IVF index for coarse lookup.
/// * `codec` - The `ResidualCodec` used for decompressing document vectors.
/// * `emb_dim` - The dimensionality of the embeddings.
/// * `doc_codes_strided` - A strided tensor containing the quantized codes for all documents.
/// * `doc_residuals_strided` - A strided tensor containing the compressed residuals for all documents.
/// * `n_ivf_probe` - The number of IVF cells to probe for candidate documents.
/// * `batch_size` - The batch size used for processing documents during scoring.
/// * `n_docs_for_full_score` - The number of top documents from the approximate scoring phase to re-rank with full scoring.
/// * `nbits_param` - The number of bits used in the quantization codec.
/// * `top_k` - The final number of top results to return.
/// * `device` - The `tch::Device` (e.g., `Device::Cuda(0)`) on which to perform computations.
///
/// # Returns
/// A `Result` containing a tuple of two vectors: the top `k` passage IDs (`Vec<i64>`) and their
/// corresponding final scores (`Vec<f32>`).
///
/// # Errors
/// This function returns an error if tensor operations fail, if tensor dimensions are mismatched,
/// or if the provided `codec` is missing components required for full decompression.
pub fn search(
    query_embeddings: &Tensor,
    ivf_index_strided: &StridedTensor,
    codec: &ResidualCodec,
    emb_dim: i64,
    doc_codes_strided: &StridedTensor,
    doc_residuals_strided: &StridedTensor,
    n_ivf_probe: i64,
    batch_size: i64,
    n_docs_for_full_score: i64,
    nbits_param: i64,
    top_k: usize,
    device: Device,
) -> anyhow::Result<(Vec<i64>, Vec<f32>)> {
    let (pids, scores) = tch::no_grad(|| {
        let query_embeddings_typed = query_embeddings.to_kind(Kind::Float);

        let query_embeddings_unsqueezed = query_embeddings_typed.unsqueeze(0);

        let query_centroid_scores = codec
            .centroids
            .matmul(&query_embeddings_typed.transpose(0, 1));

        let selected_ivf_cells_indices = if n_ivf_probe == 1 {
            query_centroid_scores.argmax(0, true).permute(&[1, 0])
        } else {
            query_centroid_scores
                .topk(n_ivf_probe, 0, true, false)
                .1
                .permute(&[1, 0])
        };

        let flat_selected_ivf_cells = selected_ivf_cells_indices.flatten(0, -1).contiguous();
        let (unique_ivf_cells_to_probe, _, _) =
            flat_selected_ivf_cells.unique_dim(-1, false, false, false);

        let (retrieved_passage_ids_ivf, _) =
            ivf_index_strided.lookup(&unique_ivf_cells_to_probe, device);
        let (sorted_passage_ids_ivf, _) = retrieved_passage_ids_ivf.sort(0, false);
        let (unique_passage_ids_after_ivf, _, _) =
            sorted_passage_ids_ivf.unique_consecutive(false, false, 0);

        if unique_passage_ids_after_ivf.numel() == 0 {
            return Ok((vec![], vec![]));
        }

        let mut approx_score_chunks = Vec::new();
        let total_pids_for_approx = unique_passage_ids_after_ivf.size()[0];
        let num_approx_batches = (total_pids_for_approx + batch_size - 1) / batch_size;

        for batch_idx in 0..num_approx_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = ((batch_idx + 1) * batch_size).min(total_pids_for_approx);
            if batch_start >= batch_end {
                continue;
            }

            let batch_pids =
                unique_passage_ids_after_ivf.narrow(0, batch_start, batch_end - batch_start);
            let (batch_packed_codes, batch_doc_lengths) =
                doc_codes_strided.lookup(&batch_pids, device);

            if batch_packed_codes.numel() == 0 {
                approx_score_chunks.push(Tensor::zeros(
                    &[batch_pids.size()[0]],
                    (Kind::Float, device),
                ));
                continue;
            }

            let batch_approx_scores =
                query_centroid_scores.index_select(0, &batch_packed_codes.to_kind(Kind::Int64));
            let (padded_approx_scores, mask) =
                direct_pad_sequences(&batch_approx_scores, &batch_doc_lengths, 0.0, device)?;
            approx_score_chunks.push(colbert_score_reduce(&padded_approx_scores, &mask));
        }

        let approx_scores = if !approx_score_chunks.is_empty() {
            Tensor::cat(&approx_score_chunks, 0)
        } else {
            Tensor::empty(&[0], (Kind::Float, device))
        };

        if approx_scores.size().get(0) != Some(&unique_passage_ids_after_ivf.size()[0]) {
            return Err(anyhow!(
                "PID ({}) and approx scores ({}) count mismatch.",
                unique_passage_ids_after_ivf.size()[0],
                approx_scores.size().get(0).unwrap_or(&-1),
            ));
        }

        let mut passage_ids_to_rerank = unique_passage_ids_after_ivf;
        let mut working_approx_scores = approx_scores;

        if n_docs_for_full_score < working_approx_scores.size()[0]
            && working_approx_scores.numel() > 0
        {
            let (top_scores, top_indices) =
                working_approx_scores.topk(n_docs_for_full_score, 0, true, true);
            passage_ids_to_rerank = passage_ids_to_rerank.index_select(0, &top_indices);
            working_approx_scores = top_scores;
        }

        let n_pids_for_decomp = (n_docs_for_full_score / 4).max(1);
        if n_pids_for_decomp < working_approx_scores.size()[0] && working_approx_scores.numel() > 0
        {
            let (_, top_indices) = working_approx_scores.topk(n_pids_for_decomp, 0, true, true);
            passage_ids_to_rerank = passage_ids_to_rerank.index_select(0, &top_indices);
        }

        if passage_ids_to_rerank.numel() == 0 {
            return Ok((vec![], vec![]));
        }

        let (final_codes, final_doc_lengths) =
            doc_codes_strided.lookup(&passage_ids_to_rerank, device);
        let (final_residuals, _) = doc_residuals_strided.lookup(&passage_ids_to_rerank, device);

        let bucket_weights = codec
            .bucket_weights
            .as_ref()
            .ok_or_else(|| anyhow!("Codec missing bucket_weights for decompression."))?;
        let decomp_lookup = codec
            .decomp_indices_lookup
            .as_ref()
            .ok_or_else(|| anyhow!("Codec missing decomp_indices_lookup for decompression."))?;

        let decompressed_embs = decompress_residuals(
            &final_residuals,
            bucket_weights,
            &codec.byte_reversed_bits_map,
            decomp_lookup,
            &final_codes,
            &codec.centroids,
            emb_dim,
            nbits_param,
        );

        let (padded_doc_embs, mask) =
            direct_pad_sequences(&decompressed_embs, &final_doc_lengths, 0.0, device)?;
        let final_scores = padded_doc_embs.matmul(&query_embeddings_unsqueezed.transpose(-2, -1));
        let reduced_scores = colbert_score_reduce(&final_scores, &mask);

        let (sorted_scores, sorted_indices) = reduced_scores.sort(0, true);
        let sorted_pids = passage_ids_to_rerank.index_select(0, &sorted_indices);

        let pids_vec: Vec<i64> = sorted_pids.try_into()?;
        let scores_vec: Vec<f32> = sorted_scores.try_into()?;

        let result_count = top_k.min(pids_vec.len());
        Ok((
            pids_vec[..result_count].to_vec(),
            scores_vec[..result_count].to_vec(),
        ))
    })?;

    Ok((pids, scores))
}
