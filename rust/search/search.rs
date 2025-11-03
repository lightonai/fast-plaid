use anyhow::{anyhow, Result};
use indicatif::{ProgressBar, ProgressIterator};
use pyo3::prelude::*;
use serde::Serialize;
use tch::{Device, Kind, Tensor};

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
/// - `(embedding_dimension * nbits)` is perfectly divisible by 8.
/// - 8 is perfectly divisible by `nbits`.
/// - The first dimension of `packed_residuals` matches the first dimension of `codes`.
/// - The second dimension of `packed_residuals` is `(embedding_dimension * nbits) / 8`.
///
/// # Arguments
///
/// * `packed_residuals` - The tensor of compressed residuals, where multiple codes are packed into each byte.
/// * `bucket_weights` - The codebook containing the fine-grained quantization vectors.
/// * `byte_reversed_bits_map` - A lookup table to efficiently unpack `nbits` codes from a byte.
/// * `bucket_weight_indices_lookup` - An intermediate table to map unpacked codes to `bucket_weights` indices.
/// * `codes` - Indices used to select the initial coarse centroids for each embedding.
/// * `centroids` - The codebook of coarse centroids.
/// * `embedding_dimension` - The dimensionality of the final, decompressed embedding vectors.
/// * `nbits` - The number of bits used for each sub-quantizer code within the packed residuals.
///
/// # Returns
///
/// A `Tensor` of shape `[num_embeddings, embedding_dimension]` containing the fully decompressed embeddings.
pub fn decompress_residuals(
    packed_residuals: &Tensor,
    bucket_weights: &Tensor,
    byte_reversed_bits_map: &Tensor,
    bucket_weight_indices_lookup: &Tensor,
    codes: &Tensor,
    centroids: &Tensor,
    embedding_dimension: i64,
    nbits: i64,
) -> Tensor {
    let num_embeddings = codes.size()[0];

    const BITS_PER_PACKED_UNIT: i64 = 8;
    let packed_dim = (embedding_dimension * nbits) / BITS_PER_PACKED_UNIT;
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
    let decompressed_embeddings =
        output_contributions_sum.view([num_embeddings, embedding_dimension]);

    let norms = decompressed_embeddings
        .norm_scalaropt_dim(2.0, &[-1], true)
        .clamp_min(1e-12);

    decompressed_embeddings / norms
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
/// * `subset` - An optional list of document ID lists to restrict the search for each query.
///
/// # Returns
///
/// A `Result` with a `Vec<QueryResult>`. Individual search failures result in an empty
/// `QueryResult` for that specific query, ensuring the operation doesn't halt.
pub fn search_many(
    queries: &Vec<Tensor>,
    index: &LoadedIndex,
    params: &SearchParameters,
    device: Device,
    show_progress: bool,
    subset: Option<Vec<Vec<i64>>>,
) -> Result<Vec<QueryResult>> {
    let num_queries = queries.len();
    if num_queries == 0 {
        return Ok(Vec::new());
    }
    let query_dim = queries[0].size()[queries[0].dim() - 1];

    let search_closure = |query_index: usize| {
        let query_embedding = &queries[query_index].to(device);

        // Handle the per-query subset list
        let query_subset = subset.as_ref().and_then(|s| s.get(query_index as usize));
        let subset_tensor = query_subset.map(|ids| {
            Tensor::from_slice(ids)
                .to_device(device)
                .to_kind(Kind::Int64)
        });

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
            subset_tensor.as_ref(),
        )
        .unwrap_or_default();

        QueryResult {
            query_id: query_index as usize,
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

/// Intersects two tensors of integer IDs, returning a new tensor with the common elements.
///
/// This function implements an efficient intersection algorithm for tensors on a `tch` device.
/// It works by concatenating the two input tensors, sorting the result, and then identifying
/// adjacent duplicate elements, which correspond to the elements present in both original tensors.
///
/// # Preconditions
///
/// * The `passage_ids` tensor must be sorted and contain unique elements.
///
/// # Arguments
///
/// * `passage_ids` - A 1D tensor of passage IDs, assumed to be sorted and unique.
/// * `subset` - A 1D tensor of passage IDs to intersect with `passage_ids`. This tensor does not
///   need to be sorted or unique, as this function will handle it.
/// * `device` - The `tch::Device` on which to create an empty tensor if the result is empty.
///
/// # Returns
///
/// A new 1D `Tensor` containing only the elements that are present in both `passage_ids` and `subset`,
/// sorted in ascending order.
fn filter_passage_ids_with_subset(passage_ids: &Tensor, subset: &Tensor, device: Device) -> Tensor {
    if subset.numel() == 0 || passage_ids.numel() == 0 {
        return Tensor::empty(&[0], (Kind::Int64, device));
    }

    let (sorted_subset, _) = subset.sort(0, false);
    let (unique_sorted_subset, _, _) = sorted_subset.unique_consecutive(false, false, 0);

    let concatenated = Tensor::cat(&[passage_ids, &unique_sorted_subset], 0);

    let (sorted_concatenated, _) = concatenated.sort(0, false);
    let size = sorted_concatenated.size()[0];

    if size < 2 {
        return Tensor::empty(&[0], (Kind::Int64, device));
    }

    let duplicates_mask = sorted_concatenated
        .narrow(0, 0, size - 1)
        .eq_tensor(&sorted_concatenated.narrow(0, 1, size - 1));

    sorted_concatenated
        .narrow(0, 1, size - 1)
        .masked_select(&duplicates_mask)
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
/// * `embedding_dimension` - The dimensionality of the embeddings.
/// * `doc_codes_strided` - A strided tensor containing the quantized codes for all documents.
/// * `doc_residuals_strided` - A strided tensor containing the compressed residuals for all documents.
/// * `n_ivf_probe` - The number of IVF cells to probe for candidate documents.
/// * `batch_size` - The batch size used for processing documents during scoring.
/// * `n_docs_for_full_score` - The number of top documents from the approximate scoring phase to re-rank with full scoring.
/// * `nbits_param` - The number of bits used in the quantization codec.
/// * `top_k` - The final number of top results to return.
/// * `device` - The `tch::Device` (e.g., `Device::Cuda(0)`) on which to perform computations.
/// * `subset` - An optional tensor of document IDs to restrict the search to.
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
    embedding_dimension: i64,
    doc_codes_strided: &StridedTensor,
    doc_residuals_strided: &StridedTensor,
    n_ivf_probe: i64,
    batch_size: i64,
    n_docs_for_full_score: i64,
    nbits_param: i64,
    top_k: usize,
    device: Device,
    subset: Option<&Tensor>,
) -> anyhow::Result<(Vec<i64>, Vec<f32>)> {
    let (passage_ids, scores) = tch::no_grad(|| {
        let query_embeddings_unsqueezed = query_embeddings.unsqueeze(0);

        let query_centroid_scores = codec.centroids.matmul(&query_embeddings.transpose(0, 1));

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

        let (mut unique_passage_ids, _, _) =
            sorted_passage_ids_ivf.unique_consecutive(false, false, 0);

        if let Some(subset_tensor) = subset {
            unique_passage_ids =
                filter_passage_ids_with_subset(&unique_passage_ids, subset_tensor, device);
        }

        if unique_passage_ids.numel() == 0 {
            return Ok((vec![], vec![]));
        }

        let mut approx_score_chunks = Vec::new();
        let total_passage_ids_for_approx = unique_passage_ids.size()[0];
        let num_approx_batches = (total_passage_ids_for_approx + batch_size - 1) / batch_size;

        for step in 0..num_approx_batches {
            let batch_start = step * batch_size;
            let batch_end = ((step + 1) * batch_size).min(total_passage_ids_for_approx);
            if batch_start >= batch_end {
                continue;
            }

            let batch_passage_ids =
                unique_passage_ids.narrow(0, batch_start, batch_end - batch_start);
            let (batch_packed_codes, batch_doc_lengths) =
                doc_codes_strided.lookup(&batch_passage_ids, device);

            if batch_packed_codes.numel() == 0 {
                approx_score_chunks.push(Tensor::zeros(
                    &[batch_passage_ids.size()[0]],
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

        let mut approx_scores = if !approx_score_chunks.is_empty() {
            Tensor::cat(&approx_score_chunks, 0)
        } else {
            Tensor::empty(&[0], (Kind::Float, device))
        };

        if approx_scores.size().get(0) != Some(&unique_passage_ids.size()[0]) {
            return Err(anyhow!(
                "PID ({}) and approx scores ({}) count mismatch.",
                unique_passage_ids.size()[0],
                approx_scores.size().get(0).unwrap_or(&-1),
            ));
        }

        let mut passage_ids_to_rerank = unique_passage_ids;

        if n_docs_for_full_score < approx_scores.size()[0] && approx_scores.numel() > 0 {
            let (top_scores, top_indices) =
                approx_scores.topk(n_docs_for_full_score, 0, true, true);
            passage_ids_to_rerank = passage_ids_to_rerank.index_select(0, &top_indices);
            approx_scores = top_scores;
        }

        let n_passage_ids_for_decompression = (n_docs_for_full_score / 4).max(1);
        if n_passage_ids_for_decompression < approx_scores.size()[0] && approx_scores.numel() > 0 {
            let (_, top_indices) =
                approx_scores.topk(n_passage_ids_for_decompression, 0, true, true);
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
        let bucket_weight_indices_lookup =
            codec.bucket_weight_indices_lookup.as_ref().ok_or_else(|| {
                anyhow!("Codec missing bucket_weight_indices_lookup for decompression.")
            })?;

        let decompressed_embeddings = decompress_residuals(
            &final_residuals,
            bucket_weights,
            &codec.byte_reversed_bits_map,
            bucket_weight_indices_lookup,
            &final_codes,
            &codec.centroids,
            embedding_dimension,
            nbits_param,
        );

        let (padded_doc_embeddings, mask) =
            direct_pad_sequences(&decompressed_embeddings, &final_doc_lengths, 0.0, device)?;
        let scores = padded_doc_embeddings.matmul(&query_embeddings_unsqueezed.transpose(-2, -1));
        let scores = colbert_score_reduce(&scores, &mask);

        let (scores, sorted_indices) = scores.sort(0, true);
        let sorted_passage_ids = passage_ids_to_rerank.index_select(0, &sorted_indices);

        let sorted_passage_ids: Vec<i64> = sorted_passage_ids.try_into()?;
        let scores: Vec<f32> = scores.try_into()?;

        let result_count = top_k.min(sorted_passage_ids.len());

        Ok((
            sorted_passage_ids[..result_count].to_vec(),
            scores[..result_count].to_vec(),
        ))
    })?;

    Ok((passage_ids, scores))
}
