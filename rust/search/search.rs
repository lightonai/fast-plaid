use anyhow::{anyhow, bail, Result};
use indicatif::{ProgressBar, ProgressIterator};
use pyo3::prelude::*;
use serde::Serialize;
use tch::{Device, IndexOp, Kind, Tensor};

use pyo3_tch::PyTensor;

use crate::search::load::LoadedIndex;
use crate::search::padding::direct_pad_sequences;
use crate::search::tensor::StridedTensor;
use crate::utils::residual_codec::ResidualCodec;

/// Fallback per-device workspace budget when the caller does not provide one.
const DEFAULT_MEMORY_BUDGET_BYTES: i64 = 4 * (1 << 30);

/// Never shrink a stage budget below this during OOM retries.
const MIN_STAGE_BUDGET_BYTES: i64 = 64 * (1 << 20);

/// Approximate-scoring bytes per padded (doc, token) cell: fp16 score gather + padded copy, plus code gathers.
fn approx_bytes_per_cell(q_tokens: i64) -> i64 {
    4 * q_tokens + 32
}

/// Exact-scoring bytes per padded (doc, token) cell: fp16 token scores plus decompression intermediates and embedding copies.
fn exact_bytes_per_cell(q_tokens: i64) -> i64 {
    4 * q_tokens + 1792
}

fn is_oom_error(error: &anyhow::Error) -> bool {
    let message = error.to_string().to_lowercase();
    message.contains("out of memory") || message.contains("cuda oom")
}

/// Converts panics into errors: tch ops panic on CUDA OOM, and the retry logic needs an `Err` to inspect.
fn catch_stage_panic(run: impl FnOnce() -> Result<Tensor>) -> Result<Tensor> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic during scoring stage".to_string());
            Err(anyhow!("{}", message))
        },
    }
}

fn sort_passage_ids_by_doc_length_with_restore(
    passage_ids: &Tensor,
    doc_codes_strided: &StridedTensor,
    device: Device,
) -> (Tensor, Tensor) {
    if passage_ids.numel() <= 1 {
        return (
            passage_ids.shallow_clone(),
            Tensor::arange(passage_ids.numel() as i64, (Kind::Int64, device)),
        );
    }

    let lengths_device = doc_codes_strided.element_lengths.device();
    let ids_for_lengths = passage_ids.to_device(lengths_device).to_kind(Kind::Int64);
    let passage_lengths = doc_codes_strided
        .element_lengths
        .index_select(0, &ids_for_lengths);
    let (_, length_order) = passage_lengths.sort(0, false);
    let length_order = length_order.to_device(device);
    let sorted_passage_ids = passage_ids.index_select(0, &length_order);
    let (_, restore_order) = length_order.sort(0, false);

    (sorted_passage_ids, restore_order)
}

/// Greedy chunk boundaries over doc lengths keeping each chunk's padded workspace within `budget_bytes`.
/// Ascending lengths pack short docs densely; otherwise uniform ranges use the global max. Chunks hold >= 1 doc.
fn plan_chunk_ranges(
    lengths: &[i64],
    bytes_per_cell: i64,
    budget_bytes: i64,
    lengths_ascending: bool,
) -> Vec<(i64, i64)> {
    let n = lengths.len() as i64;
    if n == 0 {
        return Vec::new();
    }

    if !lengths_ascending {
        let max_len = lengths.iter().copied().max().unwrap_or(1).max(1);
        let docs_per_chunk = (budget_bytes / (max_len * bytes_per_cell)).max(1);
        return (0..n)
            .step_by(docs_per_chunk as usize)
            .map(|start| (start, docs_per_chunk.min(n - start)))
            .collect();
    }

    let mut ranges = Vec::new();
    let mut start: i64 = 0;
    while start < n {
        let mut len: i64 = 1;
        let mut token_sum: i128 = lengths[start as usize].max(1) as i128;
        while start + len < n {
            let chunk_max = lengths[(start + len) as usize].max(1);
            let padded = (len + 1) as i128 * chunk_max as i128;
            // Break on budget, or when padding would exceed 2x the true tokens (skewed lengths).
            if padded * bytes_per_cell as i128 > budget_bytes as i128
                || padded > 2 * (token_sum + chunk_max as i128)
            {
                break;
            }
            token_sum += chunk_max as i128;
            len += 1;
        }
        ranges.push((start, len));
        start += len;
    }
    ranges
}

/// Scores candidates chunk-by-chunk within a memory budget, or in fixed `manual_docs`-sized
/// chunks when positive. Returned scores are always aligned with the input `ids` order.
#[allow(clippy::too_many_arguments)]
fn run_scoring_stage(
    ids: &Tensor,
    lengths_cpu: &[i64],
    bytes_per_cell: i64,
    budget_bytes: i64,
    manual_docs: i64,
    doc_codes_strided: &StridedTensor,
    device: Device,
    sort_enabled: bool,
    score_chunk: &mut dyn FnMut(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let n = ids.size()[0];
    if n == 0 {
        return Ok(Tensor::empty(&[0], (Kind::Float, device)));
    }

    let max_len = lengths_cpu.iter().copied().max().unwrap_or(1).max(1);
    let padded_cells = n as i128 * max_len as i128;
    let token_sum: i128 = lengths_cpu.iter().map(|&l| l.max(1) as i128).sum();
    // Single unsorted chunk only when it fits the budget and padding stays tight.
    if manual_docs <= 0
        && padded_cells * bytes_per_cell as i128 <= budget_bytes as i128
        && padded_cells <= 2 * token_sum
    {
        return score_chunk(ids);
    }

    let (ids_exec, restore_order) = if sort_enabled {
        let (sorted_ids, restore) =
            sort_passage_ids_by_doc_length_with_restore(ids, doc_codes_strided, device);
        (sorted_ids, Some(restore))
    } else {
        (ids.shallow_clone(), None)
    };

    let ranges: Vec<(i64, i64)> = if manual_docs > 0 {
        (0..n)
            .step_by(manual_docs as usize)
            .map(|start| (start, manual_docs.min(n - start)))
            .collect()
    } else {
        let mut planned_lengths = lengths_cpu.to_vec();
        if sort_enabled {
            planned_lengths.sort_unstable();
        }
        plan_chunk_ranges(&planned_lengths, bytes_per_cell, budget_bytes, sort_enabled)
    };

    let mut score_chunks = Vec::with_capacity(ranges.len());
    for (start, len) in ranges {
        let batch_ids = ids_exec.narrow(0, start, len);
        score_chunks.push(score_chunk(&batch_ids)?);
    }

    let mut scores = Tensor::cat(&score_chunks, 0);
    if let Some(restore) = restore_order {
        scores = scores.index_select(0, &restore);
    }
    Ok(scores)
}

/// Runs a scoring stage, halving the budget and retrying on CUDA OOM; manual chunk sizes
/// are respected verbatim, so their OOM errors propagate.
#[allow(clippy::too_many_arguments)]
fn run_scoring_stage_with_oom_retry(
    ids: &Tensor,
    lengths_cpu: &[i64],
    bytes_per_cell: i64,
    budget_bytes: i64,
    manual_docs: i64,
    doc_codes_strided: &StridedTensor,
    device: Device,
    sort_enabled: bool,
    score_chunk: &mut dyn FnMut(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let mut budget = budget_bytes.max(MIN_STAGE_BUDGET_BYTES);
    loop {
        let attempt = catch_stage_panic(|| {
            run_scoring_stage(
                ids,
                lengths_cpu,
                bytes_per_cell,
                budget,
                manual_docs,
                doc_codes_strided,
                device,
                sort_enabled,
                score_chunk,
            )
        });
        match attempt {
            Ok(scores) => return Ok(scores),
            Err(error)
                if manual_docs <= 0 && is_oom_error(&error) && budget > MIN_STAGE_BUDGET_BYTES =>
            {
                budget = (budget / 2).max(MIN_STAGE_BUDGET_BYTES);
                eprintln!(
                    "fast-plaid: scoring stage hit CUDA OOM; retrying with chunk budget {} MiB",
                    budget / (1 << 20)
                );
            },
            Err(error) => return Err(error),
        }
    }
}

/// Decompresses residual vectors from a packed, quantized format.
///
/// This function reconstructs full embedding vectors by combining coarse centroids with
/// fine-grained, quantized residuals. The residuals are packed with multiple codes per byte
/// (determined by `nbits`) and are unpacked using a series of lookup tables. This is a
/// typical operation in multi-stage vector quantization schemes designed to reduce
/// memory footprint.
///
///
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

    // Retrieve coarse centroids
    let retrieved_centroids = centroids.index_select(0, codes);
    let reshaped_centroids =
        retrieved_centroids.view([num_embeddings, packed_dim, codes_per_packed_unit]);

    // Unpack bits via lookup table
    let flat_packed_residuals_indices = packed_residuals.flatten(0, -1).to_kind(Kind::Int);
    let flat_reversed_bits = byte_reversed_bits_map
        .index_select(0, &flat_packed_residuals_indices)
        .to_kind(Kind::Uint8);
    let reshaped_reversed_bits = flat_reversed_bits.view([num_embeddings, packed_dim]);

    // Map bits to weight indices
    let flat_reversed_bits_for_lookup = reshaped_reversed_bits.flatten(0, -1);
    let flat_selected_bucket_indices = bucket_weight_indices_lookup
        .index_select(0, &flat_reversed_bits_for_lookup.to_kind(Kind::Int))
        .to_kind(Kind::Uint8);
    let reshaped_selected_bucket_indices =
        flat_selected_bucket_indices.view([num_embeddings, packed_dim, codes_per_packed_unit]);

    // Retrieve fine-grained residual weights
    let flat_bucket_indices_for_weights = reshaped_selected_bucket_indices.flatten(0, -1);
    let flat_gathered_weights =
        bucket_weights.index_select(0, &flat_bucket_indices_for_weights.to_kind(Kind::Int));
    let reshaped_gathered_weights =
        flat_gathered_weights.view([num_embeddings, packed_dim, codes_per_packed_unit]);

    // Reconstruct and normalize
    let output_contributions_sum = reshaped_gathered_weights + reshaped_centroids;
    let decompressed_embeddings =
        output_contributions_sum.view([num_embeddings, embedding_dimension]);

    let norms = decompressed_embeddings
        .norm_scalaropt_dim(2.0, &[-1], true)
        .clamp_min(1e-12);

    let normalized_embeddings = decompressed_embeddings / norms;
    normalized_embeddings
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

/// Represents the results of a single search query with token-level similarity matrices.
///
/// Similar to `QueryResult` but also includes per-document token similarity
/// matrices of shape `[query_tokens, doc_tokens]` for each returned document.
///
/// # Safety
/// `tch::Tensor` is internally reference-counted and thread-safe, but its raw
/// pointer prevents auto-deriving `Sync`. We assert it here since PyO3 requires
/// `Sync` for `#[pyclass]` structs.
#[pyclass]
#[derive(Debug)]
pub struct QueryResultWithTokenScores {
    #[pyo3(get)]
    pub query_id: usize,
    #[pyo3(get)]
    pub passage_ids: Vec<i64>,
    #[pyo3(get)]
    pub scores: Vec<f32>,
    pub token_scores_inner: Vec<Tensor>,
}

// SAFETY: tch::Tensor uses ATen's intrusive_ptr which is internally
// atomic-reference-counted and safe to share across threads.
unsafe impl Send for QueryResultWithTokenScores {}
unsafe impl Sync for QueryResultWithTokenScores {}

#[pymethods]
impl QueryResultWithTokenScores {
    /// Returns the per-document token similarity matrices.
    ///
    /// Each tensor has shape `[query_tokens, doc_tokens]` where values are
    /// the dot-product similarity between each query token and each document token
    /// (equivalent to cosine similarity when embeddings are L2-normalized).
    #[getter]
    fn token_scores(&self) -> Vec<PyTensor> {
        self.token_scores_inner
            .iter()
            .map(|t| PyTensor(t.shallow_clone()))
            .collect()
    }
}

/// Search configuration parameters, exposed to Python.
#[pyclass]
#[derive(Clone, Debug)]
pub struct SearchParameters {
    /// Manual docs per scoring chunk; 0 plans chunks from the memory budget.
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
    /// Per-device scoring workspace budget in bytes. 0 = a conservative default.
    #[pyo3(get, set)]
    pub memory_budget_bytes: usize,
}

#[pymethods]
impl SearchParameters {
    /// Creates a new `SearchParameters` instance from Python.
    #[new]
    #[pyo3(signature = (batch_size, n_full_scores, top_k, n_ivf_probe, memory_budget_bytes=0))]
    fn new(
        batch_size: usize,
        n_full_scores: usize,
        top_k: usize,
        n_ivf_probe: usize,
        memory_budget_bytes: usize,
    ) -> Self {
        Self {
            batch_size,
            n_full_scores,
            top_k,
            n_ivf_probe,
            memory_budget_bytes,
        }
    }
}

fn resolve_memory_budget(params: &SearchParameters) -> i64 {
    if params.memory_budget_bytes > 0 {
        params.memory_budget_bytes as i64
    } else {
        DEFAULT_MEMORY_BUDGET_BYTES
    }
}

/// Computes per-query offsets from packed query lengths.
fn packed_query_offsets(query_lengths: &[i64]) -> Vec<i64> {
    let mut offsets = Vec::with_capacity(query_lengths.len() + 1);
    let mut acc: i64 = 0;
    offsets.push(0);
    for len in query_lengths {
        acc += len;
        offsets.push(acc);
    }
    offsets
}

/// Processes a batch of queries against the loaded index.
///
/// This function iterates through query embeddings, executes the core search logic for each,
/// and collects the results, displaying a progress bar.
///
/// # Arguments
///
/// * `queries` - A packed 2D tensor of query embeddings with shape `[total_tokens, dim]`,
///   moved to the device once and viewed per query.
/// * `index` - The `LoadedIndex` containing all necessary index components.
/// * `params` - `SearchParameters` for search configuration.
/// * `device` - The `tch::Device` for computation.
/// * `show_progress` - Whether to display a progress bar.
/// * `subset` - An optional list of document ID lists to restrict the search for each query.
/// * `query_lengths` - True token count per query in the packed tensor.
///
/// # Returns
///
/// A `Result` with a `Vec<QueryResult>`, one per query.
pub fn search_many(
    queries: &Tensor,
    index: &LoadedIndex,
    params: &SearchParameters,
    device: Device,
    show_progress: bool,
    subset: Option<Vec<Vec<i64>>>,
    query_lengths: Vec<i64>,
) -> Result<Vec<QueryResult>> {
    let ivf_index = index.ivf_index_strided.as_ref().ok_or_else(|| {
        anyhow!(
            "This index was built with compress_only=True and does not support search. \
             Rebuild with compress_only=False to enable search."
        )
    })?;

    let (num_queries, query_dim, packed_queries, offsets) =
        prepare_query_layout(queries, &query_lengths, device)?;

    let search_closure = |query_index: i64| -> Result<QueryResult> {
        let query_embedding = get_query_embedding(&packed_queries, &offsets, query_index);

        // Handle the per-query subset list
        let query_subset = subset.as_ref().and_then(|s| s.get(query_index as usize));
        let subset_tensor = query_subset.map(|ids| {
            Tensor::from_slice(ids)
                .to_device(device)
                .to_kind(Kind::Int64)
        });

        let (passage_ids, scores, _) = search(
            &query_embedding,
            ivf_index,
            &index.codec,
            query_dim,
            &index.doc_codes_strided,
            &index.doc_residuals_strided,
            params.n_ivf_probe as i64,
            params.n_full_scores as i64,
            index.nbits,
            params.top_k,
            device,
            subset_tensor.as_ref(),
            false,
            params.batch_size as i64,
            resolve_memory_budget(params),
        )?;

        Ok(QueryResult {
            query_id: query_index as usize,
            passage_ids,
            scores,
        })
    };

    let results: Result<Vec<QueryResult>> = if show_progress {
        let bar = ProgressBar::new(num_queries.try_into().unwrap());
        (0..num_queries)
            .progress_with(bar)
            .map(search_closure)
            .collect()
    } else {
        (0..num_queries).map(search_closure).collect()
    };

    results
}

/// Moves the packed 2D queries to the device, returning `(num_queries, dim, packed, offsets)`.
fn prepare_query_layout(
    queries: &Tensor,
    query_lengths: &[i64],
    device: Device,
) -> Result<(i64, i64, Tensor, Vec<i64>)> {
    let [total_tokens, query_dim] = queries.size()[..] else {
        bail!(
            "Expected a 2D packed tensor [total_tokens, dim], got shape {:?}",
            queries.size()
        );
    };
    let offsets = packed_query_offsets(query_lengths);
    if *offsets.last().unwrap_or(&0) != total_tokens {
        bail!(
            "query_lengths sum ({}) does not match packed token count ({})",
            offsets.last().unwrap_or(&0),
            total_tokens
        );
    }
    let mut packed = queries.to_device(device);
    if packed.kind() != Kind::Half {
        packed = packed.to_kind(Kind::Half);
    }
    Ok((query_lengths.len() as i64, query_dim, packed, offsets))
}

/// Returns query `query_index` as a zero-copy view into the packed tensor.
fn get_query_embedding(packed_queries: &Tensor, offsets: &[i64], query_index: i64) -> Tensor {
    let start = offsets[query_index as usize];
    packed_queries.narrow(0, start, offsets[query_index as usize + 1] - start)
}

/// Processes a batch of queries and returns results with token-level similarity matrices.
///
/// Similar to `search_many` but each result includes per-document token similarity
/// matrices of shape `[query_tokens, doc_tokens]`.
pub fn search_many_with_token_scores(
    queries: &Tensor,
    index: &LoadedIndex,
    params: &SearchParameters,
    device: Device,
    show_progress: bool,
    subset: Option<Vec<Vec<i64>>>,
    query_lengths: Vec<i64>,
) -> Result<Vec<QueryResultWithTokenScores>> {
    let ivf_index = index.ivf_index_strided.as_ref().ok_or_else(|| {
        anyhow!(
            "This index was built with compress_only=True and does not support search. \
             Rebuild with compress_only=False to enable search."
        )
    })?;

    let (num_queries, query_dim, packed_queries, offsets) =
        prepare_query_layout(queries, &query_lengths, device)?;

    let search_closure = |query_index: i64| -> Result<QueryResultWithTokenScores> {
        let query_embedding = get_query_embedding(&packed_queries, &offsets, query_index);

        let query_subset = subset.as_ref().and_then(|s| s.get(query_index as usize));
        let subset_tensor = query_subset.map(|ids| {
            Tensor::from_slice(ids)
                .to_device(device)
                .to_kind(Kind::Int64)
        });

        let (passage_ids, scores, token_matrices) = search(
            &query_embedding,
            ivf_index,
            &index.codec,
            query_dim,
            &index.doc_codes_strided,
            &index.doc_residuals_strided,
            params.n_ivf_probe as i64,
            params.n_full_scores as i64,
            index.nbits,
            params.top_k,
            device,
            subset_tensor.as_ref(),
            true,
            params.batch_size as i64,
            resolve_memory_budget(params),
        )?;

        Ok(QueryResultWithTokenScores {
            query_id: query_index as usize,
            passage_ids,
            scores,
            token_scores_inner: token_matrices.unwrap_or_default(),
        })
    };

    let results: Result<Vec<QueryResultWithTokenScores>> = if show_progress {
        let bar = ProgressBar::new(num_queries.try_into().unwrap());
        (0..num_queries)
            .progress_with(bar)
            .map(search_closure)
            .collect()
    } else {
        (0..num_queries).map(search_closure).collect()
    };

    results
}

/// Reduces token-level similarity scores into a final document score using the ColBERT MaxSim strategy.
///
/// Takes ownership of `token_scores` (always freshly materialized by the caller) so the
/// padding mask can be applied in place instead of allocating a full-size copy.
///
/// # Arguments
///
/// * `token_scores` - A 3D `Tensor` of shape `[batch_size, doc_length, query_length]`
///   containing the token-level similarity scores (freshly allocated).
/// * `attention_mask` - A 2D `Tensor` of shape `[batch_size, doc_length]` where `true`
///   indicates a valid token and `false` indicates a padded token.
///
/// # Returns
///
/// A 1D `Tensor` of shape `[batch_size]`, where each element is the final aggregated
/// ColBERT score for a query-document pair.
pub fn colbert_score_reduce(token_scores: Tensor, attention_mask: &Tensor) -> Tensor {
    // Broadcast the document padding mask across query tokens without
    // materializing a full [batch, doc_len, query_len] boolean tensor.
    let padding_mask = attention_mask.logical_not().unsqueeze(-1);

    // Mask padded positions in place: the tensor is freshly materialized, so no aliasing.
    let mut token_scores = token_scores;
    let masked_scores = token_scores.masked_fill_(&padding_mask, -9999.0);

    // For each document token, find the maximum similarity score across all query tokens (MaxSim).
    let (max_scores_per_token, _) = masked_scores.max_dim(1, false);

    // Sum the MaxSim scores for all tokens in each document to get the final score.
    max_scores_per_token.sum_dim_intlist(-1, false, Kind::Float)
}

/// Helper function: Intersects two 1D tensors that are ALREADY sorted and unique.
///
/// Used for optimizing subset filtering by avoiding hash sets or broadcasting checks.
fn intersect_sorted_unique_tensors(t1: &Tensor, t2: &Tensor) -> Tensor {
    if t1.numel() == 0 || t2.numel() == 0 {
        return Tensor::empty(&[0], (t1.kind(), t1.device()));
    }

    let concatenated = Tensor::cat(&[t1, t2], 0);
    let (sorted, _) = concatenated.sort(0, false);

    let size = sorted.size()[0];
    if size < 2 {
        return Tensor::empty(&[0], (t1.kind(), t1.device()));
    }

    let duplicates_mask = sorted
        .narrow(0, 0, size - 1)
        .eq_tensor(&sorted.narrow(0, 1, size - 1));

    sorted
        .narrow(0, 1, size - 1)
        .masked_select(&duplicates_mask)
}

/// Filters passage IDs by intersecting with a provided subset.
fn filter_passage_ids_with_subset(passage_ids: &Tensor, subset: &Tensor, device: Device) -> Tensor {
    if subset.numel() == 0 || passage_ids.numel() == 0 {
        return Tensor::empty(&[0], (Kind::Int64, device));
    }

    let (sorted_subset, _) = subset.sort(0, false);
    let (unique_sorted_subset, _, _) = sorted_subset.unique_consecutive(false, false, 0);

    intersect_sorted_unique_tensors(passage_ids, &unique_sorted_subset)
}

/// Performs a multi-stage search for a query against a quantized document index.
///
/// This function implements the standard PLAID pipeline:
/// 1.  **IVF Probing**: Identifies a set of candidate documents by selecting the nearest Inverted File (IVF) cells.
/// 2.  **Approximate Scoring**: Computes fast, approximate scores for the candidate documents using their quantized codes.
/// 3.  **Re-ranking**: Filters the candidates based on approximate scores, then decompresses the residuals for a smaller subset and computes exact scores.
/// 4.  **Top-K Selection**: Returns the highest-scoring documents.
///
/// Both scoring stages run in chunks planned from `memory_budget_bytes`, or in fixed
/// `batch_size`-doc chunks when `batch_size` > 0; chunking is score-preserving
/// (candidate set, `n_full_scores`, `n_ivf_probe` and `top_k` are unchanged).
///
/// # Arguments
/// * `query_embeddings` - A 2D tensor `[query_tokens, dim]` holding the query embeddings
///   (already trimmed of padding and resident on `device` for the packed path).
/// * `ivf_index_strided` - A strided tensor representing the IVF index for coarse lookup.
/// * `codec` - The `ResidualCodec` used for decompressing document vectors.
/// * `embedding_dimension` - The dimensionality of the embeddings.
/// * `doc_codes_strided` - A strided tensor containing the quantized codes for all documents.
/// * `doc_residuals_strided` - A strided tensor containing the compressed residuals for all documents.
/// * `n_ivf_probe` - The number of IVF cells to probe for candidate documents.
/// * `n_docs_for_full_score` - The number of top documents from the approximate scoring phase to re-rank with full scoring.
/// * `nbits_param` - The number of bits used in the quantization codec.
/// * `top_k` - The final number of top results to return.
/// * `device` - The `tch::Device` (e.g., `Device::Cuda(0)`) on which to perform computations.
/// * `subset` - An optional tensor of document IDs to restrict the search to.
/// * `return_token_scores` - If true, also returns per-document token similarity matrices.
/// * `batch_size` - Manual docs per scoring chunk; 0 plans chunks from the memory budget.
/// * `memory_budget_bytes` - Per-device scoring workspace budget.
///
/// # Returns
/// A `Result` containing a tuple of: the top `k` passage IDs (`Vec<i64>`), their
/// corresponding final scores (`Vec<f32>`), and optionally a list of token similarity
/// matrices (each of shape `[query_tokens, doc_tokens]`).
#[allow(clippy::too_many_arguments)]
pub fn search(
    query_embeddings: &Tensor,
    ivf_index_strided: &StridedTensor,
    codec: &ResidualCodec,
    embedding_dimension: i64,
    doc_codes_strided: &StridedTensor,
    doc_residuals_strided: &StridedTensor,
    n_ivf_probe: i64,
    n_docs_for_full_score: i64,
    nbits_param: i64,
    top_k: usize,
    device: Device,
    subset: Option<&Tensor>,
    return_token_scores: bool,
    batch_size: i64,
    memory_budget_bytes: i64,
) -> anyhow::Result<(Vec<i64>, Vec<f32>, Option<Vec<Tensor>>)> {
    let (passage_ids, scores, token_matrices) = tch::no_grad(|| {
        let q_tokens = query_embeddings.size()[0];

        let query_embeddings_unsqueezed = query_embeddings.unsqueeze(0);

        // Compute query-centroid scores
        let query_centroid_scores = codec.centroids.matmul(&query_embeddings.transpose(0, 1));

        // The centroid-score matrix stays alive through both stages; deduct it from the planner budget.
        let num_centroids = codec.centroids.size()[0];
        let centroid_scores_bytes = num_centroids * q_tokens * 2;
        let stage_budget =
            (memory_budget_bytes - centroid_scores_bytes).max(MIN_STAGE_BUDGET_BYTES);
        // Manual chunking preserves the input order; auto mode sorts candidates by doc length.
        let sort_enabled = batch_size <= 0;

        // Select IVF cells to probe
        let flat_cells_to_probe = if let Some(subset_tensor) = subset {
            // Subset optimization: restrict to centroids containing subset documents
            let (subset_doc_codes, _) = doc_codes_strided.lookup(subset_tensor, device);

            if subset_doc_codes.numel() == 0 {
                Tensor::empty(&[0], (Kind::Int64, device))
            } else {
                let (unique_subset_centroids, _, _) = subset_doc_codes
                    .flatten(0, -1)
                    .unique_dim(0, true, false, false);

                let subset_scores = query_centroid_scores.index_select(0, &unique_subset_centroids);
                let available_centroids = unique_subset_centroids.size()[0];
                let actual_k = n_ivf_probe.min(available_centroids);

                let top_indices_local = if actual_k == 1 {
                    subset_scores.argmax(0, true)
                } else {
                    subset_scores.topk(actual_k, 0, true, false).1
                };

                let flat_local_indices = top_indices_local.flatten(0, -1);
                unique_subset_centroids.index_select(0, &flat_local_indices)
            }
        } else {
            // Standard path
            let selected_ivf_cells_indices = if n_ivf_probe == 1 {
                query_centroid_scores.argmax(0, true).permute(&[1, 0])
            } else {
                query_centroid_scores
                    .topk(n_ivf_probe, 0, true, false)
                    .1
                    .permute(&[1, 0])
            };
            selected_ivf_cells_indices.flatten(0, -1).contiguous()
        };

        let (unique_ivf_cells_to_probe, _, _) =
            flat_cells_to_probe.unique_dim(-1, true, false, false);

        // Retrieve candidate documents via IVF lookup
        let (retrieved_passage_ids_ivf, _) =
            ivf_index_strided.lookup(&unique_ivf_cells_to_probe, device);

        let (sorted_passage_ids_ivf, _) = retrieved_passage_ids_ivf.sort(0, false);

        let (mut unique_passage_ids, _, _) =
            sorted_passage_ids_ivf.unique_consecutive(false, false, 0);

        // Filter to subset if provided
        if let Some(subset_tensor) = subset {
            unique_passage_ids =
                filter_passage_ids_with_subset(&unique_passage_ids, subset_tensor, device);
        }

        if unique_passage_ids.numel() == 0 {
            return Ok((vec![], vec![], None));
        }

        // Candidate doc lengths drive the chunk planner for both stages.
        let candidate_lengths = doc_codes_strided
            .element_lengths
            .index_select(
                0,
                &unique_passage_ids.to_device(doc_codes_strided.element_lengths.device()),
            )
            .to_device(Device::Cpu);
        let candidate_lengths_cpu: Vec<i64> = candidate_lengths.try_into()?;

        // Approximate scoring using coarse centroids
        let mut approx_chunk_fn = |batch_ids: &Tensor| -> Result<Tensor> {
            let (batch_packed_codes, batch_doc_lengths) =
                doc_codes_strided.lookup(batch_ids, device);

            if batch_packed_codes.numel() == 0 {
                return Ok(Tensor::zeros(&[batch_ids.size()[0]], (Kind::Float, device)));
            }

            let batch_approx_scores = query_centroid_scores.index_select(0, &batch_packed_codes);

            let (padded_approx_scores, mask) =
                direct_pad_sequences(&batch_approx_scores, &batch_doc_lengths, 0.0, device)?;

            Ok(colbert_score_reduce(padded_approx_scores, &mask))
        };

        let approx_scores = run_scoring_stage_with_oom_retry(
            &unique_passage_ids,
            &candidate_lengths_cpu,
            approx_bytes_per_cell(q_tokens),
            stage_budget,
            batch_size,
            doc_codes_strided,
            device,
            sort_enabled,
            &mut approx_chunk_fn,
        )?;

        if approx_scores.size().get(0) != Some(&unique_passage_ids.size()[0]) {
            return Err(anyhow!(
                "PID ({}) and approx scores ({}) count mismatch.",
                unique_passage_ids.size()[0],
                approx_scores.size().get(0).unwrap_or(&-1),
            ));
        }

        let mut passage_ids_to_rerank = unique_passage_ids;

        // Prune to n_full_scores, then to the n/4 decompression set.
        let n_passage_ids_for_decompression = (n_docs_for_full_score / 4).max(1);
        if n_docs_for_full_score < approx_scores.size()[0] && approx_scores.numel() > 0 {
            let (_, top_indices) = approx_scores.topk(n_docs_for_full_score, 0, true, true);

            passage_ids_to_rerank = passage_ids_to_rerank.index_select(0, &top_indices);

            // top_indices is sorted by descending score, so the n/4 selection is a narrow.
            if n_passage_ids_for_decompression < n_docs_for_full_score {
                passage_ids_to_rerank =
                    passage_ids_to_rerank.narrow(0, 0, n_passage_ids_for_decompression);
            }
        } else if n_passage_ids_for_decompression < approx_scores.size()[0]
            && approx_scores.numel() > 0
        {
            let (_, top_indices) =
                approx_scores.topk(n_passage_ids_for_decompression, 0, true, true);
            passage_ids_to_rerank = passage_ids_to_rerank.index_select(0, &top_indices);
        }

        if passage_ids_to_rerank.numel() == 0 {
            return Ok((vec![], vec![], None));
        }

        let bucket_weights = codec
            .bucket_weights
            .as_ref()
            .ok_or_else(|| anyhow!("Codec missing bucket_weights for decompression."))?;
        let bucket_weight_indices_lookup =
            codec.bucket_weight_indices_lookup.as_ref().ok_or_else(|| {
                anyhow!("Codec missing bucket_weight_indices_lookup for decompression.")
            })?;

        // Full decompression and exact scoring, planned like the approximate stage.
        let rerank_lengths = doc_codes_strided
            .element_lengths
            .index_select(
                0,
                &passage_ids_to_rerank.to_device(doc_codes_strided.element_lengths.device()),
            )
            .to_device(Device::Cpu);
        let rerank_lengths_cpu: Vec<i64> = rerank_lengths.try_into()?;

        let mut exact_chunk_fn = |batch_ids: &Tensor| -> Result<Tensor> {
            let (batch_codes, batch_doc_lengths) = doc_codes_strided.lookup(batch_ids, device);
            let (batch_residuals, _) = doc_residuals_strided.lookup(batch_ids, device);

            let decompressed_embeddings = decompress_residuals(
                &batch_residuals,
                bucket_weights,
                &codec.byte_reversed_bits_map,
                bucket_weight_indices_lookup,
                &batch_codes,
                &codec.centroids,
                embedding_dimension,
                nbits_param,
            );

            let (padded_doc_embeddings, mask) =
                direct_pad_sequences(&decompressed_embeddings, &batch_doc_lengths, 0.0, device)?;

            let token_scores_3d =
                padded_doc_embeddings.matmul(&query_embeddings_unsqueezed.transpose(-2, -1));

            Ok(colbert_score_reduce(token_scores_3d, &mask))
        };

        let reduced_scores = run_scoring_stage_with_oom_retry(
            &passage_ids_to_rerank,
            &rerank_lengths_cpu,
            exact_bytes_per_cell(q_tokens),
            stage_budget,
            batch_size,
            doc_codes_strided,
            device,
            sort_enabled,
            &mut exact_chunk_fn,
        )?;

        // Final top-k sort
        let (reduced_scores, sorted_indices) = reduced_scores.sort(0, true);

        let sorted_passage_ids = passage_ids_to_rerank.index_select(0, &sorted_indices);

        let sorted_passage_ids: Vec<i64> = sorted_passage_ids.try_into()?;
        let reduced_scores: Vec<f32> = reduced_scores.try_into()?;

        let result_count = top_k.min(sorted_passage_ids.len());
        let result_passage_ids = sorted_passage_ids[..result_count].to_vec();
        let result_scores = reduced_scores[..result_count].to_vec();

        let token_matrices = if return_token_scores {
            let result_passage_ids_tensor = Tensor::from_slice(&result_passage_ids)
                .to_device(device)
                .to_kind(Kind::Int64);
            let (result_codes, result_doc_lengths) =
                doc_codes_strided.lookup(&result_passage_ids_tensor, device);
            let (result_residuals, _) =
                doc_residuals_strided.lookup(&result_passage_ids_tensor, device);

            let result_embeddings = decompress_residuals(
                &result_residuals,
                bucket_weights,
                &codec.byte_reversed_bits_map,
                bucket_weight_indices_lookup,
                &result_codes,
                &codec.centroids,
                embedding_dimension,
                nbits_param,
            );

            let (padded_doc_embeddings, _) =
                direct_pad_sequences(&result_embeddings, &result_doc_lengths, 0.0, device)?;
            let sorted_token_scores =
                padded_doc_embeddings.matmul(&query_embeddings_unsqueezed.transpose(-2, -1));
            let sorted_doc_lengths: Vec<i64> = result_doc_lengths.try_into()?;

            let mut matrices = Vec::with_capacity(result_count);
            for i in 0..result_count {
                let doc_len = sorted_doc_lengths[i];
                // [max_doc_len, query_len] → [doc_len, query_len] → [query_len, doc_len]
                let mat = sorted_token_scores
                    .i(i as i64)
                    .narrow(0, 0, doc_len)
                    .transpose(0, 1);
                matrices.push(mat);
            }
            Some(matrices)
        } else {
            None
        };

        Ok((result_passage_ids, result_scores, token_matrices))
    })?;

    Ok((passage_ids, scores, token_matrices))
}
