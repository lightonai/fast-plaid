//! Scoring candidates from the index's stored codes, read in place.
//!
//! Both scoring stages of a search gather their candidates into a padded
//! `[docs, max_doc_len, ...]` rectangle and reduce it. The rectangle is the
//! expensive part and it is read exactly once: on SciFact the approximate
//! stage's gather and the exact stage's `f32` reconstruction are together ~70%
//! of a CPU search, to produce one number per candidate.
//!
//! This module computes the same two reductions by walking each candidate's
//! own rows where the index already holds them. Nothing is gathered and
//! nothing is padded.
//!
//! * The approximate stage keeps a running per-query-token maximum over the
//!   centroid scores the search has already computed. It is exact: identical
//!   results to the padded path.
//! * The exact stage hands the packed rows to `maxsim-lut`, which expands each
//!   byte to its bucket weights by an in-register table lookup and dots it
//!   against an int8-quantized query on integer MACs. The reconstruction never
//!   happens, so scores are quantized rather than equal to the float path's --
//!   `tests/test_asym.py` pins the agreement.
//!
//! Both need the codes and the residuals in host memory, which the CPU device
//! and the `low` placement tier give. Under `medium` the codes are
//! device-resident, and under `high` so are the residuals; both decline here
//! and the float path serves them unchanged.

use anyhow::{anyhow, Result};
use maxsim_lut::{Codes, DocView, Lut, PreparedQuery, Scorer, MAX_DIM};
use rayon::prelude::*;
use tch::{Device, Kind, Tensor};

use crate::search::search::{reconstruct_residuals, reconstruction_norms};
use crate::search::tensor::StridedTensor;
use crate::utils::residual_codec::ResidualCodec;

/// Tokens reconstructed per chunk while building the inverse norms. Bounds the
/// transient of a one-off pass whose cost is bandwidth, not arithmetic.
const NORM_CHUNK_TOKENS: i64 = 65_536;

/// Per-index tables for asymmetric scoring, built once and shared by every
/// search against that index.
///
/// The inverse norms are the only allocation: 4 bytes per token, against the
/// 4 x `dim` bytes per token the float path materializes for each candidate of
/// each query.
pub struct AsymIndex {
    lut: Lut,
    /// `1 / ||centroid[code] + bucket_weights[residual]||` per token, in the
    /// index's own token order.
    inv_norms: Vec<f32>,
    /// Token offset of each document into the residual storage.
    offsets: Vec<i64>,
    /// Token count of each document.
    lengths: Vec<i64>,
    /// Packed residual bytes per token.
    row_stride: usize,
    dim: usize,
}

impl AsymIndex {
    /// Builds the tables, or explains why this index cannot use them.
    ///
    /// Every rejection is a property of the index or the build, never of a
    /// particular query, so a refusal here is final for the process.
    pub fn build(
        codec: &ResidualCodec,
        doc_codes: &StridedTensor,
        doc_residuals: &StridedTensor,
        nbits: i64,
    ) -> Result<Self> {
        // Both stages read these two in place, so both must be on the host.
        // Under the `medium` tier the codes are device-resident while the
        // residuals are not, which is a decline rather than an error: the
        // float path serves that placement exactly as before.
        for (name, tensor) in [
            ("codes", &doc_codes.underlying_data),
            ("residuals", &doc_residuals.underlying_data),
        ] {
            if tensor.device() != Device::Cpu {
                return Err(anyhow!(
                    "{name} are on {:?}, not host memory",
                    tensor.device()
                ));
            }
        }
        let bucket_weights = codec
            .bucket_weights
            .as_ref()
            .ok_or_else(|| anyhow!("codec has no bucket weights"))?;
        let lookup = codec
            .bucket_weight_indices_lookup
            .as_ref()
            .ok_or_else(|| anyhow!("codec has no bucket weight index lookup"))?;

        let dim = codec.centroids.size()[1];
        if dim as usize > MAX_DIM {
            return Err(anyhow!("dim {dim} exceeds the kernel maximum {MAX_DIM}"));
        }

        let weights: Vec<f32> = bucket_weights
            .to_kind(Kind::Float)
            .to_device(Device::Cpu)
            .try_into()?;
        let lut = Lut::colbert(nbits as usize, &weights)
            .map_err(|error| anyhow!("unsupported residual layout: {error}"))?;

        let row_stride = doc_residuals.inner_dims.first().copied().unwrap_or(0);
        if row_stride != (dim * nbits) / 8 {
            return Err(anyhow!(
                "residual row stride {row_stride} does not match dim {dim} at {nbits} bits"
            ));
        }

        let inv_norms = build_inverse_norms(codec, doc_codes, doc_residuals, lookup, dim, nbits)?;

        Ok(Self {
            lut,
            inv_norms,
            offsets: (&doc_codes.cumulative_lengths.to_device(Device::Cpu)).try_into()?,
            lengths: (&doc_codes.element_lengths.to_device(Device::Cpu)).try_into()?,
            row_stride: row_stride as usize,
            dim: dim as usize,
        })
    }

    /// Quantizes the query and takes a host copy of the centroid scores, once
    /// for both stages.
    ///
    /// `query_centroid_scores` is the `[num_centroids, query_tokens]` product
    /// the search already computed; row-major, that is exactly the
    /// centroid-major layout both stages want.
    pub fn prepare(
        &self,
        query_embeddings: &Tensor,
        query_centroid_scores: &Tensor,
    ) -> Result<QueryTables> {
        let n_query_tokens = query_embeddings.size()[0] as usize;
        let query: Vec<f32> = query_embeddings
            .to_kind(Kind::Float)
            .to_device(Device::Cpu)
            .contiguous()
            .reshape([-1])
            .try_into()?;
        let prepared = PreparedQuery::new(&self.lut, &query, n_query_tokens, self.dim)
            .map_err(|error| anyhow!("query preparation failed: {error}"))?;

        Ok(QueryTables {
            prepared,
            centroid_scores: query_centroid_scores
                .to_kind(Kind::Float)
                .to_device(Device::Cpu)
                .contiguous()
                .reshape([-1])
                .try_into()?,
            num_centroids: query_centroid_scores.size()[0] as usize,
            n_query_tokens,
        })
    }

    /// Approximate scores for `passage_ids`, from centroid identity alone.
    ///
    /// The float path gathers one `[query_tokens]` row per candidate token
    /// into a `[docs, max_doc_len, query_tokens]` rectangle and reduces that.
    /// The rows are read straight out of the centroid-score matrix here, so
    /// nothing is gathered and nothing is padded; the running maximum is the
    /// only state, and it is `query_tokens` wide.
    pub fn approximate_scores(
        &self,
        tables: &QueryTables,
        passage_ids: &Tensor,
        doc_codes: &StridedTensor,
        device: Device,
    ) -> Result<Tensor> {
        let nq = tables.n_query_tokens;
        let cdot = &tables.centroid_scores;
        let ids: Vec<i64> = (&passage_ids.to_device(Device::Cpu)).try_into()?;
        let codes = as_slice::<i64>(&doc_codes.underlying_data, Kind::Int64)?;

        let scores: Vec<f32> = ids
            .par_iter()
            .map_init(
                || vec![f32::NEG_INFINITY; nq],
                |best, &id| {
                    let start = self.offsets[id as usize] as usize;
                    let len = self.lengths[id as usize] as usize;
                    if len == 0 {
                        return 0.0;
                    }
                    best.fill(f32::NEG_INFINITY);
                    for &code in &codes[start..start + len] {
                        let row = &cdot[code as usize * nq..][..nq];
                        for (slot, &value) in best.iter_mut().zip(row) {
                            if value > *slot {
                                *slot = value;
                            }
                        }
                    }
                    best.iter().sum()
                },
            )
            .collect();

        Ok(Tensor::from_slice(&scores).to_device(device))
    }

    /// Exact scores for `passage_ids`, in that order.
    pub fn exact_scores(
        &self,
        tables: &QueryTables,
        passage_ids: &Tensor,
        doc_residuals: &StridedTensor,
        doc_codes: &StridedTensor,
        device: Device,
    ) -> Result<Tensor> {
        let scorer = Scorer::new(&self.lut, &tables.prepared)
            .with_centroid_term(&tables.centroid_scores, tables.num_centroids)
            .map_err(|error| anyhow!("centroid term rejected: {error}"))?;

        let ids: Vec<i64> = (&passage_ids.to_device(Device::Cpu)).try_into()?;
        let packed = as_slice::<u8>(&doc_residuals.underlying_data, Kind::Uint8)?;
        let codes = as_slice::<i64>(&doc_codes.underlying_data, Kind::Int64)?;

        // One task per document: candidate lengths vary by an order of
        // magnitude, so fixed chunks straggle where rayon's adaptive split
        // does not.
        let scores: Vec<f32> = ids
            .par_iter()
            .map(|&id| {
                let start = self.offsets[id as usize] as usize;
                let len = self.lengths[id as usize] as usize;
                let bytes = start * self.row_stride;
                let doc = DocView::new(
                    &packed[bytes..bytes + len * self.row_stride],
                    len,
                    self.row_stride,
                )
                .codes(Codes::I64(&codes[start..start + len]))
                .inv_norms(&self.inv_norms[start..start + len]);
                scorer.score(doc)
            })
            .collect();

        Ok(Tensor::from_slice(&scores).to_device(device))
    }
}

/// The per-query state both scoring stages read: the int8-quantized query and
/// a host copy of the query-by-centroid scores.
///
/// Built once per search rather than once per stage, because the centroid
/// matrix is the larger of the two and both stages want the same copy of it.
pub struct QueryTables {
    prepared: PreparedQuery,
    centroid_scores: Vec<f32>,
    num_centroids: usize,
    n_query_tokens: usize,
}

/// Reconstructs the index once to record `1 / ||token||` per token.
///
/// The float path recomputes these for every candidate of every query; here
/// they are paid once and kept.
fn build_inverse_norms(
    codec: &ResidualCodec,
    doc_codes: &StridedTensor,
    doc_residuals: &StridedTensor,
    lookup: &Tensor,
    dim: i64,
    nbits: i64,
) -> Result<Vec<f32>> {
    let bucket_weights = codec
        .bucket_weights
        .as_ref()
        .ok_or_else(|| anyhow!("codec has no bucket weights"))?;
    let n_tokens = doc_residuals.underlying_data.size()[0];

    let mut inv_norms = Vec::with_capacity(n_tokens as usize);
    let mut start = 0;
    while start < n_tokens {
        let len = NORM_CHUNK_TOKENS.min(n_tokens - start);
        let embeddings = reconstruct_residuals(
            &doc_residuals.underlying_data.narrow(0, start, len),
            bucket_weights,
            &codec.byte_reversed_bits_map,
            lookup,
            &doc_codes.underlying_data.narrow(0, start, len),
            &codec.centroids,
            dim,
            nbits,
        );
        let chunk: Vec<f32> = reconstruction_norms(&embeddings)
            .reciprocal()
            .to_kind(Kind::Float)
            .reshape([-1])
            .try_into()?;
        inv_norms.extend_from_slice(&chunk);
        start += len;
    }
    Ok(inv_norms)
}

/// Borrows a contiguous CPU tensor's storage as a slice.
///
/// The index is immutable for as long as it is loaded, and the returned slice
/// borrows the tensor it is taken from, so the data outlives every use.
fn as_slice<T>(tensor: &Tensor, expected: Kind) -> Result<&[T]> {
    if tensor.device() != Device::Cpu {
        return Err(anyhow!("expected a CPU tensor, got {:?}", tensor.device()));
    }
    if tensor.kind() != expected {
        return Err(anyhow!("expected {expected:?}, got {:?}", tensor.kind()));
    }
    if !tensor.is_contiguous() {
        return Err(anyhow!("expected a contiguous tensor"));
    }
    let len = tensor.numel();
    // SAFETY: checked contiguous, on the host, and of the expected kind; the
    // slice borrows `tensor`, so the storage cannot be freed while it lives.
    Ok(unsafe { std::slice::from_raw_parts(tensor.data_ptr() as *const T, len) })
}
