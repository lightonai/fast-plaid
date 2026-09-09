//! Asymmetric int8-query x fused-LUT MaxSim over stored residual codes.
//!
//! The float exact-scoring path reconstructs every candidate token to `f32`,
//! pads the batch to a rectangle and runs a GEMM against the query. Most of
//! that work is spent producing bytes that are read once: on SciFact the
//! reconstruction alone is ~60% of a CPU search.
//!
//! This path scores the *stored* codes instead. Each packed byte expands to
//! its bucket weights through an in-register table lookup, the dot runs on
//! integer MACs, and the centroid term is the query-by-centroid product the
//! candidate-generation stage has already computed. Nothing is decompressed
//! and nothing is padded.
//!
//! It serves whenever the codes and the residuals both live in host memory --
//! the CPU device, and the `low` placement tier, which exists to keep those
//! bytes off the GPU -- so they are read where they already are. Under
//! `medium` the codes are device-resident, and under `high` so are the
//! residuals; both decline here and the float path serves them unchanged.
//!
//! Scoring is quantized, so scores are close to the float path's rather than
//! equal to them; see `tests/test_asym.py` for the measured agreement.

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
        // The scorer reads both of these in place, so both must be on the
        // host. Under the `medium` tier the codes are device-resident while
        // the residuals are not, which is a decline rather than an error: the
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

    /// Exact scores for `passage_ids`, in that order.
    ///
    /// `query_centroid_scores` is the `[num_centroids, query_tokens]` product
    /// the candidate stage already computed; the kernels read it as the
    /// centroid term rather than recomputing it.
    pub fn score(
        &self,
        passage_ids: &Tensor,
        query_embeddings: &Tensor,
        query_centroid_scores: &Tensor,
        doc_residuals: &StridedTensor,
        doc_codes: &StridedTensor,
        device: Device,
    ) -> Result<Tensor> {
        let n_query_tokens = query_embeddings.size()[0];
        let query: Vec<f32> = query_embeddings
            .to_kind(Kind::Float)
            .to_device(Device::Cpu)
            .contiguous()
            .reshape([-1])
            .try_into()?;
        let prepared = PreparedQuery::new(&self.lut, &query, n_query_tokens as usize, self.dim)
            .map_err(|error| anyhow!("query preparation failed: {error}"))?;

        let centroid_scores: Vec<f32> = query_centroid_scores
            .to_kind(Kind::Float)
            .to_device(Device::Cpu)
            .contiguous()
            .reshape([-1])
            .try_into()?;
        let num_centroids = query_centroid_scores.size()[0] as usize;
        let scorer = Scorer::new(&self.lut, &prepared)
            .with_centroid_term(&centroid_scores, num_centroids)
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
