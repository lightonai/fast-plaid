use anyhow::anyhow;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use rayon::prelude::*;
use tch::{Kind, Tensor};

use crate::search::load::{get_device, PyLoadedIndex};
use crate::search::search::decompress_residuals;
use crate::utils::errors::anyhow_to_pyerr;

#[pyfunction]
pub fn reconstruct_embeddings(
    py: Python<'_>,
    index: &PyLoadedIndex,
    subset: Vec<i64>,
    device: String,
) -> PyResult<Vec<PyTensor>> {
    let device = get_device(&device)?;
    let inner = &index.inner;

    let tensors: Vec<Tensor> = py
        .allow_threads(move || {
            subset
                .into_par_iter()
                .map(|doc_id| {
                    let centroids = &inner.codec.centroids;
                    let bucket_weights = inner
                        .codec
                        .bucket_weights
                        .as_ref()
                        .ok_or_else(|| anyhow!("Index is missing bucket weights"))?;
                    let bucket_weight_indices_lookup = inner
                        .codec
                        .bucket_weight_indices_lookup
                        .as_ref()
                        .ok_or_else(|| anyhow!("Index is missing bucket weight indices lookup"))?;
                    let byte_reversed_bits_map = &inner.codec.byte_reversed_bits_map;
                    let embedding_dim = centroids.size()[1];

                    let id_tensor = Tensor::from_slice(&[doc_id]).to_device(device);
                    let (doc_codes, _) = inner.doc_codes_strided.lookup(&id_tensor, device);

                    if doc_codes.size()[0] == 0 {
                        return Ok(Tensor::empty(&[0, embedding_dim], (Kind::Float, device)));
                    }

                    let (doc_residuals, _) = inner.doc_residuals_strided.lookup(&id_tensor, device);

                    let reconstructed = decompress_residuals(
                        &doc_residuals,
                        bucket_weights,
                        byte_reversed_bits_map,
                        bucket_weight_indices_lookup,
                        &doc_codes,
                        centroids,
                        embedding_dim,
                        inner.nbits,
                    );

                    Ok(reconstructed.to_kind(Kind::Float))
                })
                .collect::<Result<Vec<Tensor>, anyhow::Error>>()
        })
        .map_err(anyhow_to_pyerr)?;

    let output_list = tensors.into_iter().map(PyTensor).collect();

    Ok(output_list)
}
