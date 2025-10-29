use itertools::Itertools;
use std::iter;
use tch::{Device, Kind, Tensor};

/// A codec for managing residual quantization of vectors.
///
/// This structure holds the necessary tensors and pre-computed lookup tables
/// for encoding and decoding vectors based on a residual quantization scheme.
/// It is initialized once with the quantization parameters and then used
/// for efficient data transformation.
pub struct ResidualCodec {
    /// The number of bits used for quantization.
    pub nbits: i64,
    /// The base centroids for quantization.
    pub centroids: Tensor,
    /// The average residual vector, added back during decoding.
    pub avg_residual: Tensor,
    /// Optional tensor defining the cutoff values for quantization buckets.
    pub bucket_cutoffs: Option<Tensor>,
    /// Optional tensor containing weights for each quantization bucket.
    pub bucket_weights: Option<Tensor>,
    /// A helper tensor for bit manipulation, typically an arange tensor.
    pub bit_helper: Tensor,
    /// A pre-computed lookup table to reverse the order of `nbits` segments within a byte.
    pub byte_reversed_bits_map: Tensor,
    /// An optional lookup table for decomposing compressed byte values into bucket indices.
    pub bucket_weight_indices_lookup: Option<Tensor>,
}

impl ResidualCodec {
    /// Creates and initializes a new `ResidualCodec`.
    ///
    /// This function sets up the codec by moving the provided tensors to the specified
    /// compute device and pre-calculating helper tensors and lookup tables used
    /// during the encoding and decoding process.
    ///
    /// # Arguments
    ///
    /// * `nbits_param` - The number of bits for quantization.
    /// * `centroids_tensor_initial` - The initial tensor of centroids.
    /// * `avg_residual_tensor_initial` - The initial tensor for the average residual.
    /// * `bucket_cutoffs_tensor_initial` - An optional tensor for bucket cutoffs.
    /// * `bucket_weights_tensor_initial` - An optional tensor for bucket weights.
    /// * `device` - The `tch::Device` to which all tensors will be moved.
    ///
    /// # Returns
    ///
    /// A `Result` containing the fully initialized `ResidualCodec` on success.
    pub fn load(
        nbits_param: i64,
        centroids_tensor_initial: Tensor,
        avg_residual_tensor_initial: Tensor,
        bucket_cutoffs_tensor_initial: Option<Tensor>,
        bucket_weights_tensor_initial: Option<Tensor>,
        device: Device,
    ) -> anyhow::Result<Self> {
        let bit_helper_tensor = Tensor::arange_start(0, nbits_param, (Kind::Int8, device));
        let mut reversed_bits_map_u8 = Vec::with_capacity(256);
        let nbits_mask = (1 << nbits_param) - 1;

        for byte_val in 0..256u32 {
            let mut reversed_bits = 0u32;
            let mut bit_pos = 8;
            while bit_pos >= nbits_param {
                let nbits_segment = (byte_val >> (bit_pos - nbits_param)) & nbits_mask;
                let mut reversed_segment = 0u32;
                for k in 0..nbits_param {
                    if (nbits_segment & (1 << k)) != 0 {
                        reversed_segment |= 1 << (nbits_param - 1 - k);
                    }
                }
                reversed_bits |= reversed_segment;
                if bit_pos > nbits_param {
                    reversed_bits <<= nbits_param;
                }
                bit_pos -= nbits_param;
            }
            reversed_bits_map_u8.push((reversed_bits & 0xFF) as u8);
        }

        let byte_map_tensor = Tensor::from_slice(
            &reversed_bits_map_u8
                .iter()
                .map(|&val| val as i64)
                .collect::<Vec<_>>(),
        )
        .to_kind(Kind::Int64)
        .to_device(device);

        let keys_per_byte = 8 / nbits_param;
        let opt_bucket_weight_indices_lookup_table =
            if let Some(ref weights) = bucket_weights_tensor_initial {
                let num_buckets = weights.size()[0] as usize;
                let bucket_indices = (0..num_buckets as i64).collect::<Vec<_>>();

                let combinations = iter::repeat(bucket_indices)
                    .take(keys_per_byte as usize)
                    .multi_cartesian_product()
                    .flatten()
                    .collect::<Vec<_>>();

                let lookup_shape = vec![
                    (num_buckets as i64).pow(keys_per_byte as u32),
                    keys_per_byte,
                ];

                Some(
                    Tensor::from_slice(&combinations)
                        .reshape(&lookup_shape)
                        .to_kind(Kind::Int64)
                        .to_device(device),
                )
            } else {
                None
            };

        Ok(Self {
            nbits: nbits_param,
            centroids: centroids_tensor_initial,
            avg_residual: avg_residual_tensor_initial,
            bucket_cutoffs: bucket_cutoffs_tensor_initial,
            bucket_weights: bucket_weights_tensor_initial,
            bit_helper: bit_helper_tensor,
            byte_reversed_bits_map: byte_map_tensor,
            bucket_weight_indices_lookup: opt_bucket_weight_indices_lookup_table,
        })
    }
}
