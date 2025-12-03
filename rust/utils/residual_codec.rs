use itertools::Itertools;
use std::iter;
use tch::{Device, Kind, Tensor};

/// A codec that manages the quantization parameters and lookup tables for the index.
///
/// This struct acts as a container for all the read-only tensors required to
/// decompress vectors during a search. To maximize performance on GPUs, bitwise
/// unpacking operations are replaced by **pre-computed lookup tables**.
///
///
///
/// Instead of performing bit-shifts and masking for every vector on the fly,
/// this codec pre-calculates the results for every possible byte value (0-255).
/// During search, unpacking a compressed vector becomes a fast memory lookup
/// (gather) operation.
pub struct ResidualCodec {
    /// The number of bits used to represent each residual bucket (e.g., 2 or 4).
    pub nbits: i64,
    /// The coarse centroids (codebook) of shape `[num_centroids, dim]`.
    pub centroids: Tensor,
    /// The average residual vector, added to reconstructed vectors to reduce error.
    pub avg_residual: Tensor,
    /// The boundaries defining which bucket a residual value falls into.
    pub bucket_cutoffs: Option<Tensor>,
    /// The actual values (weights) corresponding to each quantization bucket.
    pub bucket_weights: Option<Tensor>,
    /// A small helper tensor `[0, 1, ... nbits-1]` used for bitwise expansions.
    pub bit_helper: Tensor,
    /// A lookup table (256 entries) used to handle bit-endianness or internal
    /// bit ordering differences during unpacking.
    pub byte_reversed_bits_map: Tensor,
    /// The primary decompression table. It maps a byte value (0-255) directly
    /// to a sequence of bucket indices, avoiding runtime bit-shifting.
    pub bucket_weight_indices_lookup: Option<Tensor>,
}

impl Clone for ResidualCodec {
    fn clone(&self) -> Self {
        Self {
            nbits: self.nbits,
            // tch::Tensor::shallow_clone() creates a new Tensor object sharing the
            // same underlying storage, which is efficient for this read-only struct.
            centroids: self.centroids.shallow_clone(),
            avg_residual: self.avg_residual.shallow_clone(),
            bucket_cutoffs: self.bucket_cutoffs.as_ref().map(|t| t.shallow_clone()),
            bucket_weights: self.bucket_weights.as_ref().map(|t| t.shallow_clone()),
            bit_helper: self.bit_helper.shallow_clone(),
            byte_reversed_bits_map: self.byte_reversed_bits_map.shallow_clone(),
            bucket_weight_indices_lookup: self
                .bucket_weight_indices_lookup
                .as_ref()
                .map(|t| t.shallow_clone()),
        }
    }
}

impl ResidualCodec {
    /// Initializes the codec and pre-computes acceleration lookup tables.
    ///
    /// This function moves the provided tensors to the target device and generates
    /// the `bucket_weight_indices_lookup` table. This table generation involves
    /// calculating the cartesian product of all possible bucket combinations that
    /// can fit into a single byte.
    ///
    /// # Arguments
    ///
    /// * `nbits_param` - The number of bits per code (e.g., 2 bits = 4 buckets).
    /// * `centroids_tensor_initial` - The coarse centroids.
    /// * `avg_residual_tensor_initial` - The global average residual.
    /// * `bucket_cutoffs_tensor_initial` - Boundaries for quantization (used in indexing/update).
    /// * `bucket_weights_tensor_initial` - Values for reconstruction (used in search).
    /// * `device` - The `tch::Device` to store the tables on.
    pub fn load(
        nbits_param: i64,
        centroids_tensor_initial: Tensor,
        avg_residual_tensor_initial: Tensor,
        bucket_cutoffs_tensor_initial: Option<Tensor>,
        bucket_weights_tensor_initial: Option<Tensor>,
        device: Device,
    ) -> anyhow::Result<Self> {
        // 1. Create Bit Helper
        // Used for parallel bit extraction in some update routines.
        let bit_helper_tensor = Tensor::arange_start(0, nbits_param, (Kind::Int8, device));

        // 2. Generate Bit Reversal Map (0..255)
        // This handles potential endianness mismatches or specific packing formats
        // by reversing the bits *within* each n-bit segment of a byte.
        let mut reversed_bits_map_u8 = Vec::with_capacity(256);
        let nbits_mask = (1 << nbits_param) - 1;

        for byte_val in 0..256u32 {
            let mut reversed_bits = 0u32;
            let mut bit_pos = 8;

            // Iterate through the byte in chunks of `nbits_param`
            while bit_pos >= nbits_param {
                let nbits_segment = (byte_val >> (bit_pos - nbits_param)) & nbits_mask;
                let mut reversed_segment = 0u32;

                // Reverse the bits inside the segment
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
        .to_kind(Kind::Uint8)
        .to_device(device);

        // 3. Generate Decompression Lookup Table
        // If we have weights (search mode), we pre-calculate the expansion of every
        // possible byte into its constituent bucket indices.
        //
        // Example (nbits=2):
        // A byte holds 4 codes. There are 4 possible buckets (0-3).
        // We generate a table of size [256, 4].
        // Entry 0 (0b00000000) -> [0, 0, 0, 0]
        // Entry 255 (0b11111111) -> [3, 3, 3, 3]
        let keys_per_byte = 8 / nbits_param;
        let opt_bucket_weight_indices_lookup_table =
            if let Some(ref weights) = bucket_weights_tensor_initial {
                let num_buckets = weights.size()[0] as usize;
                let bucket_indices = (0..num_buckets as i64).collect::<Vec<_>>();

                // Generate Cartesian product: bucket_indices ^ keys_per_byte
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
