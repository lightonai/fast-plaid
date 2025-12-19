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
        let bit_helper_tensor = Tensor::arange_start(0, nbits_param, (Kind::Int8, device));

        // Build bit reversal map for unpacking
        let mut reversed_bits_map_u8 = vec![0u8; 256];
        let nbits_mask = (1 << nbits_param) - 1;

        for i in 0..256usize {
            let val = i as u32;
            let mut out = 0u32;
            let mut pos = 8;

            while pos >= nbits_param {
                let segment = (val >> (pos - nbits_param)) & nbits_mask;

                let mut rev_segment = 0;
                for k in 0..nbits_param {
                    if (segment & (1 << k)) != 0 {
                        rev_segment |= 1 << (nbits_param - 1 - k);
                    }
                }

                out |= rev_segment;

                if pos > nbits_param {
                    out <<= nbits_param;
                }

                pos -= nbits_param;
            }
            reversed_bits_map_u8[i] = out as u8;
        }

        let byte_map_tensor = Tensor::from_slice(&reversed_bits_map_u8)
            .to_kind(Kind::Uint8)
            .to_device(device);

        // Build lookup table for bucket weight indices
        let keys_per_byte = (8 / nbits_param) as usize;

        let opt_bucket_weight_indices_lookup_table =
            if let Some(ref _weights) = bucket_weights_tensor_initial {
                let mask = (1 << nbits_param) - 1;
                let mut table_data = Vec::with_capacity(256 * keys_per_byte);

                for byte_val in 0..256 {
                    for k in (0..keys_per_byte).rev() {
                        let shift = k as i64 * nbits_param;
                        let index = (byte_val >> shift) & mask;
                        table_data.push(index);
                    }
                }

                Some(
                    Tensor::from_slice(&table_data)
                        .reshape(&[256, keys_per_byte as i64])
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
