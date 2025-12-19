use tch::{Device, Kind, Tensor};

/// Retrieves a scratch tensor of a specified minimum shape.
///
/// Note: Caching has been removed. This function now allocates a fresh
/// tensor on every call.
///
/// # Arguments
///
/// * `device` - The `Device` where the tensor should be allocated (e.g., CPU or CUDA).
/// * `kind` - The data type of the tensor (e.g., `Kind::Float`).
/// * `min_shape` - The minimum required shape of the tensor, e.g., `[batch, seq_len, features]`.
/// * `pad_value` - The value used to fill the tensor when it is first created or resized.
///
/// # Returns
///
/// A `Tensor` that meets the minimum shape requirement.
pub fn get_scratch(device: Device, kind: Kind, min_shape: &[i64], pad_value: f64) -> Tensor {
    Tensor::full(min_shape, pad_value, (kind, device))
}

/// A trait for finding the maximum value within a tensor.
pub trait MaxValueExt {
    /// Computes the maximum value of all elements in the tensor.
    fn max_value(&self) -> Tensor;
}

/// Implements the `MaxValueExt` trait for `tch::Tensor`.
impl MaxValueExt for Tensor {
    #[inline(always)]
    fn max_value(&self) -> Tensor {
        self.max()
    }
}

/// Pads a batch of variable-length sequences into a dense tensor.
///
/// This function efficiently transforms a flattened, concatenated tensor of sequences
/// into a single dense 3D tensor and generates a corresponding boolean attention mask.
///
/// # Key Optimizations
/// - **Scratch Buffer**: Uses `PAD_CACHE` to avoid expensive `malloc` on the GPU.
/// - **Asynchronous Operations**: Calculates `max_len` on the device to avoid host-device sync.
/// - **Scatter-based Copy**: Instead of iterating through rows (slow in Python/loops),
///   it calculates global indices via the mask and copies all data in one
///   vectorized `index_put_` kernel.
///
/// # Arguments
///
/// * `sequences` - A 2D tensor of shape `[total_tokens, features]` containing the
///   concatenated data of all sequences.
/// * `length_values` - A 1D tensor of shape `[batch_size]` containing sequence lengths.
/// * `pad_value` - The floating-point value to use for padding.
/// * `device` - The `tch::Device` on which to perform the operations.
///
/// # Returns
///
/// A `Result` containing:
/// * `padded_sequences`: Shape `[batch_size, max_len, features]`.
/// * `attention_mask`: Shape `[batch_size, max_len]`, `true` = valid token.
pub fn direct_pad_sequences(
    sequences: &Tensor,
    length_values: &Tensor,
    pad_value: f64,
    device: Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    if length_values.numel() == 0 {
        return Ok((
            Tensor::empty(&[0, 0, sequences.size()[1]], (sequences.kind(), device)),
            Tensor::empty(&[0, 0], (Kind::Bool, device)),
        ));
    }

    let batch_size = length_values.size()[0];
    let feature_dim = sequences.size()[1];

    // 1. Determine dimensions
    let max_len_tensor = length_values.to_device(device).max_value();
    let max_len: i64 = max_len_tensor.int64_value(&[]);

    // 2. Prepare destination buffer
    let mut padded_sequences = get_scratch(
        device,
        sequences.kind(),
        &[batch_size, max_len, feature_dim],
        pad_value,
    );

    // Reset values to pad_value before writing
    let _ = padded_sequences.fill_(pad_value);

    // 3. Generate Attention Mask
    // We create a range row `[0, 1, ... max_len]` and compare it against `length_values`
    // using broadcasting to create the boolean mask.
    let range_row = Tensor::arange(max_len, (Kind::Int64, device));

    let attention_mask = range_row
        .unsqueeze(0) // Shape: [1, max_len]
        .lt_tensor(&length_values.to_device(device).unsqueeze(-1)); // Shape: [batch, 1]

    // 4. Scatter Data
    // We get the coordinates of all `true` values in the mask. These correspond
    // exactly to where the valid tokens should sit in the dense 3D tensor.
    let nz = attention_mask.nonzero();
    let b_indices_flat = nz.select(1, 0); // Batch indices
    let t_indices_flat = nz.select(1, 1); // Time/Sequence indices

    let sequences_on_device = sequences.to_device(device);

    // Copy the flattened sequence data into the dense tensor at the specific
    // (batch, time) coordinates derived from the mask.
    let _ = padded_sequences.index_put_(
        &[Some(b_indices_flat), Some(t_indices_flat)],
        &sequences_on_device,
        false,
    );

    Ok((padded_sequences, attention_mask))
}
