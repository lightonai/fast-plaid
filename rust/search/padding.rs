use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use tch::{Device, Kind, Tensor};

/// A global, thread-safe cache for reusable scratch tensors.
///
/// This cache stores one tensor per `(Device, Kind)` combination. It is designed
/// to reduce the frequency of memory allocations by reusing and expanding tensors
/// as needed, which is particularly effective when dealing with variably sized
/// batches on devices like GPUs.
static PAD_CACHE: Lazy<Mutex<HashMap<(Device, Kind), Tensor>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Retrieves a scratch tensor of a specified minimum shape.
///
/// This function provides an efficient way to get a temporary tensor, amortizing
/// allocation costs by maintaining a global cache. If a cached tensor for the
/// given `device` and `kind` is available but too small, it is resized to fit
/// the `min_shape`. If no tensor is cached, a new one is created.
///
/// The returned tensor is a shallow clone and may be larger than `min_shape`.
/// The caller is responsible for slicing it to the exact required dimensions
/// (e.g., using `narrow`).
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
    let mut map = PAD_CACHE.lock();
    match map.entry((device, kind)) {
        Entry::Occupied(mut e) => {
            let t = e.get_mut();
            if t.size()[0] < min_shape[0]
                || t.size()[1] < min_shape[1]
                || t.size()[2] < min_shape[2]
            {
                let new_shape = [
                    t.size()[0].max(min_shape[0]),
                    t.size()[1].max(min_shape[1]),
                    t.size()[2].max(min_shape[2]),
                ];
                *t = Tensor::full(&new_shape, pad_value, (kind, device));
            }
            t.shallow_clone()
        },
        Entry::Vacant(v) => {
            let t = Tensor::full(min_shape, pad_value, (kind, device));
            v.insert(t.shallow_clone());
            t
        },
    }
}

/// A trait for finding the maximum value within a tensor.
///
/// This extension trait provides a `max_value` method to simplify the
/// process of finding the single largest value across all elements
/// of a `Tensor`.
pub trait MaxValueExt {
    /// Computes the maximum value of all elements in the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` containing a single element, which is the maximum
    /// value from the original tensor.
    fn max_value(&self) -> Tensor;
}

/// Implements the `MaxValueExt` trait for `tch::Tensor`.
impl MaxValueExt for Tensor {
    /// This implementation simply calls the built-in `max()` method.
    /// It is marked with `#[inline(always)]` to encourage the compiler
    /// to make it a zero-cost abstraction.
    #[inline(always)]
    fn max_value(&self) -> Tensor {
        self.max()
    }
}

/// A global, thread-safe cache for `arange` tensors.
///
/// This cache stores tensors created by `Tensor::arange` to avoid redundant
/// generation, which is common when creating attention masks of similar lengths.
/// Tensors are keyed by their `(Device, length)` tuple.
static RANGE_CACHE: Lazy<Mutex<HashMap<(Device, i64), Tensor>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Pads a batch of variable-length sequences into a dense tensor.
///
/// This function efficiently transforms a concatenated tensor of sequences into a
/// single padded tensor and generates a corresponding boolean attention mask. The
/// implementation is optimized for performance on hardware accelerators (e.g., GPUs)
/// by minimizing host-device synchronization and reusing memory buffers.
///
/// # Key Optimizations
/// - **Scratch Buffer**: Uses a cached scratch tensor via `get_scratch` to avoid repeated memory allocations.
/// - **Asynchronous Operations**: Determines the maximum sequence length on the device without blocking.
/// - **Efficient Masking**: Creates the attention mask using broadcasting, avoiding large intermediate tensors.
/// - **Scatter-based Copy**: Uses `index_put_` with indices derived from the attention mask to efficiently copy data into the correct positions.
///
/// # Arguments
///
/// * `sequences` - A 2D tensor of shape `[total_tokens, features]` containing the concatenated data of all sequences.
/// * `length_values` - A 1D tensor of shape `[batch_size]` where each element is the length of a sequence.
/// * `pad_value` - The floating-point value to use for padding.
/// * `device` - The `tch::Device` on which to perform the operations.
///
/// # Returns
///
/// A `Result` containing a tuple of:
/// * The padded sequences as a 3D tensor of shape `[batch_size, max_len, features]`.
/// * A boolean attention mask of shape `[batch_size, max_len]`, where `true` indicates a valid token.
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

    let max_len_tensor = length_values.to_device(device).max_value();
    let max_len: i64 = max_len_tensor.int64_value(&[]);

    let padded_scratch = get_scratch(
        device,
        sequences.kind(),
        &[batch_size, max_len, feature_dim],
        pad_value,
    );

    let mut padded_sequences = padded_scratch
        .narrow(0, 0, batch_size)
        .narrow(1, 0, max_len)
        .narrow(2, 0, feature_dim);
    let _ = padded_sequences.fill_(pad_value);

    let range_row = {
        let mut map = RANGE_CACHE.lock();
        map.entry((device, max_len))
            .or_insert_with(|| Tensor::arange(max_len, (Kind::Int64, device)))
            .shallow_clone()
    };

    let attention_mask = range_row
        .unsqueeze(0)
        .lt_tensor(&length_values.to_device(device).unsqueeze(-1));

    let nz = attention_mask.nonzero();
    let b_indices_flat = nz.select(1, 0);
    let t_indices_flat = nz.select(1, 1);

    let sequences_on_device = sequences.to_device(device);

    let _ = padded_sequences.index_put_(
        &[Some(b_indices_flat), Some(t_indices_flat)],
        &sequences_on_device,
        false,
    );
    Ok((padded_sequences, attention_mask))
}
