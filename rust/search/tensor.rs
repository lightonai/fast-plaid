use std::collections::HashMap;
use tch::{Device, Kind, Tensor};

/// Computes a single quantile for a 1D tensor using `kthvalue`.
///
/// https://github.com/pytorch/pytorch/issues/157431#issuecomment-3026856373
fn scalar_quantile_kthvalue(tensor: &Tensor, q: f64) -> Tensor {
    let n = tensor.size()[0];

    // 1. Calculate target float index
    let idx_float = q * (n - 1) as f64;
    let lower_idx = idx_float.floor() as i64;
    let upper_idx = idx_float.ceil() as i64;

    // Note: kthvalue is 1-indexed in Torch, so we add 1 to our 0-indexed calculations.

    // Optimization: If the index is exactly an integer, we only need one lookup.
    if lower_idx == upper_idx {
        // We take the 0-th element of the tuple returned by kthvalue (values, indices)
        return tensor.kthvalue(lower_idx + 1, 0, true).0;
    }

    // 2. Retrieve bounds
    let (lower_val, _) = tensor.kthvalue(lower_idx + 1, 0, true);
    let (upper_val, _) = tensor.kthvalue(upper_idx + 1, 0, true);

    // 3. Linear Interpolation (Lerp)
    // weight is the fractional part: idx_float - floor(idx_float)
    let weight = idx_float - lower_idx as f64;

    // Performs: start + weight * (end - start)
    lower_val.lerp(&upper_val, weight)
}

pub fn create_view(source_tensor: &Tensor, stride_len: i64, inner_dims: &[i64]) -> Tensor {
    let output_dim = source_tensor.size()[0] - stride_len + 1;

    let mut view_shape = vec![output_dim, stride_len];
    view_shape.extend_from_slice(inner_dims);

    let source_strides = source_tensor.stride();
    let mut new_strides = vec![source_strides[0], source_strides[0]];

    if source_strides.len() > 1 {
        new_strides.extend_from_slice(&source_strides[1..]);
    }

    source_tensor.as_strided(&view_shape, &new_strides, None)
}

/// Creates a boolean mask from a tensor of sequence lengths.
///
/// This function is commonly used to generate an attention mask in sequence models.
/// It produces a boolean tensor where `true` values correspond to valid (non-padded)
/// tokens and `false` values correspond to padded tokens.
///
/// # Arguments
///
/// * `lengths_tensor` - A 1-D `Tensor` of kind `Int64` representing the true
///   lengths of each sequence in a batch.
/// * `max_len` - The maximum sequence length, defining the size of the
///   mask's second dimension.
/// * `match_tensor_dims` - An optional `Tensor` reference. If provided, the
///   output mask will be expanded with trailing dimensions of size 1 to match
///   the rank of this tensor, which is useful for broadcasting.
///
/// # Returns
///
/// A boolean `Tensor` of shape `[batch_size, max_len]`, potentially with
/// additional trailing dimensions. For a sequence of length `L`, the first `L`
/// elements in its corresponding mask row will be `true`, and the rest will be `false`.
///
pub fn create_mask(
    lengths_tensor: &Tensor,
    max_len: i64,
    match_tensor_dims: Option<&Tensor>,
) -> Tensor {
    let device = lengths_tensor.device();
    let position_indices = Tensor::arange(max_len, (Kind::Int64, device)).unsqueeze(0) + 1;

    let lengths = lengths_tensor.unsqueeze(-1);

    let mut mask = position_indices.le_tensor(&lengths);

    if let Some(target_tensor) = match_tensor_dims {
        let num_extra_dims = target_tensor.dim() - mask.dim();
        for _ in 0..num_extra_dims {
            mask = mask.unsqueeze(-1);
        }
    }

    mask
}

/// A data structure for efficient batch lookups on tensors of varying lengths.
///
/// `StridedTensor` stores a collection of tensors as a single, contiguous
/// `underlying_data` tensor. It precomputes several views into this data with
/// different strides to optimize retrieval. This is useful when batching
/// sequences of non-uniform length, as it avoids expensive iteration and
/// concatenation at lookup time.
pub struct StridedTensor {
    /// The flattened, contiguous tensor containing all sequence data, with padding.
    pub underlying_data: Tensor,
    /// The shape of each individual element within the `underlying_data` tensor.
    pub inner_dims: Vec<i64>,
    /// A 1D tensor storing the length of each element sequence.
    pub element_lengths: Tensor,
    /// The maximum length found among all element sequences.
    pub max_element_len: i64,
    /// A sorted vector of strides used to create precomputed views.
    pub precomputed_strides: Vec<i64>,
    /// The cumulative sum of `element_lengths`, used to calculate offsets.
    pub cumulative_lengths: Tensor,
    /// A map from a stride value to its precomputed, strided tensor view.
    pub views_by_stride: HashMap<i64, Tensor>,
}

impl StridedTensor {
    /// Computes optimal strides based on the distribution of element lengths.
    ///
    /// This now uses the optimized `scalar_quantile_kthvalue` logic.
    fn compute_strides(lengths: &Tensor, max_len: i64, device: Device) -> Vec<i64> {
        if lengths.numel() == 0 {
            return if max_len > 0 {
                vec![max_len]
            } else {
                Vec::new()
            };
        }

        // Sample for quantile calculation to improve performance on large tensors.
        let sampled_lengths = if lengths.size()[0] >= 5000 {
            let indices = Tensor::randint(lengths.size()[0], &[2000], (Kind::Int64, device));
            lengths.index_select(0, &indices)
        } else {
            lengths.shallow_clone()
        }
        .to_kind(Kind::Float);

        let target_quantiles = [0.5, 0.75, 0.9, 0.95];

        let mut strides: Vec<i64> = target_quantiles
            .iter()
            .map(|&q| {
                let val_tensor = scalar_quantile_kthvalue(&sampled_lengths, q);
                val_tensor.int64_value(&[]) // Extract scalar result as i64
            })
            .filter(|&s| s > 0)
            .collect();

        // Always include the max length as a possible stride.
        strides.push(max_len);
        strides.sort_unstable();
        strides.dedup();

        // If max_len is 0 and no other positive strides were found, return empty.
        if strides.len() == 1 && strides[0] == 0 {
            return Vec::new();
        }

        strides
    }

    /// Creates a new `StridedTensor`.
    ///
    /// This constructor initializes the structure by preparing the data for efficient
    /// lookups. It pads the data tensor, computes optimal strides, and generates
    /// precomputed strided views.
    ///
    /// # Arguments
    /// * `data` - A tensor containing the concatenated data of all elements.
    /// * `lengths` - A 1D tensor where each entry is the length of an element.
    /// * `device` - The `tch::Device` (e.g., CPU or CUDA) for tensor operations.
    pub fn new(data: Tensor, lengths: Tensor, device: Device) -> Self {
        let inner_dims = if data.dim() > 1 {
            data.size()[1..].to_vec()
        } else {
            Vec::new()
        };
        let element_lengths = lengths.to_device(device).to_kind(Kind::Int64);

        let max_element_len = if element_lengths.numel() > 0 {
            element_lengths.max().int64_value(&[])
        } else {
            0
        };

        let precomputed_strides = Self::compute_strides(&element_lengths, max_element_len, device);
        let cumulative_lengths = {
            let zero_start = Tensor::zeros(&[1], (Kind::Int64, device));
            Tensor::cat(&[zero_start, element_lengths.cumsum(0, Kind::Int64)], 0)
        };

        // Pad the data tensor to ensure any view from any offset is safe.
        let underlying_data = {
            let mut padded_data = data.to_device(device);
            // Padding is only necessary if there are elements to process.
            if cumulative_lengths.size()[0] > 1 {
                // Required length is the start of the last element plus the max possible length.
                let last_element_offset =
                    cumulative_lengths.int64_value(&[cumulative_lengths.size()[0] - 2]);
                let required_len = last_element_offset + max_element_len;

                if required_len > padded_data.size()[0] {
                    let padding_needed = required_len - padded_data.size()[0];
                    let mut padding_shape = vec![padding_needed];
                    padding_shape.extend_from_slice(&inner_dims);
                    let padding = Tensor::zeros(&padding_shape, (padded_data.kind(), device));
                    padded_data = Tensor::cat(&[padded_data, padding], 0);
                }
            }
            padded_data
        };

        let views_by_stride = precomputed_strides
            .iter()
            .map(|&stride| {
                let view = create_view(&underlying_data, stride, &inner_dims);
                (stride, view)
            })
            .collect();

        Self {
            underlying_data,
            inner_dims,
            element_lengths,
            max_element_len,
            precomputed_strides,
            cumulative_lengths,
            views_by_stride,
        }
    }

    /// Retrieves a batch of elements specified by their indices.
    ///
    /// This method efficiently looks up elements by selecting an optimal precomputed
    /// view and applying a mask to remove padding, returning a clean, packed tensor.
    ///
    /// # Arguments
    /// * `indices` - A 1D `Int64` tensor of element indices to retrieve.
    /// * `device` - The target `tch::Device` for the output tensors.
    ///
    /// # Returns
    /// A tuple containing the `(data, lengths)` for the requested indices.
    pub fn lookup(&self, indices: &Tensor, device: Device) -> (Tensor, Tensor) {
        let indices = indices.to_device(device).to_kind(Kind::Int64);

        if indices.numel() == 0 {
            let mut empty_shape = vec![0];
            empty_shape.extend_from_slice(&self.inner_dims);
            return (
                Tensor::empty(&empty_shape, (self.underlying_data.kind(), device)),
                Tensor::empty(&[0], (self.element_lengths.kind(), device)),
            );
        }

        let selected_lengths = self.element_lengths.index_select(0, &indices);
        let selected_offsets = self.cumulative_lengths.index_select(0, &indices);

        let max_selected_len = if selected_lengths.numel() > 0 {
            selected_lengths.max().int64_value(&[])
        } else {
            0
        };

        let chosen_stride = self
            .precomputed_strides
            .iter()
            .find(|&&stride| stride >= max_selected_len)
            .copied()
            .unwrap_or(self.max_element_len);

        if chosen_stride == 0 {
            let mut empty_shape = vec![0];
            empty_shape.extend_from_slice(&self.inner_dims);
            return (
                Tensor::empty(&empty_shape, (self.underlying_data.kind(), device)),
                selected_lengths,
            );
        }

        let view = self.views_by_stride.get(&chosen_stride).unwrap_or_else(|| {
            panic!(
                "Internal error: Stride view not found for stride: {}. Available: {:?}.",
                chosen_stride, self.precomputed_strides
            )
        });

        let strided_data = view.index_select(0, &selected_offsets);
        let mask = create_mask(&selected_lengths, chosen_stride, None).to_kind(Kind::Bool);
        let final_data = strided_data.index(&[Some(mask)]);

        (final_data, selected_lengths)
    }
}
