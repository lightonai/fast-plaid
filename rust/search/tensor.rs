use std::collections::HashMap;

use rayon::prelude::*;
use tch::{Device, Kind, Tensor};

/// Computes a single quantile for a 1D tensor using `kthvalue`.
///
/// This function calculates the value below which a given percentage of data falls.
/// It employs linear interpolation between the two nearest ranks if the target
/// index is not an integer.
///
/// # Arguments
///
/// * `tensor` - The input 1D tensor.
/// * `q` - The quantile to compute (between 0.0 and 1.0).
///
/// # Returns
///
/// A scalar `Tensor` containing the computed quantile value.
pub fn scalar_quantile_kthvalue(tensor: &Tensor, q: f64) -> Tensor {
    let n = tensor.size()[0];

    let idx_float = q * (n - 1) as f64;
    let lower_idx = idx_float.floor() as i64;
    let upper_idx = idx_float.ceil() as i64;

    if lower_idx == upper_idx {
        return tensor.kthvalue(lower_idx + 1, 0, true).0;
    }

    let (lower_val, _) = tensor.kthvalue(lower_idx + 1, 0, true);
    let (upper_val, _) = tensor.kthvalue(upper_idx + 1, 0, true);

    let weight = idx_float - lower_idx as f64;
    lower_val.lerp(&upper_val, weight)
}

/// Creates a sliding window view of a tensor without copying data.
///
/// This function generates a new tensor view by sliding a window of a specified
/// length over the first dimension of the source tensor. It leverages the
/// `as_strided` method to create this view efficiently, avoiding memory
/// duplication.
///
///
///
/// # Arguments
///
/// * `source_tensor` - A reference to the input `Tensor`. The sliding window
///   will be applied along its first dimension.
/// * `stride_len` - The length of the sliding window. This determines the size
///   of the second dimension in the output tensor.
/// * `inner_dims` - A slice representing the remaining dimensions of the
///   `source_tensor` that should be preserved within each window.
///
/// # Returns
///
/// A new `Tensor` that is a strided view of the original. The new shape will be
/// `[source_tensor.size()[0] - stride_len + 1, stride_len, ...inner_dims]`.
///
/// # Panics
///
/// This function may panic if `stride_len` is greater than the size of the
/// first dimension of `source_tensor`, as this would result in a negative
/// output dimension.
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
///
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
    /// Strides are determined by sampling quantiles, ensuring that common sequence
    /// lengths are well-represented. The maximum element length is always included.
    fn compute_strides(lengths: &Tensor, max_len: i64, device: Device) -> Vec<i64> {
        if lengths.numel() == 0 {
            return if max_len > 0 {
                vec![max_len]
            } else {
                Vec::new()
            };
        }

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
                val_tensor.int64_value(&[])
            })
            .filter(|&s| s > 0)
            .collect();

        strides.push(max_len);
        strides.sort_unstable();
        strides.dedup();

        if strides.len() == 1 && strides[0] == 0 {
            return Vec::new();
        }

        strides
    }

    /// Creates a `StridedTensor` whose (tiny) length tensors live on `lengths_device`, so
    /// per-query sorting and chunk planning stay on the search device when data is CPU-resident.
    pub fn new_with_lengths_device(
        data: Tensor,
        lengths: Tensor,
        storage_device: Device,
        lengths_device: Device,
    ) -> Self {
        let mut built = Self::new(data, lengths, storage_device);
        if lengths_device != storage_device {
            built.element_lengths = built.element_lengths.to_device(lengths_device);
            built.cumulative_lengths = built.cumulative_lengths.to_device(lengths_device);
        }
        built
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

        // Lengths are small, safe to move immediately
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

        let underlying_data = {
            // Check if conversion is actually needed to avoid redundant copy
            let data_on_dev = if data.device() != device {
                data.to_device(device)
            } else {
                data
            };

            let current_size = data_on_dev.size()[0];
            let mut required_len = current_size;

            if cumulative_lengths.size()[0] > 1 {
                let last_element_offset =
                    cumulative_lengths.int64_value(&[cumulative_lengths.size()[0] - 2]);
                required_len = last_element_offset + max_element_len;
            }

            if required_len > current_size {
                // Calculate padding needed
                let mut new_shape = vec![required_len];
                new_shape.extend_from_slice(&inner_dims);

                // Allocate the FULL target memory once (zeros)
                // This respects the original data's kind (e.g., Uint8)
                let result = Tensor::zeros(&new_shape, (data_on_dev.kind(), device));

                // Copy the original data into the start of the new buffer
                // This keeps memory usage at (Size_Original + Size_New) instead of
                // (Size_Original + Size_Padding + Size_New) which `cat` would do.
                result.narrow(0, 0, current_size).copy_(&data_on_dev);

                result
            } else {
                // No padding needed, return the tensor as-is
                data_on_dev
            }
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

    /// An empty `[0, ...inner_dims]` tensor of the stored data's kind.
    fn empty_data(&self, device: Device) -> Tensor {
        let mut shape = vec![0];
        shape.extend_from_slice(&self.inner_dims);
        Tensor::empty(&shape, (self.underlying_data.kind(), device))
    }

    /// Bytes occupied by one row of `underlying_data`.
    fn row_bytes(&self) -> usize {
        let cells: i64 = self.inner_dims.iter().product();
        cells.max(1) as usize * self.underlying_data.kind().elt_size_in_bytes()
    }

    /// Copies the requested elements' rows into a packed buffer.
    ///
    /// An element's rows are contiguous in storage and contiguous in the
    /// output, so each element moves as a single block. That is the whole
    /// difference from the strided path, which writes every byte once as
    /// padding and reads it back to compact it. Copying blocks also avoids a
    /// per-row index, which for narrow rows would cost as much as the rows.
    ///
    /// Returns `None` if the storage cannot be addressed as flat bytes, which
    /// leaves the caller on the strided path.
    fn gather_packed(&self, offsets: &[i64], lengths: &[i64], total: i64) -> Option<Tensor> {
        if !self.underlying_data.is_contiguous() {
            return None;
        }
        if total == 0 {
            return Some(self.empty_data(Device::Cpu));
        }

        let row_bytes = self.row_bytes();
        let mut shape = vec![total];
        shape.extend_from_slice(&self.inner_dims);
        let output = Tensor::empty(&shape, (self.underlying_data.kind(), Device::Cpu));

        // SAFETY: both tensors are contiguous, on the host and of a single
        // kind, so each holds exactly `rows * row_bytes` addressable bytes.
        // `output` was allocated here and is unaliased; `underlying_data` is
        // immutable for as long as the index is loaded. Neither borrow leaves
        // this function.
        let source = unsafe {
            std::slice::from_raw_parts(
                self.underlying_data.data_ptr() as *const u8,
                self.underlying_data.size()[0] as usize * row_bytes,
            )
        };
        let mut remaining = unsafe {
            std::slice::from_raw_parts_mut(output.data_ptr() as *mut u8, total as usize * row_bytes)
        };

        // Hand every element its own disjoint slice of the output up front, so
        // the copies are independent and may run in any order.
        let mut blocks = Vec::with_capacity(lengths.len());
        for &length in lengths {
            let (block, rest) = remaining.split_at_mut(length as usize * row_bytes);
            blocks.push(block);
            remaining = rest;
        }

        blocks
            .into_par_iter()
            .zip(offsets.par_iter())
            .for_each(|(block, &offset)| {
                let start = offset as usize * row_bytes;
                block.copy_from_slice(&source[start..start + block.len()]);
            });

        Some(output)
    }

    /// Retrieves a batch of elements specified by their indices.
    ///
    /// This method efficiently looks up elements by selecting an optimal precomputed
    /// view and applying a mask to remove padding, returning a clean, packed tensor.
    ///
    /// # Hybrid Storage Support:
    /// This method is device-aware. If the `underlying_data` is on the CPU but the
    /// `target_device` is the GPU, it will perform the gather operation on the CPU
    /// (saving VRAM) and then stream the result to the GPU.
    ///
    /// # Arguments
    /// * `indices` - A 1D `Int64` tensor of element indices to retrieve.
    /// * `target_device` - The `tch::Device` where the result should end up.
    ///
    /// # Returns
    /// A tuple containing the `(data, lengths)` on the `target_device`.
    pub fn lookup(&self, indices: &Tensor, target_device: Device) -> (Tensor, Tensor) {
        let storage_device = self.underlying_data.device();
        // Length/offset math runs where the lengths live; only the small offsets move for the gather.
        let lengths_device = self.element_lengths.device();

        let indices_local = indices.to_device(lengths_device).to_kind(Kind::Int64);

        if indices_local.numel() == 0 {
            return (
                self.empty_data(target_device),
                Tensor::empty(&[0], (self.element_lengths.kind(), target_device)),
            );
        }

        let selected_lengths = self.element_lengths.index_select(0, &indices_local);
        let selected_offsets = self.cumulative_lengths.index_select(0, &indices_local);

        // Host storage reads its rows where they already sit. The strided path
        // below builds a `[elements, stride, ...]` rectangle and then compacts
        // it with a boolean mask, so every byte is written once as padding and
        // read twice; the packed gather selects the same bytes in one pass. It
        // also never needs the widest selected length, so the scalar sync that
        // chooses a stride goes with it.
        if storage_device == Device::Cpu {
            let lengths_host = selected_lengths.to_device(Device::Cpu);
            let offsets_host = selected_offsets.to_device(Device::Cpu);
            if let (Ok(lengths), Ok(offsets)) = (
                Vec::<i64>::try_from(&lengths_host),
                Vec::<i64>::try_from(&offsets_host),
            ) {
                let total: i64 = lengths.iter().sum();
                if let Some(data) = self.gather_packed(&offsets, &lengths, total) {
                    return (
                        data.to_device(target_device),
                        selected_lengths.to_device(target_device),
                    );
                }
            }
        }

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
            return (
                self.empty_data(target_device),
                selected_lengths.to_device(target_device),
            );
        }

        let view = self.views_by_stride.get(&chosen_stride).unwrap_or_else(|| {
            panic!(
                "Internal error: Stride view not found for stride: {}. Available: {:?}.",
                chosen_stride, self.precomputed_strides
            )
        });

        // Select data and apply mask (on the storage device)
        let offsets_for_gather = selected_offsets.to_device(storage_device);
        let lengths_for_mask = selected_lengths.to_device(storage_device);
        let strided_data = view.index_select(0, &offsets_for_gather);
        let mask = crate::search::tensor::create_mask(&lengths_for_mask, chosen_stride, None)
            .to_kind(Kind::Bool);
        let final_data = strided_data.index(&[Some(mask)]);

        (
            final_data.to_device(target_device),
            selected_lengths.to_device(target_device),
        )
    }
}
