use tch::Tensor;

/// Calculates and formats the memory usage of a tensor.
pub fn print_tensor_memory(name: &str, tensor: &Tensor) {
    // Check if tensor is initialized (not an empty handle)
    if tensor.numel() == 0 && tensor.size().is_empty() {
        println!("  - {}: (empty tensor)", name);
        return;
    }

    let bytes = tensor.numel() as usize * tensor.kind().elt_size_in_bytes();
    println!(
        "  - {}: {} (Shape: {:?}, Kind: {:?}), max value: {:?}",
        name,
        format_bytes(bytes),
        tensor.size(),
        tensor.kind(),
        tensor.max(),
    );
}

/// Formats bytes into a human-readable string (KB, MB, GB).
fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * KB;
    const GB: f64 = 1024.0 * MB;

    let bytes_f = bytes as f64;

    if bytes_f >= GB {
        format!("{:.2} GB", bytes_f / GB)
    } else if bytes_f >= MB {
        format!("{:.2} MB", bytes_f / MB)
    } else if bytes_f >= KB {
        format!("{:.2} KB", bytes_f / KB)
    } else {
        format!("{} bytes", bytes)
    }
}
