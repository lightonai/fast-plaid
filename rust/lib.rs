// Local library modules.
pub mod index;
pub mod search;
pub mod utils;

// External crate imports.
use anyhow::anyhow;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use tch::{Device, Kind};

// Standard library imports.
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, RwLock};

// Conditional imports for cross-platform dynamic library loading.
#[cfg(windows)]
use winapi::um::errhandlingapi::GetLastError;
#[cfg(windows)]
use winapi::um::libloaderapi::LoadLibraryA;

// Internal module imports.
use crate::index::create::create_index;
use crate::index::delete::delete_from_index;
use crate::index::update::update_index;
use search::load::{load_index, LoadedIndex};
use search::search::{search_many, QueryResult, SearchParameters};

use once_cell::sync::Lazy;

static INDEX_CACHE: Lazy<RwLock<HashMap<(String, String), Arc<LoadedIndex>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Converts an `anyhow::Error` into a Python `PyValueError`.
fn anyhow_to_pyerr(err: anyhow::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Dynamically loads the native Torch shared library (libtorch).
///
/// This function is necessary to ensure that the `tch` crate can find and
/// link to the PyTorch library at runtime, especially when distributed
/// via Python packages where the exact path isn't known at compile time.
///
/// It uses `dlopen` on Unix-like systems and `LoadLibraryA` on Windows.
///
/// # Arguments
///
/// * `torch_path` - The absolute path to the `libtorch` shared library
///   (e.g., `libtorch.so`, `libtorch.dylib`, or `torch.dll`).
///
/// # Errors
///
/// Returns an `anyhow::Error` if the library fails to load, providing
/// details from `dlerror` (Unix) or `GetLastError` (Windows).
fn call_torch(torch_path: String) -> Result<(), anyhow::Error> {
    let torch_path_cstr = CString::new(torch_path.clone())
        .map_err(|e| anyhow!("Failed to create CString for libtorch path: {}", e))?;

    #[cfg(unix)]
    {
        let handle = unsafe { libc::dlopen(torch_path_cstr.as_ptr(), libc::RTLD_LAZY) };
        if handle.is_null() {
            return Err(anyhow!(
                "Failed to load Torch library '{}' via dlopen. Check the path and permissions.",
                torch_path
            ));
        }
    }

    #[cfg(windows)]
    {
        let handle = unsafe { LoadLibraryA(torch_path_cstr.as_ptr()) };
        if handle.is_null() {
            let error_code = unsafe { GetLastError() };
            return Err(anyhow!(
                "Failed to load Torch library '{}' via LoadLibraryA. Windows error code: {}",
                torch_path,
                error_code
            ));
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        return Err(anyhow!(
            "Dynamic library loading is not supported on this operating system."
        ));
    }

    Ok(())
}

/// Parses a Python-style device string into a `tch::Device`.
///
/// # Arguments
///
/// * `device` - A string slice representing the device, e.g., "cpu", "cuda", "cuda:0".
///
/// # Returns
///
/// A `PyResult` containing the corresponding `tch::Device` if successful.
///
/// # Errors
///
/// Returns a `PyValueError` if the device string is unsupported or invalid
/// (e.g., "cuda:foo" or "tpu").
fn get_device(device: &str) -> Result<Device, PyErr> {
    match device.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)),
        s if s.starts_with("cuda:") => {
            let parts: Vec<&str> = s.split(':').collect();
            if parts.len() == 2 {
                parts[1].parse::<usize>().map(Device::Cuda).map_err(|_| {
                    PyValueError::new_err(format!("Invalid CUDA device index: '{}'", parts[1]))
                })
            } else {
                Err(PyValueError::new_err(
                    "Invalid CUDA device format. Expected 'cuda:N'.",
                ))
            }
        },
        _ => Err(PyValueError::new_err(format!(
            "Unsupported device string: '{}'",
            device
        ))),
    }
}

/// Retrieves a loaded index from the global cache or loads it if not present.
///
/// This function uses a `RwLock` on the `INDEX_CACHE` to ensure thread-safe
/// access. It first attempts to retrieve the index with a read lock. If not
/// found, it acquires a write lock to load the index and insert it into the cache.
///
/// # Arguments
///
/// * `index_path` - The file system path to the index directory.
/// * `device_str` - The string representation of the device (e.g., "cuda:0"),
///   used as part of the cache key.
/// * `device` - The `tch::Device` to load the index onto.
///
/// # Returns
///
/// A `PyResult` containing an `Arc<LoadedIndex>` on success.
///
/// # Errors
///
/// Returns a `PyErr` if `load_index` fails.
fn get_or_load_index(
    index_path: &str,
    device_str: &str,
    device: Device,
) -> PyResult<Arc<LoadedIndex>> {
    let key = (index_path.to_string(), device_str.to_string());

    // --- First check with a read lock ---
    {
        let cache = INDEX_CACHE.read().unwrap();
        if let Some(index_arc) = cache.get(&key) {
            return Ok(Arc::clone(index_arc));
        }
    }

    // --- If not found, acquire a write lock ---
    let mut cache = INDEX_CACHE.write().unwrap();

    // --- Double-check in case another thread loaded it while we waited for the write lock ---
    if let Some(index_arc) = cache.get(&key) {
        return Ok(Arc::clone(index_arc));
    }

    // --- Load the index, insert into cache, and return ---
    let loaded_index = load_index(index_path, device).map_err(anyhow_to_pyerr)?;
    let index_arc = Arc::new(loaded_index);

    cache.insert(key, Arc::clone(&index_arc));

    Ok(index_arc)
}

/// Manually initializes and loads the libtorch shared library.
///
/// This function is called automatically by other functions in this module,
/// but can be called explicitly if needed.
///
/// Args:
///     torch_path (str): The absolute path to the `libtorch` shared library
///         (e.g., `libtorch.so` or `torch.dll`).
///
/// Raises:
///     RuntimeError: If the torch library fails to load.
#[pyfunction]
fn initialize_torch(_py: Python<'_>, torch_path: String) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize Torch: {}", e)))
}

/// [Internal] Creates and saves a new FastPlaid index.
///
/// This is the low-level Rust implementation called by `FastPlaid.create()`.
/// It's generally recommended to use the `FastPlaid` class wrapper instead.
///
/// This function builds the index from pre-computed centroids and embeddings,
/// quantizes the embeddings, and saves the index files to disk.
///
/// Args:
///     index (str): The file path to the directory to save the index.
///     torch_path (str): Path to the `libtorch` shared library.
///     device (str): Device to use for computation (e.g., "cpu", "cuda:0").
///     embedding_dim (int): The dimension of the token embeddings (e.g., 128).
///     nbits (int): Number of bits for product quantization (e.g., 4).
///     embeddings (list[torch.Tensor]): A list of 2D tensors
///         (num_tokens, embedding_dim), one for each document.
///     centroids (torch.Tensor): A 2D tensor of (num_centroids, embedding_dim)
///         pre-computed by K-means.
///     batch_size (int): Batch size for processing embeddings during creation.
///     seed (int | None): Optional seed for reproducible index creation.
///
/// Raises:
///     RuntimeError: If index creation fails or `libtorch` fails to load.
#[pyfunction]
fn create(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    embedding_dim: i64,
    nbits: i64,
    embeddings: Vec<PyTensor>,
    centroids: PyTensor,
    batch_size: i64,
    seed: Option<u64>,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let mut cache = INDEX_CACHE.write().unwrap();
    cache.clear();

    let device = get_device(&device)?;
    let centroids = centroids.to_device(device).to_kind(Kind::Half);

    let embeddings: Vec<_> = embeddings
        .into_iter()
        .map(|tensor| tensor.to_kind(Kind::Half))
        .collect();

    let result = create_index(
        &embeddings,
        &index,
        embedding_dim,
        nbits,
        device,
        centroids,
        batch_size,
        seed,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create index: {}", e)));

    result
}

/// [Internal] Adds new documents to an existing FastPlaid index.
///
/// This is the low-level Rust implementation called by `FastPlaid.update()`.
/// It's generally recommended to use the `FastPlaid` class wrapper instead.
///
/// This function appends new quantized embeddings to the index files. It does
/// not re-compute centroids and assumes the existing index structure.
/// Clears the *entire* index cache upon successful completion.
///
/// Args:
///     index (str): The file path to the directory of the existing index.
///     torch_path (str): Path to the `libtorch` shared library.
///     device (str): Device to use for computation (e.g., "cpu", "cuda:0").
///     embeddings (list[torch.Tensor]): A list of 2D tensors
///         (num_tokens, embedding_dim), one for each new document.
///     batch_size (int): Batch size for processing new embeddings.
///
/// Raises:
///     RuntimeError: If updating the index fails or `libtorch` fails to load.
#[pyfunction]
fn update(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    embeddings: Vec<PyTensor>,
    batch_size: i64,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let mut cache = INDEX_CACHE.write().unwrap();
    cache.clear();

    let device = get_device(&device)?;

    let embeddings: Vec<_> = embeddings
        .into_iter()
        .map(|tensor| tensor.to_device(device).to_kind(Kind::Half))
        .collect();

    let result = update_index(&embeddings, &index, device, batch_size)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to update index: {}", e)));

    result
}

/// [Internal] Loads an index into the global cache for fast access.
///
/// This is the low-level Rust implementation called by `FastPlaid._load_index()`.
/// It's generally recommended to use the `FastPlaid` class wrapper instead.
///
/// If the index (for the specified path and device) is already in the cache,
/// this function does nothing. Otherwise, it loads the index from disk
/// and stores it in a global, thread-safe cache.
///
/// Args:
///     index (str): The file path to the directory of the existing index.
///     torch_path (str): Path to the `libtorch` shared library.
///     device (str): Device to load the index onto (e.g., "cpu", "cuda:0").
///
/// Raises:
///     RuntimeError: If loading the index fails or `libtorch` fails to load.
#[pyfunction]
fn preload_index(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;
    let device_tch = get_device(&device)?;
    // Call get_or_load_index to trigger a load if not already cached
    get_or_load_index(&index, &device, device_tch)?;

    Ok(())
}

/// [Internal] Loads an index (from cache or disk) and performs a search.
///
/// This is the low-level Rust implementation called by `FastPlaid.search()`.
/// It's generally recommended to use the `FastPlaid` class wrapper instead.
///
/// This function first retrieves the index from the global cache (or loads it
/// if not present), then executes the multi-vector search against it.
///
/// Args:
///     index (str): The file path to the directory of the existing index.
///     torch_path (str): Path to the `libtorch` shared library.
///     device (str): Device to perform the search on (e.g., "cpu", "cuda:0").
///     queries_embeddings (torch.Tensor): A 3D tensor of query embeddings
///         with shape (num_queries, num_query_tokens, embedding_dim).
///     search_parameters (SearchParameters): A SearchParameters object
///         containing `top_k`, `n_ivf_probe`, etc.
///     show_progress (bool): Whether to display a progress bar during search.
///     subset (list[list[int]] | None): An optional filter to restrict the
///         search. Must be a list of lists, where each inner list contains
///         the document IDs to search for that specific query.
///
/// Returns:
///     list[QueryResult]: A list of `QueryResult` objects, one for each query.
///
/// Raises:
///     RuntimeError: If searching fails or `libtorch` fails to load.
///     ValueError: If search parameters are invalid.
#[pyfunction]
fn load_and_search(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    queries_embeddings: PyTensor,
    search_parameters: &SearchParameters,
    show_progress: bool,
    preload_index: bool,
    subset: Option<Vec<Vec<i64>>>,
) -> PyResult<Vec<QueryResult>> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let device_tch = get_device(&device)?;

    // Get the index from cache or load it
    let index = if preload_index {
        get_or_load_index(&index, &device, device_tch)
    } else {
        println!("Loading index from disk without caching.");
        let loaded_index = load_index(&index, device_tch)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load index: {}", e)))?;
        Ok(Arc::new(loaded_index))
    }?;

    // Perform the search
    let results = search_many(
        &queries_embeddings.to_kind(Kind::Half),
        &index,
        search_parameters,
        device_tch,
        show_progress,
        subset,
    )
    .map_err(anyhow_to_pyerr)?;
    Ok(results)
}

/// [Internal] Deletes documents from an existing FastPlaid index.
///
/// This is the low-level Rust implementation called by `FastPlaid.delete()`.
/// It's generally recommended to use the `FastPlaid` class wrapper instead.
///
/// This function removes the specified document IDs from the index files.
/// The remaining documents are re-indexed to maintain sequential IDs.
/// Clears the *entire* index cache upon successful completion.
///
/// Args:
///     index (str): The file path to the directory of the existing index.
///     torch_path (str): Path to the `libtorch` shared library.
///     device (str): Device to use for the operation (e.g., "cpu", "cuda:0").
///     subset (list[int]): A list of document IDs to delete. IDs correspond
///         to the original insertion order.
///
/// Raises:
///     RuntimeError: If deletion fails or `libtorch` fails to load.
#[pyfunction]
fn delete(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    subset: Vec<i64>,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let mut cache = INDEX_CACHE.write().unwrap();
    cache.clear();

    let device = get_device(&device)?;

    let result = delete_from_index(&subset, &index, device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete from index: {}", e)));

    result
}

/// Defines the Python module `fast_plaid_rust`.
///
/// This function is the main entry point for the PyO3 library. It registers
/// all the exposed Python functions (`create`, `update`, `load_and_search`, etc.)
/// and classes (`SearchParameters`, `QueryResult`) to make them available
/// for import in Python.
#[pymodule]
#[pyo3(name = "fast_plaid_rust")]
fn python_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchParameters>()?;
    m.add_class::<QueryResult>()?;

    m.add_function(wrap_pyfunction!(initialize_torch, m)?)?;
    m.add_function(wrap_pyfunction!(create, m)?)?;
    m.add_function(wrap_pyfunction!(update, m)?)?;
    m.add_function(wrap_pyfunction!(preload_index, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_search, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;
    Ok(())
}
