use fastembed::{
    EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, InitOptions,
    RerankInitOptions, RerankerModel, SparseInitOptions, SparseModel, SparseTextEmbedding,
    TextEmbedding, TextRerank,
};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::slice;
use std::time::Instant;

// Opaque handles for the models
pub struct TextEmbeddingHandle(Box<TextEmbedding>);
pub struct SparseTextEmbeddingHandle(Box<SparseTextEmbedding>);
pub struct ImageEmbeddingHandle(Box<ImageEmbedding>);
pub struct TextRerankHandle(Box<TextRerank>);

// Error handling
#[repr(C)]
pub struct FastEmbedError {
    pub message: *mut c_char,
}

impl FastEmbedError {
    fn from_string(s: String) -> *mut FastEmbedError {
        let c_str = CString::new(s).unwrap_or_else(|_| CString::new("Invalid error message").unwrap());
        Box::into_raw(Box::new(FastEmbedError {
            message: c_str.into_raw(),
        }))
    }
}

#[no_mangle]
pub extern "C" fn fastembed_error_free(error: *mut FastEmbedError) {
    if !error.is_null() {
        unsafe {
            let error = Box::from_raw(error);
            if !error.message.is_null() {
                let _ = CString::from_raw(error.message);
            }
        }
    }
}

// Result types
#[repr(C)]
pub struct FloatArray {
    pub data: *mut f32,
    pub len: usize,
}

#[repr(C)]
pub struct FloatArrayVec {
    pub arrays: *mut FloatArray,
    pub len: usize,
}

#[repr(C)]
pub struct SparseEmbeddingC {
    pub indices: *mut usize,
    pub values: *mut f32,
    pub len: usize,
}

#[repr(C)]
pub struct SparseEmbeddingVec {
    pub embeddings: *mut SparseEmbeddingC,
    pub len: usize,
}

#[repr(C)]
pub struct RerankResultC {
    pub index: usize,
    pub score: f32,
    pub document: *mut c_char,
}

#[repr(C)]
pub struct RerankResultVec {
    pub results: *mut RerankResultC,
    pub len: usize,
}

// Text Embedding Functions
#[no_mangle]
pub extern "C" fn fastembed_text_embedding_new(
    model_name: *const c_char,
    error: *mut *mut FastEmbedError,
) -> *mut TextEmbeddingHandle {
    let model_str = unsafe {
        if model_name.is_null() {
            "BAAI/bge-small-en-v1.5"
        } else {
            match CStr::from_ptr(model_name).to_str() {
                Ok(s) => s,
                Err(e) => {
                    if !error.is_null() {
                        *error = FastEmbedError::from_string(format!("Invalid model name: {}", e));
                    }
                    return ptr::null_mut();
                }
            }
        }
    };

    let model = match model_str {
        "AllMiniLML6V2" => EmbeddingModel::AllMiniLML6V2,
        "BGESmallENV15" => EmbeddingModel::BGESmallENV15,
        "BGEBaseENV15" => EmbeddingModel::BGEBaseENV15,
        "BGELargeENV15" => EmbeddingModel::BGELargeENV15,
        _ => EmbeddingModel::BGESmallENV15, // default
    };

    // NOTE: CoreML execution provider has limited support for transformer models
    // Most BERT-style operations will fall back to CPU even when CoreML is "available"
    // See: https://github.com/microsoft/onnxruntime/issues/16934
    // 
    // For now, we'll use CPU-only execution which is well-optimized by ONNX Runtime
    let init_options = InitOptions::new(model);

    match TextEmbedding::try_new(init_options) {
        Ok(embedding) => {
            eprintln!("[FASTEMBED-RUST] Text embedding model initialized (CPU-optimized)");
            Box::into_raw(Box::new(TextEmbeddingHandle(Box::new(embedding))))
        },
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string(format!("Failed to create text embedding: {}", e));
                }
            }
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn fastembed_text_embedding_embed(
    handle: *mut TextEmbeddingHandle,
    texts: *const *const c_char,
    num_texts: usize,
    batch_size: usize,
    error: *mut *mut FastEmbedError,
) -> *mut FloatArrayVec {
    if handle.is_null() || texts.is_null() {
        if !error.is_null() {
            unsafe {
                *error = FastEmbedError::from_string("Null pointer provided".to_string());
            }
        }
        return ptr::null_mut();
    }

    let handle = unsafe { &mut *handle };
    let text_slice = unsafe { slice::from_raw_parts(texts, num_texts) };

    let mut text_vec = Vec::new();
    for &text_ptr in text_slice {
        if text_ptr.is_null() {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string("Null text pointer in array".to_string());
                }
            }
            return ptr::null_mut();
        }
        let text = unsafe { CStr::from_ptr(text_ptr).to_str() };
        match text {
            Ok(s) => text_vec.push(s.to_string()),
            Err(e) => {
                if !error.is_null() {
                    unsafe {
                        *error = FastEmbedError::from_string(format!("Invalid UTF-8 in text: {}", e));
                    }
                }
                return ptr::null_mut();
            }
        }
    }

    let batch_size_opt = if batch_size > 0 { Some(batch_size) } else { None };

    // Time the embedding operation
    let start = Instant::now();
    let result = handle.0.embed(text_vec, batch_size_opt);
    let duration = start.elapsed();
    
    eprintln!("[FASTEMBED-RUST] Text embedding: {} texts in {:.3}s ({:.2}ms/text)", 
              num_texts, duration.as_secs_f64(), duration.as_millis() as f64 / num_texts as f64);

    match result {
        Ok(embeddings) => {
            let mut arrays: Vec<FloatArray> = embeddings
                .into_iter()
                .map(|emb| {
                    let mut boxed_slice = emb.into_boxed_slice();
                    let len = boxed_slice.len();
                    let data = boxed_slice.as_mut_ptr();
                    std::mem::forget(boxed_slice);
                    FloatArray { data, len }
                })
                .collect();

            let len = arrays.len();
            let arrays_ptr = arrays.as_mut_ptr();
            std::mem::forget(arrays);

            Box::into_raw(Box::new(FloatArrayVec {
                arrays: arrays_ptr,
                len,
            }))
        }
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string(format!("Embedding failed: {}", e));
                }
            }
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn fastembed_text_embedding_free(handle: *mut TextEmbeddingHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

// Sparse Text Embedding Functions
#[no_mangle]
pub extern "C" fn fastembed_sparse_text_embedding_new(
    model_name: *const c_char,
    error: *mut *mut FastEmbedError,
) -> *mut SparseTextEmbeddingHandle {
    let model_str = unsafe {
        if model_name.is_null() {
            "Qdrant/Splade_PP_en_v1"
        } else {
            match CStr::from_ptr(model_name).to_str() {
                Ok(s) => s,
                Err(e) => {
                    if !error.is_null() {
                        *error = FastEmbedError::from_string(format!("Invalid model name: {}", e));
                    }
                    return ptr::null_mut();
                }
            }
        }
    };

    let model = match model_str.parse::<SparseModel>() {
        Ok(m) => m,
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string(format!("Invalid sparse model: {}", e));
                }
            }
            return ptr::null_mut();
        }
    };

    // NOTE: CoreML has limited support for transformer models - using CPU-only
    let init_options = SparseInitOptions::new(model);

    match SparseTextEmbedding::try_new(init_options) {
        Ok(embedding) => {
            eprintln!("[FASTEMBED-RUST] Sparse text embedding model initialized (CPU-optimized)");
            Box::into_raw(Box::new(SparseTextEmbeddingHandle(Box::new(embedding))))
        },
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string(format!("Failed to create sparse text embedding: {}", e));
                }
            }
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn fastembed_sparse_text_embedding_embed(
    handle: *mut SparseTextEmbeddingHandle,
    texts: *const *const c_char,
    num_texts: usize,
    batch_size: usize,
    error: *mut *mut FastEmbedError,
) -> *mut SparseEmbeddingVec {
    if handle.is_null() || texts.is_null() {
        if !error.is_null() {
            unsafe {
                *error = FastEmbedError::from_string("Null pointer provided".to_string());
            }
        }
        return ptr::null_mut();
    }

    let handle = unsafe { &mut *handle };
    let text_slice = unsafe { slice::from_raw_parts(texts, num_texts) };

    let mut text_vec = Vec::new();
    for &text_ptr in text_slice {
        if text_ptr.is_null() {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string("Null text pointer in array".to_string());
                }
            }
            return ptr::null_mut();
        }
        let text = unsafe { CStr::from_ptr(text_ptr).to_str() };
        match text {
            Ok(s) => text_vec.push(s.to_string()),
            Err(e) => {
                if !error.is_null() {
                    unsafe {
                        *error = FastEmbedError::from_string(format!("Invalid UTF-8 in text: {}", e));
                    }
                }
                return ptr::null_mut();
            }
        }
    }

    let batch_size_opt = if batch_size > 0 { Some(batch_size) } else { None };

    // Time the sparse embedding operation
    let start = Instant::now();
    let result = handle.0.embed(text_vec, batch_size_opt);
    let duration = start.elapsed();
    
    eprintln!("[FASTEMBED-RUST] Sparse embedding: {} texts in {:.3}s ({:.2}ms/text)", 
              num_texts, duration.as_secs_f64(), duration.as_millis() as f64 / num_texts as f64);

    match result {
        Ok(embeddings) => {
            let mut sparse_embs: Vec<SparseEmbeddingC> = embeddings
                .into_iter()
                .map(|emb| {
                    let mut indices_vec = emb.indices;
                    let mut values_vec = emb.values;
                    let len = indices_vec.len();
                    let indices_ptr = indices_vec.as_mut_ptr();
                    let values_ptr = values_vec.as_mut_ptr();
                    std::mem::forget(indices_vec);
                    std::mem::forget(values_vec);
                    SparseEmbeddingC {
                        indices: indices_ptr,
                        values: values_ptr,
                        len,
                    }
                })
                .collect();

            let len = sparse_embs.len();
            let embeddings_ptr = sparse_embs.as_mut_ptr();
            std::mem::forget(sparse_embs);

            Box::into_raw(Box::new(SparseEmbeddingVec {
                embeddings: embeddings_ptr,
                len,
            }))
        }
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string(format!("Sparse embedding failed: {}", e));
                }
            }
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn fastembed_sparse_text_embedding_free(handle: *mut SparseTextEmbeddingHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

// Image Embedding Functions
#[no_mangle]
pub extern "C" fn fastembed_image_embedding_new(
    model_name: *const c_char,
    error: *mut *mut FastEmbedError,
) -> *mut ImageEmbeddingHandle {
    let model_str = unsafe {
        if model_name.is_null() {
            "Qdrant/clip-ViT-B-32-vision"
        } else {
            match CStr::from_ptr(model_name).to_str() {
                Ok(s) => s,
                Err(e) => {
                    if !error.is_null() {
                        *error = FastEmbedError::from_string(format!("Invalid model name: {}", e));
                    }
                    return ptr::null_mut();
                }
            }
        }
    };

    let model = match model_str.parse::<ImageEmbeddingModel>() {
        Ok(m) => m,
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string(format!("Invalid image model: {}", e));
                }
            }
            return ptr::null_mut();
        }
    };

    match ImageEmbedding::try_new(ImageInitOptions::new(model)) {
        Ok(embedding) => Box::into_raw(Box::new(ImageEmbeddingHandle(Box::new(embedding)))),
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string(format!("Failed to create image embedding: {}", e));
                }
            }
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn fastembed_image_embedding_embed(
    handle: *mut ImageEmbeddingHandle,
    image_paths: *const *const c_char,
    num_images: usize,
    batch_size: usize,
    error: *mut *mut FastEmbedError,
) -> *mut FloatArrayVec {
    if handle.is_null() || image_paths.is_null() {
        if !error.is_null() {
            unsafe {
                *error = FastEmbedError::from_string("Null pointer provided".to_string());
            }
        }
        return ptr::null_mut();
    }

    let handle = unsafe { &mut *handle };
    let path_slice = unsafe { slice::from_raw_parts(image_paths, num_images) };

    let mut path_vec = Vec::new();
    for &path_ptr in path_slice {
        if path_ptr.is_null() {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string("Null path pointer in array".to_string());
                }
            }
            return ptr::null_mut();
        }
        let path = unsafe { CStr::from_ptr(path_ptr).to_str() };
        match path {
            Ok(s) => path_vec.push(s.to_string()),
            Err(e) => {
                if !error.is_null() {
                    unsafe {
                        *error = FastEmbedError::from_string(format!("Invalid UTF-8 in path: {}", e));
                    }
                }
                return ptr::null_mut();
            }
        }
    }

    let batch_size_opt = if batch_size > 0 { Some(batch_size) } else { None };

    match handle.0.embed(path_vec, batch_size_opt) {
        Ok(embeddings) => {
            let mut arrays: Vec<FloatArray> = embeddings
                .into_iter()
                .map(|emb| {
                    let mut boxed_slice = emb.into_boxed_slice();
                    let len = boxed_slice.len();
                    let data = boxed_slice.as_mut_ptr();
                    std::mem::forget(boxed_slice);
                    FloatArray { data, len }
                })
                .collect();

            let len = arrays.len();
            let arrays_ptr = arrays.as_mut_ptr();
            std::mem::forget(arrays);

            Box::into_raw(Box::new(FloatArrayVec {
                arrays: arrays_ptr,
                len,
            }))
        }
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string(format!("Image embedding failed: {}", e));
                }
            }
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn fastembed_image_embedding_free(handle: *mut ImageEmbeddingHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

// Text Rerank Functions
#[no_mangle]
pub extern "C" fn fastembed_text_rerank_new(
    model_name: *const c_char,
    error: *mut *mut FastEmbedError,
) -> *mut TextRerankHandle {
    let model_str = unsafe {
        if model_name.is_null() {
            "BAAI/bge-reranker-base"
        } else {
            match CStr::from_ptr(model_name).to_str() {
                Ok(s) => s,
                Err(e) => {
                    if !error.is_null() {
                        *error = FastEmbedError::from_string(format!("Invalid model name: {}", e));
                    }
                    return ptr::null_mut();
                }
            }
        }
    };

    let model = match model_str.parse::<RerankerModel>() {
        Ok(m) => m,
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string(format!("Invalid reranker model: {}", e));
                }
            }
            return ptr::null_mut();
        }
    };

    // NOTE: CoreML has limited support for transformer models - using CPU-only
    let init_options = RerankInitOptions::new(model);

    match TextRerank::try_new(init_options) {
        Ok(reranker) => {
            eprintln!("[FASTEMBED-RUST] Text reranker model initialized (CPU-optimized)");
            Box::into_raw(Box::new(TextRerankHandle(Box::new(reranker))))
        },
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string(format!("Failed to create text reranker: {}", e));
                }
            }
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn fastembed_text_rerank_rerank(
    handle: *mut TextRerankHandle,
    query: *const c_char,
    documents: *const *const c_char,
    num_documents: usize,
    return_documents: bool,
    batch_size: usize,
    error: *mut *mut FastEmbedError,
) -> *mut RerankResultVec {
    if handle.is_null() || query.is_null() || documents.is_null() {
        if !error.is_null() {
            unsafe {
                *error = FastEmbedError::from_string("Null pointer provided".to_string());
            }
        }
        return ptr::null_mut();
    }

    let handle = unsafe { &mut *handle };
    let query_str = unsafe {
        match CStr::from_ptr(query).to_str() {
            Ok(s) => s,
            Err(e) => {
                if !error.is_null() {
                    *error = FastEmbedError::from_string(format!("Invalid UTF-8 in query: {}", e));
                }
                return ptr::null_mut();
            }
        }
    };

    let doc_slice = unsafe { slice::from_raw_parts(documents, num_documents) };
    let mut doc_strings = Vec::new();
    for &doc_ptr in doc_slice {
        if doc_ptr.is_null() {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string("Null document pointer in array".to_string());
                }
            }
            return ptr::null_mut();
        }
        let doc = unsafe { CStr::from_ptr(doc_ptr).to_str() };
        match doc {
            Ok(s) => doc_strings.push(s.to_string()),
            Err(e) => {
                if !error.is_null() {
                    unsafe {
                        *error = FastEmbedError::from_string(format!("Invalid UTF-8 in document: {}", e));
                    }
                }
                return ptr::null_mut();
            }
        }
    }

    let batch_size_opt = if batch_size > 0 { Some(batch_size) } else { None };
    
    // Convert Vec<String> to Vec<&str> for the rerank call
    let doc_vec: Vec<&str> = doc_strings.iter().map(|s| s.as_str()).collect();

    // Time the reranking operation
    let start = Instant::now();
    let result = handle.0.rerank(query_str, doc_vec, return_documents, batch_size_opt);
    let duration = start.elapsed();
    
    eprintln!("[FASTEMBED-RUST] Reranking: {} documents in {:.3}s ({:.2}ms/doc)", 
              num_documents, duration.as_secs_f64(), duration.as_millis() as f64 / num_documents as f64);

    match result {
        Ok(results) => {
            let mut c_results: Vec<RerankResultC> = results
                .into_iter()
                .map(|r| RerankResultC {
                    index: r.index,
                    score: r.score,
                    document: r.document
                        .map(|d| CString::new(d).unwrap().into_raw())
                        .unwrap_or(ptr::null_mut()),
                })
                .collect();

            let len = c_results.len();
            let results_ptr = c_results.as_mut_ptr();
            std::mem::forget(c_results);

            Box::into_raw(Box::new(RerankResultVec {
                results: results_ptr,
                len,
            }))
        }
        Err(e) => {
            if !error.is_null() {
                unsafe {
                    *error = FastEmbedError::from_string(format!("Reranking failed: {}", e));
                }
            }
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn fastembed_text_rerank_free(handle: *mut TextRerankHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

// Memory cleanup functions
#[no_mangle]
pub extern "C" fn fastembed_float_array_vec_free(vec: *mut FloatArrayVec) {
    if !vec.is_null() {
        unsafe {
            let vec = Box::from_raw(vec);
            let arrays = Vec::from_raw_parts(vec.arrays, vec.len, vec.len);
            for array in arrays {
                if !array.data.is_null() {
                    let _ = Vec::from_raw_parts(array.data, array.len, array.len);
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn fastembed_sparse_embedding_vec_free(vec: *mut SparseEmbeddingVec) {
    if !vec.is_null() {
        unsafe {
            let vec = Box::from_raw(vec);
            let embeddings = Vec::from_raw_parts(vec.embeddings, vec.len, vec.len);
            for emb in embeddings {
                if !emb.indices.is_null() {
                    let _ = Vec::from_raw_parts(emb.indices, emb.len, emb.len);
                }
                if !emb.values.is_null() {
                    let _ = Vec::from_raw_parts(emb.values, emb.len, emb.len);
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn fastembed_rerank_result_vec_free(vec: *mut RerankResultVec) {
    if !vec.is_null() {
        unsafe {
            let vec = Box::from_raw(vec);
            let results = Vec::from_raw_parts(vec.results, vec.len, vec.len);
            for result in results {
                if !result.document.is_null() {
                    let _ = CString::from_raw(result.document);
                }
            }
        }
    }
}

// Model Information Structures
#[repr(C)]
pub struct ModelInfoC {
    pub model_code: *mut c_char,
    pub description: *mut c_char,
    pub dim: usize,
}

#[repr(C)]
pub struct ModelInfoVec {
    pub models: *mut ModelInfoC,
    pub len: usize,
}

// Text Embedding Model Listing
#[no_mangle]
pub extern "C" fn fastembed_text_embedding_list_supported_models() -> *mut ModelInfoVec {
    let models = TextEmbedding::list_supported_models();
    let mut model_infos = Vec::new();

    for model in models {
        let model_code = CString::new(model.model_code).unwrap_or_else(|_| CString::new("").unwrap());
        let description = CString::new(model.description).unwrap_or_else(|_| CString::new("").unwrap());
        
        model_infos.push(ModelInfoC {
            model_code: model_code.into_raw(),
            description: description.into_raw(),
            dim: model.dim,
        });
    }

    let len = model_infos.len();
    let models_ptr = model_infos.as_mut_ptr();
    std::mem::forget(model_infos);

    Box::into_raw(Box::new(ModelInfoVec {
        models: models_ptr,
        len,
    }))
}

#[no_mangle]
pub extern "C" fn fastembed_model_info_vec_free(vec: *mut ModelInfoVec) {
    if !vec.is_null() {
        unsafe {
            let vec = Box::from_raw(vec);
            let models = Vec::from_raw_parts(vec.models, vec.len, vec.len);
            for model in models {
                if !model.model_code.is_null() {
                    let _ = CString::from_raw(model.model_code);
                }
                if !model.description.is_null() {
                    let _ = CString::from_raw(model.description);
                }
            }
        }
    }
}

// Sparse Text Embedding Model Listing
#[no_mangle]
pub extern "C" fn fastembed_sparse_text_embedding_list_supported_models() -> *mut ModelInfoVec {
    let models = SparseTextEmbedding::list_supported_models();
    let mut model_infos = Vec::new();

    for model in models {
        let model_code = CString::new(model.model_code).unwrap_or_else(|_| CString::new("").unwrap());
        let description = CString::new(model.description).unwrap_or_else(|_| CString::new("").unwrap());
        
        model_infos.push(ModelInfoC {
            model_code: model_code.into_raw(),
            description: description.into_raw(),
            dim: model.dim,
        });
    }

    let len = model_infos.len();
    let models_ptr = model_infos.as_mut_ptr();
    std::mem::forget(model_infos);

    Box::into_raw(Box::new(ModelInfoVec {
        models: models_ptr,
        len,
    }))
}

// Image Embedding Model Listing
#[no_mangle]
pub extern "C" fn fastembed_image_embedding_list_supported_models() -> *mut ModelInfoVec {
    let models = ImageEmbedding::list_supported_models();
    let mut model_infos = Vec::new();

    for model in models {
        let model_code = CString::new(model.model_code).unwrap_or_else(|_| CString::new("").unwrap());
        let description = CString::new(model.description).unwrap_or_else(|_| CString::new("").unwrap());
        
        model_infos.push(ModelInfoC {
            model_code: model_code.into_raw(),
            description: description.into_raw(),
            dim: model.dim,
        });
    }

    let len = model_infos.len();
    let models_ptr = model_infos.as_mut_ptr();
    std::mem::forget(model_infos);

    Box::into_raw(Box::new(ModelInfoVec {
        models: models_ptr,
        len,
    }))
}

// Text Rerank Model Listing
#[no_mangle]
pub extern "C" fn fastembed_text_rerank_list_supported_models() -> *mut ModelInfoVec {
    let models = TextRerank::list_supported_models();
    let mut model_infos = Vec::new();

    for model in models {
        let model_code = CString::new(model.model_code).unwrap_or_else(|_| CString::new("").unwrap());
        let description = CString::new(model.description).unwrap_or_else(|_| CString::new("").unwrap());
        
        model_infos.push(ModelInfoC {
            model_code: model_code.into_raw(),
            description: description.into_raw(),
            dim: 0, // Reranker models don't have dimensions
        });
    }

    let len = model_infos.len();
    let models_ptr = model_infos.as_mut_ptr();
    std::mem::forget(model_infos);

    Box::into_raw(Box::new(ModelInfoVec {
        models: models_ptr,
        len,
    }))
}
