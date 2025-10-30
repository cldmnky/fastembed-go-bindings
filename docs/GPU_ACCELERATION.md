# GPU Acceleration Status

## Current Implementation

This library **includes CoreML execution provider** configured to use all available compute units (CPU + GPU + ANE), but **CPU-only execution is 6-7x faster**.

## Performance Comparison (Batch of 100)

| Configuration | Text Embedding | Sparse Embedding | Reranking | vs CPU-only |
|--------------|----------------|------------------|-----------|-------------|
| **CPU-only** | 9.5ms/text | 41ms/text | 56ms/doc | **Baseline** |
| **CoreML ANE-only** | 52ms/text | 126ms/text | 294ms/doc | 5.4x slower |
| **CoreML All (CPU+GPU+ANE)** | 63ms/text | 179ms/text | 382ms/doc | **6.6x slower** |

## CoreML Configuration

Currently configured with:

- `CoreMLComputeUnits::All` - Allows use of CPU, GPU, and ANE
- `MLProgram` model format for better operator support
- `FastPrediction` specialization strategy
- Subgraph support enabled
- Compute plan profiling enabled

## Why CoreML Makes Things Worse

1. **Operator Compatibility**: Most BERT transformer operators aren't supported by CoreML/ANE
2. **Compilation Overhead**: CoreML adds model compilation and dispatch overhead  
3. **CPU Fallback**: Unsupported operators fall back to CPU, but with added latency
4. **Known Limitation**: See [ONNX Runtime Issue #16934](https://github.com/microsoft/onnxruntime/issues/16934)

## Recommendation

**Disable CoreML for production use** to get the best performance. The library includes CoreML support for experimental purposes, but CPU-only execution is 3-5x faster.

To disable CoreML and use pure CPU:
- Remove `coreml` from ort features in Cargo.toml
- Remove CoreML configuration from model initialization

## Alternatives for GPU Acceleration

If you need GPU acceleration for transformer models on macOS, consider:

### 1. **Apple MLX** (Recommended for macOS)

- Native Metal support
- Optimized for Apple Silicon
- Good transformer support
- Requires Python or Swift

### 2. **PyTorch with MPS Backend**

- Metal Performance Shaders support
- Good transformer support
- Requires Python

### 3. **CUDA (on Linux/Windows)**

- Use ONNX Runtime with CUDA EP
- Best GPU support for transformers
- Not available on macOS

## Alternatives for GPU Acceleration

If you need GPU acceleration for transformer models on macOS, consider:

### 1. **Apple MLX** (Recommended for macOS)
- Native Metal support
- Optimized for Apple Silicon
- Good transformer support
- Requires Python or Swift

### 2. **PyTorch with MPS Backend**
- Metal Performance Shaders support
- Good transformer support
- Requires Python

### 3. **CUDA (on Linux/Windows)**
- Use ONNX Runtime with CUDA EP
- Best GPU support for transformers
- Not available on macOS

## Future Work

Potential paths forward:

1. **Wait for ONNX Runtime improvements** to CoreML EP transformer support
2. **Implement custom Metal kernels** for specific operations
3. **Integrate MLX backend** (would require significant architectural changes)
4. **Use quantization** to speed up CPU inference further

## Recommendations

For now, to optimize performance with CPU-only execution:

1. **Use larger batch sizes** (100+ documents when possible) to amortize overhead
2. **Consider quantized models** if FastEmbed supports them in the future
3. **Run on Apple Silicon Macs** for better CPU performance vs Intel
4. **Parallelize across multiple cores** at the application level

## References

- [ONNX Runtime CoreML EP Documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [GitHub Issue: CoreML Running on CPU](https://github.com/microsoft/onnxruntime/issues/16934)
- [Apple MLX](https://github.com/ml-explore/mlx)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
