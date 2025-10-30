# GPU Acceleration Status

## Current Implementation

This library **attempts to use CoreML with Apple Neural Engine (ANE)** for hardware acceleration on macOS, but in practice, **performance is worse than CPU-only execution**.

## CoreML Configuration

We configure CoreML with ANE-specific optimizations:

- `CPUAndNeuralEngine` compute units (avoiding slow CoreML GPU)
- `MLProgram` model format for better operator support
- `FastPrediction` specialization strategy
- Subgraph support enabled
- Compute plan profiling enabled

## Performance Reality

Despite CoreML being available and configured, **it provides negative performance** for these models:

### CPU-Only Performance (ort without CoreML)

- Text Embedding: ~9-12ms/text (batch of 100)
- Sparse Embedding: ~40-47ms/text (batch of 100)
- Reranking: ~56-67ms/doc (batch of 99)

### With CoreML ANE Enabled

- Text Embedding: ~52ms/text (batch of 100) - **5.4x SLOWER**
- Sparse Embedding: ~126ms/text (batch of 100) - **3x SLOWER**
- Reranking: ~294ms/doc (batch of 99) - **5.2x SLOWER**

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
