# GPU Acceleration Status

## Current Implementation

This library uses **CPU-only inference** with ONNX Runtime, optimized for maximum performance on CPU.

## Why No GPU Acceleration?

We initially attempted to add macOS GPU acceleration using CoreML execution provider, but discovered:

### CoreML Execution Provider Limitations

1. **Poor Transformer Support**: BERT-style transformer models (which FastEmbed uses) contain many ONNX operators that CoreML doesn't support
2. **Silent Fallback**: Even when CoreML is "available", it silently falls back to CPU for unsupported operators
3. **No Performance Gain**: In testing, CoreML showed 0% GPU utilization despite being configured
4. **Known Issue**: See [ONNX Runtime Issue #16934](https://github.com/microsoft/onnxruntime/issues/16934)

### What We Tested

- ✅ CoreML execution provider configuration
- ✅ MLProgram model format
- ✅ Static input shapes
- ✅ Compute profiling enabled
- ❌ Result: No GPU utilization, CPU fallback

## Current Performance

Even on CPU-only, the performance is respectable:

- **Text Embedding**: ~48-52ms/text (batch size 44-50)
- **Sparse Embedding**: ~200-210ms/text (batch size 44-50)
- **Total per document**: ~250ms (dense + sparse)

With larger batches (100 texts):
- **Text Embedding**: ~9-12ms/text
- **Sparse Embedding**: ~40-47ms/text

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
