# Stable Diffusion 3.5 Medium

## Introduction

[Stable Diffusion 3.5 Medium](https://stability.ai/news/introducing-stable-diffusion-3-5) is a generative model for text-guided image synthesis.
The Medium version is a lighter and faster variant of SD3.5, optimized for performance on Wormhole systems using TT-Metal.

## Details

The architecture follows the paper
[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).

The model consists of CLIP text encoders with their tokenizers, an optional T5 text encoder, a scheduler, a compact MMDiT transformer and a VAE decoder.
The MMDiT transformer is built from spatial, prompt and time embeddings together with a stack of 24 transformer blocks.
Attention layers operate either on the spatial embedding alone or jointly on the spatial and prompt embeddings.

Compared to SD3.5-Large, the Medium variant is smaller, faster, and requires less memory while preserving the same core architecture.

## Performance

Performance measurements from the performance test suite:

### N150 (1x1 mesh) - 512×512, 40 steps, Guidance Scale=7.0

- CLIP encoding: ~0.07s
- T5 encoding: ~0.00s (disabled)
- Total encoding: ~0.15s
- Denoising (per step): ~0.60s
- Total denoising: ~24.12s
- VAE decoding: ~0.34s
- **Total pipeline runtime: ~24.63s**
- Throughput: ~0.0406 images/second

### N150 (1x1 mesh) - 1024×1024, 40 steps, Guidance Scale=7.0

- CLIP encoding: ~0.06s
- T5 encoding: ~0.00s (disabled)
- Total encoding: ~0.14s
- Denoising (per step): ~2.63s
- Total denoising: ~105.23s
- VAE decoding: ~7.86s
- **Total pipeline runtime: ~113.27s**
- Throughput: ~0.0088 images/second

### N300 (1x2 mesh) - 512×512, 40 steps, Guidance Scale=7.0

- CLIP encoding: ~0.07s
- T5 encoding: ~0.00s (disabled)
- Total encoding: ~0.15s
- Denoising (per step): ~0.71s
- Total denoising: ~28.31s
- VAE decoding: ~0.34s
- **Total pipeline runtime: ~28.83s**
- Throughput: ~0.0347 images/second

### N300 (1x2 mesh) - 1024×1024, 40 steps, Guidance Scale=7.0

- CLIP encoding: ~0.07s
- T5 encoding: ~0.00s (disabled)
- Total encoding: ~0.15s
- Denoising (per step): ~2.68s
- Total denoising: ~107.27s
- VAE decoding: ~8.06s
- **Total pipeline runtime: ~115.52s**
- Throughput: ~0.0087 images/second

*Note: Performance numbers are measured averages across multiple runs. Actual performance may vary based on system configuration and workload.*

## Prerequisites

- Request access to SD3.5-Medium on HuggingFace

## How to Run

1. Request access to the model on HuggingFace (required for gated weights):
   https://huggingface.co/stabilityai/stable-diffusion-3.5-medium

2. Authenticate with your HuggingFace token:

```bash
huggingface-cli login
```

3. Activate the repo virtual environment and set `PYTHONPATH`:

```bash
source python_env/bin/activate
export PYTHONPATH=$(pwd)
```

4. Run the SD3.5 Medium pipeline test:

```bash
# Run on N150 (1x1 mesh) at 512x512
pytest models/experimental/tt_dit/tests/models/sd35_med/test_pipeline_sd35_medium.py -k "1x1_n150 and 512x512"

# Run on N150 (1x1 mesh) at 1024x1024
pytest models/experimental/tt_dit/tests/models/sd35_med/test_pipeline_sd35_medium.py -k "1x1_n150 and 1024x1024"

# Run on N300 (1x2 mesh) at 512x512
pytest models/experimental/tt_dit/tests/models/sd35_med/test_pipeline_sd35_medium.py -k "1x2_n300 and 512x512"

# Run on N300 (1x2 mesh) at 1024x1024
pytest models/experimental/tt_dit/tests/models/sd35_med/test_pipeline_sd35_medium.py -k "1x2_n300 and 1024x1024"
```

5. (Optional) Provide custom prompts via command line arguments or environment variables:

```bash
# Using command line arguments
pytest models/experimental/tt_dit/tests/models/sd35_med/test_pipeline_sd35_medium.py --prompt "your custom prompt" --negative-prompt "your negative prompt"

# Using environment variables
PROMPT="your custom prompt" NEGATIVE_PROMPT="your negative prompt" pytest models/experimental/tt_dit/tests/models/sd35_med/test_pipeline_sd35_medium.py
```

## Performance Testing

The SD3.5 Medium pipeline includes a comprehensive performance test suite that measures and reports detailed timing metrics for all pipeline stages.

### Running Performance Tests

```bash
# Run performance test on N150 (1x1 mesh) at 512x512
pytest models/experimental/tt_dit/tests/models/sd35_med/test_performance_sd35_medium.py -k "1x1_n150 and 512"

# Run performance test on N150 (1x1 mesh) at 1024x1024
pytest models/experimental/tt_dit/tests/models/sd35_med/test_performance_sd35_medium.py -k "1x1_n150 and 1024"

# Run performance test on N300 (1x2 mesh) at 512x512
pytest models/experimental/tt_dit/tests/models/sd35_med/test_performance_sd35_medium.py -k "1x2_n300 and 512"

# Run performance test on N300 (1x2 mesh) at 1024x1024
pytest models/experimental/tt_dit/tests/models/sd35_med/test_performance_sd35_medium.py -k "1x2_n300 and 1024"
```

### Performance Test Features

The performance test suite provides:

- **Warmup Run**: Initial run to warm up the pipeline and devices
- **Multiple Iterations**: 4 performance measurement runs with different prompts
- **Detailed Timing Breakdown**:
  - CLIP encoding time
  - T5 encoding time (if enabled)
  - Total encoding time
  - Individual denoising step times
  - VAE decoding time
  - Total pipeline runtime
- **Statistical Analysis**: Mean, standard deviation, min, and max for all timing metrics
- **Throughput Metrics**: Steps per second and images per second
- **Performance Benchmarks**: Automatic performance analysis with thresholds
- **CI Integration**: Benchmark data collection for continuous performance monitoring

### Performance Test Output

The test generates:
- **Console Output**: Detailed performance report with statistics and analysis
- **Image Files**: Generated images saved as `sd35_medium_{width}x{height}_warmup.png` and `sd35_medium_{width}x{height}_perf_run{N}.png`
- **Benchmark Data**: JSON files for CI integration (when running in CI environment)

### Performance Test Configurations

The performance test runs with the following configurations:
- **Mesh Configurations**: N150 (1x1) and N300 (1x2)
- **Image Sizes**: 512×512 and 1024×1024
- **Fabric Configurations**: DISABLED and FABRIC_1D (FABRIC_1D skipped for single-device configurations)
- **Inference Steps**: 40 steps
- **Guidance Scale**: 7.0

Note: The test automatically skips FABRIC_1D configuration for single-device (1x1) mesh setups, as fabric requires multiple devices.

## Scalability

SD3.5-Medium has been implemented to support execution on:
- **N150** (1x1 mesh): Single device configuration without CFG parallelism
- **N300** (1x2 mesh): Two devices with CFG parallelism

The model has been tested on Wormhole systems.

The DiT model can be parallelized on the following axes:
1. **CFG (classifier-free guidance)**: Execute conditional and unconditional steps in parallel (factor 2 on N300)
2. **SP (sequence parallel)**: The input sequence is fractured across a mesh axis (not currently used in Medium)
3. **TP (tensor parallel)**: Weights are fractured across a mesh axis (not currently used in Medium)

The text embedding models (CLIP encoders) and the VAE decoder are parallelized with tensor parallelism when multiple devices are available.

## Architecture Details

- **Transformer**: 24 transformer blocks with 1536 hidden dimensions and 24 attention heads
- **Text Encoders**: CLIP text encoders (required) and optional T5 text encoder
- **VAE Decoder**: Fully implemented in TTNN for efficient decoding
- **Image Sizes**: Supports 512×512 and 1024×1024 output resolutions
- **Inference Steps**: Configurable (default: 40 steps)

## Output

Generated images are saved as `sd35_medium_tt_output_{width}x{height}.png` in the current working directory.
