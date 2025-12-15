# FLUX.1 Kontext

## Introduction:
[FLUX.1 Kontext](https://bfl.ai/models/flux-kontext) is Black Forest Labs’ in‑context image generation and editing family that unifies text‑to‑image and image‑guided editing in a single rectified‑flow transformer. It takes text + image inputs, performs local or global edits, preserves character/style consistency, and supports iterative, multi‑turn workflows at interactive speeds.It is a 12 billion parameter rectified flow transformer capable of editing images based on text instructions.


## Details

Core architecture is described in [Flow Matching for In-Context Image Generation](https://arxiv.org/pdf/2506.15742), based on a 12B-parameter Rectified-Flow Transformer (Flux DiT), paired with dual text encoders (CLIP-L, T5-XXL), a VAE autoencoder, and custom scheduler. Prompt + image + time positional embeddings feed transformer blocks with spatial or joint self-attention.

Unlike diffusion models using CFG, Kontext uses [**learned guidance embeddings**](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
, running only conditional paths for efficiency.

## Performance
- **FLUX.1 Kontext [dev]** — open weights, ~12B rectified‑flow transformer for text‑guided editing & generation (non‑commercial license).
Current performance and target performance for two systems are detailed below. Performance is measured in seconds per image, where the image size is 1024x1024px.


### Dev Variant (28 steps)

| System    | CFG | SP | TP | Current Performance |
|-----------|-----|----|----|---------------------|
| LoudBox   | 1   | 2  | 4  | 69.48s              |
| LoudBox   | 2   | 1  | 4  | 62.13s              |

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
1. Visit [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) to grant access to the model weights
2. Login with the HuggingFace token: `huggingface-cli login`

```bash
# [Install tt-metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

# Set the directory to cache the weights to speed up future runs
export TT_DIT_CACHE_DIR=/your/cache/path

# Run the dev variant on LoudBox (2x4 mesh)
pytest models/experimental/tt_dit/tests/models/flux1_kontext/test_pipeline_flux1_kontext.py -k "dev and 2x4sp0tp1 and traced and encoder_device"

```

## Scalability
FLUX.1 Kontext is engineered for high scalability across heterogeneous compute environments, from single-GPU setups to large multi-chip meshes. It supports execution on configurations such as 4-chip (1×4 mesh). This flexibility allows developers to deploy the model on both workstation-class hardware and enterprise-grade clusters without architectural changes. The design ensures that image generation and in-context editing tasks remain efficient even as workloads scale.

The model achieves this through advanced parallelization strategies. Two primary axes of parallelism are employed: sequence parallelism `(sp) `and tensor parallelism `(tp)`. Sequence parallelism fractures the input sequence across a mesh axis, enabling FeedForward layers to execute in parallel across sequence chunks. Attention layers use ring attention, which overlaps KV all-gather operations with computation to minimize latency. Tensor parallelism, on the other hand, distributes model weights across another axis, leveraging collective communication primitives like AllGather and ReduceScatter for synchronization. These techniques ensure balanced workload distribution and high throughput for both text-to-image generation and iterative editing.

In addition to `sp` and `tp`, FLUX.1 Kontext introduces ring parallelism `(rp)` and ulysses parallelism `(up)`, which complement `sp` and `tp` for attention modules, further improving efficiency in large-scale deployments. Parallel configurations are defined as tuples like `((sp_factor, sp_axis), (tp_factor, tp_axis))`.

For example, a 2×4 mesh uses `((2, 0), (4, 1))`.

The text encoders (CLIP-L and T5-XXL) and the VAE decoder also benefit from tensor parallelism, ensuring that both conditioning and decoding stages scale effectively. This architecture enables FLUX.1 Kontext to deliver consistent performance and quality across diverse hardware setups, making it suitable for research, production, and enterprise environments.
