# Transformer-based Selective Super-Resolution for Efficient Image Refinement (SSR)

## Platforms:
    Wormhole (n150)

## Introduction
Selective Super-Resolution (SSR) is a transformer-based framework that partitions an image into tiles, uses a multi‑scale pyramid to select only object‑relevant regions, and applies deep refinement exclusively where it matters, avoiding background over‑sharpening. By skipping heavy processing on unimportant tiles, SSR cuts computation roughly 40% while improving visual fidelity for downstream tasks, achieving large quality gains such as reducing FID on BDD100K from 26.78 to 10.41.

[Link to paper](https://arxiv.org/abs/2312.05803)

**NOTE:** Trained weights are not available at this time. The implementation uses random weights to ensure correctness against the reference implementation.

## Structure of model

```
SSR
├─ TileSelection
│  ├─ Patch Embed
│  ├─ MaskTokenInference
│  └─ Basic Layer
│     ├─ PatchMerging
│     └─ Swin Transformer Block
│        ├─ Window Attention
│        └─ MLP
└ TileRefinement
    └─ HAT
       ├─ Patch Embed
       ├─ Patch Unembed
       ├─ RHAG
       │  └─ Attention Blocks
       │     ├─ HAB
       │     │  ├─ Window Attention
       │     │  ├─ CAB
       │     │  │  └─ Channel Attention
       │     │  └─ MLP
       │     └─ OCAB
       │        └─ MLP
       ├─ Patch Embed
       ├─ Patch Unembed
       └─ Upsample
```

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run
### For performance run (Slightly lesser PCC, lesser device time):
```
pytest models/experimental/SSR/tests/test_ssr.py -k "performance"
```
### For accuracy run (Slightly more PCC, more device time):
```
pytest models/experimental/SSR/tests/test_ssr.py -k "accuracy"
```


### Demo
```
python models/experimental/SSR/demo/ssr_demo.py --accuracy --depth 6 --num_heads 6
```
NOTE: If --input is not provided, the demo uses the default image located at models/experimental/SSR/demo/images/ssr_test_image.jpg. Make sure this file exists or the demo will fail.

## Details
- The entry point to the `SSR` is located at:`models/experimental/SSR/tt/ssr.py`.
- Batch Size : `1` (Single Device).
- Supported Input Resolution - `(256, 256)` - (Height, Width).
