# bevformerv2 (experimental)

This directory mirrors the common layout used by other experimental models (for example `models/experimental/bloom` or `models/experimental/yolov5`). Populate the subfolders with the appropriate assets as the bring-up progresses.

- `demo/` – runnable scripts and configs for end-to-end demos or quick validation.
- `reference/` – source references, checkpoints, or conversion utilities that define the expected behaviour.
- `tests/` – unit/performance tests that exercise the `tt/` operators and the assembled pipelines.
- `tt/` – Tenstorrent kernels, layers, and model-building code that map the reference model to TT hardware.

Add additional folders (e.g. `scripts/`, `utils/`, `model_params/`) if the model ends up needing them, keeping parity with other models when possible.
