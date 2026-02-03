# MiniCPM-o 2.6 on Tenstorrent

This directory contains the Tenstorrent (TT) accelerated implementation of [MiniCPM-o 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6), an end-to-end multimodal model capable of vision understanding, speech understanding, and speech generation.

## Model Overview

MiniCPM-o 2.6 is a multimodal large language model developed by OpenBMB that supports:

- **Vision Understanding**: Process images and answer questions about visual content
- **Audio Understanding**: Transcribe and comprehend spoken audio input
- **Audio Generation (Mimick)**: Generate speech output that mimics the style and content of input audio

The model architecture consists of several key components:

| Component | Description |
|-----------|-------------|
| Qwen2 LLM | Core language model for text generation |
| SigLip Vision Encoder | Vision transformer with NaViT dynamic position embeddings |
| Whisper Audio Encoder | Audio encoder for speech-to-embedding conversion |
| ChatTTS Decoder | Text-to-speech decoder for audio generation |
| DVAE | Discrete variational autoencoder for audio code generation |
| Resampler | Projects vision/audio embeddings to LLM dimension |



## TT-Accelerated Modules

The following modules have TT implementations for accelerated inference:

| Module | Test File | PCC |
|--------|-----------|-----|
| Qwen2 LLM (MLP) | `tests/test_mlp_layer.py` | 0.999 |
| Qwen2 LLM (Attention) | `tests/test_multi_head_attn.py` | 0.99 |
| SigLip Vision Encoder | `tests/test_siglip.py` | 0.97 |
| SigLip Vision Embedding | `tests/test_siglip_vision_emb.py` | 0.99 |
| Whisper Audio Encoder | `tests/test_whisper.py` | 0.997 |
| Whisper Attention | `tests/test_whisper_attn.py` | 0.999 |
| Whisper Projection | `tests/test_whisper_projection.py` | 0.999 |
| ChatTTS Decoder | `tests/test_tts_decoder.py` | 0.90 |
| DVAE Encoder/Decoder | `tests/test_ttnn_dvae.py` | 0.99 |
| Resampler | `tests/test_resampler.py` | 0.99 |

Supporting infrastructure:

| File | Purpose |
|------|---------|
| `tt/drop_in_replacements.py` | Drop-in replacement classes for HuggingFace components |
| `tt/tt_model_wrapper.py` | Wrapper utilities to enable TT acceleration |
| `tt/model_setup.py` | Model file download and setup utilities |
| `tt/minicpm_weight_bridge.py` | Weight extraction and conversion from HuggingFace format |
| `tt_transformers/` | Core TT transformer implementations (attention, MLP, decoder blocks) |

## Setup

### Prerequisites

1. Tenstorrent hardware (N150 or N300) with TT-Metalium installed
2. Python 3.10+
3. Set up the TT environment:

```bash
source ~/quickstart.sh
```

### Install Dependencies

Install Python dependencies before running any demos or tests:

```bash
pip install -r models/experimental/miniCPMo/demo/requirements.txt
```

Required packages include:
- torch, torchaudio, torchvision
- transformers
- Pillow
- librosa, soundfile
- vector-quantize-pytorch
- vocos
- decord, moviepy

### Model Files

Model files are downloaded automatically on first run. The `ensure_model_files()` function downloads:
- Model weights (17GB in 4 safetensor shards)
- Tokenizer files

Files are stored in `models/experimental/miniCPMo/reference/`.

## Architecture: Drop-in Replacements

The TT acceleration uses a drop-in replacement pattern that allows seamless integration with the HuggingFace pipeline. This means you can load the model using standard HuggingFace APIs and then replace specific components with TT-accelerated versions.

### How It Works

1. **Load the model from HuggingFace** using `AutoModel.from_pretrained()`:

```python
from transformers import AutoModel, AutoTokenizer
from models.experimental.miniCPMo.tt.model_setup import ensure_model_files, REFERENCE_DIR

ensure_model_files()
model = AutoModel.from_pretrained(
    str(REFERENCE_DIR),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    init_vision=True,
)
```

2. **Enable TT acceleration** using `enable_tt_acceleration()`:

```python
from models.experimental.miniCPMo.tt.tt_model_wrapper import enable_tt_acceleration

device = ttnn.open_device(device_id=0)
model = enable_tt_acceleration(
    model,
    device,
    components=["vision", "llm"],  # Components to accelerate
    model_path=str(REFERENCE_DIR),
)
```

3. **Use the model normally** - TT components are transparent drop-in replacements:

```python
result = model.chat(msgs=msgs, tokenizer=tokenizer, max_new_tokens=128)
```

### Drop-in Replacement Classes

The `tt/drop_in_replacements.py` module provides drop-in classes that wrap TT implementations with the same interface as HuggingFace components:

| Class | Replaces | Purpose |
|-------|----------|---------|
| `DropInQwen2LLM` | `model.llm` | TT-accelerated LLM generation |
| `DropInVisionEncoder` | `model.vpm` | TT-accelerated vision encoder |
| `DropInAudioEncoder` | `model.apm` | TT-accelerated Whisper encoder |
| `DropInChatTTSDecoder` | `model.tts` | TT-accelerated TTS decoder |
| `DropInDVAE` | `model.tts.dvae` | TT-accelerated DVAE |

Each drop-in class:
- Wraps the reference model and TT implementation
- Provides the same interface as the HuggingFace component
- Forwards unknown attributes to the reference model for compatibility
- Uses TT hardware for compute-intensive operations

### TT Model Wrapper

The `tt/tt_model_wrapper.py` module provides two main functions:

**`enable_tt_acceleration(model, device, components, model_path)`**

Replaces specified model components with TT implementations. Available components:
- `"llm"`: Replace Qwen2 LLM (main computational bottleneck)
- `"vision"`: Replace SigLip vision encoder
- `"audio"`: Replace Whisper audio encoder
- `"tts"`: Replace ChatTTS decoder
- `"dvae"`: Replace DVAE (inside TTS module)

**`load_minicpmo_with_tt(model_path, device, init_vision, init_audio, init_tts, tt_components)`**

Convenience function that loads the model and enables TT acceleration in one call.

## Demo Scripts

### Vision Understanding Demo

Processes an image and generates a text description.

```bash
python models/experimental/miniCPMo/demo/demo_image.py
```

**What it does:**
1. Loads MiniCPM-o with vision module initialized
2. Enables TT acceleration for vision encoder and LLM
3. Loads a sample image (cat image from sample_data folder)
4. Runs vision understanding to describe the image content

**TT Components Used:** Vision Encoder, LLM

### Audio Understanding Demo

Processes audio input and generates a text description.

```bash
python models/experimental/miniCPMo/demo/demo_audio_understanding.py
```

**What it does:**
1. Loads MiniCPM-o with audio module initialized
2. Enables TT acceleration for audio encoder and LLM
3. Downloads audio asset on first run (stored in `assets/` folder)
4. Runs audio understanding to describe the audio content

**TT Components Used:** Audio Encoder, LLM

### Audio Mimick Demo

Processes audio input and generates speech output that mimics the input.

```bash
python models/experimental/miniCPMo/demo/demo_audio_mimick.py
```

**What it does:**
1. Loads MiniCPM-o with audio and TTS modules initialized
2. Enables TT acceleration for TTS decoder, DVAE, audio encoder, and LLM
3. Downloads audio asset on first run
4. Runs mimick task to transcribe and regenerate the audio
5. Saves output audio to `result_mimick_full_demo.wav`

**TT Components Used:** Audio Encoder, LLM, TTS Decoder, DVAE

## Running Tests

### Run All Tests

```bash
pytest models/experimental/miniCPMo/tests/ -v -s
```

### Run Specific Test Files

**MLP Layer Test:**
```bash
pytest models/experimental/miniCPMo/tests/test_mlp_layer.py -v -s
```

**Multi-Head Attention Test:**
```bash
pytest models/experimental/miniCPMo/tests/test_multi_head_attn.py -v -s
```

**SigLip Vision Encoder Test:**
```bash
pytest models/experimental/miniCPMo/tests/test_siglip.py -v -s
```

**SigLip Vision Embedding Test:**
```bash
pytest models/experimental/miniCPMo/tests/test_siglip_vision_emb.py -v -s
```

**Whisper Audio Encoder Test:**
```bash
pytest models/experimental/miniCPMo/tests/test_whisper.py -v -s
```

**Whisper Attention Test:**
```bash
pytest models/experimental/miniCPMo/tests/test_whisper_attn.py -v -s
```

**Whisper Projection Test:**
```bash
pytest models/experimental/miniCPMo/tests/test_whisper_projection.py -v -s
```

**DVAE Tests:**
```bash
pytest models/experimental/miniCPMo/tests/test_ttnn_dvae.py -v -s
pytest models/experimental/miniCPMo/tests/test_ttnn_dvae_encoder.py -v -s
pytest models/experimental/miniCPMo/tests/test_ttnn_dvae_decoder.py -v -s
```

**TTS Decoder Test:**
```bash
pytest models/experimental/miniCPMo/tests/test_tts_decoder.py -v -s
```

**Resampler Test:**
```bash
pytest models/experimental/miniCPMo/tests/test_resampler.py -v -s
```

**Qwen2 LLM Audio Test:**
```bash
pytest models/experimental/miniCPMo/tests/test_tt_qwen2_audio.py -v -s
```

### Test Descriptions

| Test File | Component Tested | Description |
|-----------|------------------|-------------|
| `test_mlp_layer.py` | Qwen2 MLP | Tests MLP layer forward pass accuracy |
| `test_multi_head_attn.py` | Qwen2 Attention | Tests multi-head attention accuracy |
| `test_siglip.py` | SigLip Vision | Tests full vision encoder |
| `test_siglip_vision_emb.py` | SigLip Embeddings | Tests vision embedding layer |
| `test_whisper.py` | Whisper Encoder | Tests full Whisper encoder |
| `test_whisper_attn.py` | Whisper Attention | Tests Whisper attention layer |
| `test_whisper_projection.py` | Whisper Projection | Tests Whisper projection layer |
| `test_ttnn_dvae.py` | DVAE | Tests full DVAE encode/decode |
| `test_ttnn_dvae_encoder.py` | DVAE Encoder | Tests DVAE encoder |
| `test_ttnn_dvae_decoder.py` | DVAE Decoder | Tests DVAE decoder |
| `test_tts_decoder.py` | TTS Decoder | Tests ChatTTS decoder |
| `test_resampler.py` | Resampler | Tests embedding resampler |
| `test_tt_qwen2_audio.py` | Qwen2 LLM | Tests LLM generation with audio inputs |

## Directory Structure

```
models/experimental/miniCPMo/
|-- README.md                    # This file
|-- demo/
|   |-- demo_image.py            # Vision understanding demo
|   |-- demo_audio_understanding.py  # Audio understanding demo
|   |-- demo_audio_mimick.py     # Audio mimick demo
|   |-- requirements.txt         # Python dependencies
|-- reference/                   # Local model files (downloaded on first run)
|   |-- config.json              # Model configuration
|   |-- modeling_minicpmo.py     # Patched modeling code (no flash_attn)
|   |-- model-*.safetensors      # Model weights (downloaded)
|   |-- tokenizer.json           # Tokenizer (downloaded)
|-- assets/                      # Audio assets (downloaded on first run)
|   |-- input_examples/
|       |-- audio_understanding.mp3
|       |-- Trump_WEF_2018_10s.mp3
|-- tt/
|   |-- drop_in_replacements.py  # Drop-in replacement classes
|   |-- tt_model_wrapper.py      # Wrapper utilities
|   |-- model_setup.py           # Model download utilities
|   |-- tt_qwen2_for_causal_lm.py    # TT Qwen2 LLM implementation
|   |-- ttnn_siglip_vision.py    # TT SigLip vision encoder
|   |-- ttnn_whisper_encoder.py  # TT Whisper audio encoder
|   |-- ttnn_chattts_decoder.py  # TT ChatTTS decoder
|   |-- ttnn_dvae.py             # TT DVAE
|   |-- minicpm_weight_bridge.py # Weight conversion utilities
|-- tt_transformers/             # Core TT transformer implementations
|   |-- attention.py             # TT attention layer
|   |-- mlp.py                   # TT MLP layer
|   |-- decoder.py               # TT transformer decoder block
|   |-- model.py                 # TT transformer model
|-- tests/                       # Unit tests
```

## Notes

- The reference folder contains patched Python files that remove the `flash_attn` dependency, allowing the model to run without that optional package.
- Model weights and tokenizer files are downloaded on first run from HuggingFace.
- Audio assets for demos are also downloaded on first run.
- All TT implementations use bfloat16 precision for optimal performance on TT hardware.
