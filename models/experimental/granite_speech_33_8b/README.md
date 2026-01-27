# Granite-speech-3.3-8b

## Platforms:
Wormhole (n150)

## Introduction
Granite Speech is a state-of-the-art speech-to-text model developed by IBM, capable of transcribing audio inputs into written text. The model combines a Conformer-based encoder for processing audio features with a transformer-based decoder for text generation. It supports multimodal inputs, allowing both audio and text prompts to be processed together. The model architecture includes a CTC encoder for audio feature extraction, a projector layer for cross-modal attention, and a language model(granite-3.3-8b-instruct) decoder for text generation.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Login to huggingface with: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
   - To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens
- Model repository: [ibm-granite/granite-speech-3.3-8b](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)
- Install `torchaudio==2.7.1+cpu` in python_env

## How to Run
- Use the following command to run the `Granite-speech-3.3-8b` model:
  ```
  python3 models/experimental/granite_speech_33_8b/tt/utils.py (Optional)
  export HF_MODEL="granite_instruct_weights_from_speech"
  pytest models/experimental/granite_speech_33_8b/test/test_granite.py
  ```

## Blockwise Testing

### Encoder Block Test
- Use the following command to test the encoder block:
  ```
  pytest models/experimental/granite_speech_33_8b/test/test_encoder_block.py::test_encoder_block
  ```

### Projector Block Test
- Use the following command to test the projector block:
  ```
  pytest models/experimental/granite_speech_33_8b/test/test_projector_block.py::test_projector_output
  ```

## Details
- Entry point for the model is `models/experimental/granite_speech_33_8b/tt/granite_speech.py`
- Batch Size: `1` (Single Device)
- Model: `ibm-granite/granite-speech-3.3-8b` from HuggingFace
- Used tt_trasformers to run granite speech language model(granite-3.3-8b-instruct)
- Audio Input: Mono channel, 16kHz sample rate

## TODO
- The Granite Speech language model uses LoRA adapters for accurate results; however, LoRA adapters are not currently supported because the TTNN model is using tt_transformer.

## Issues

- [36541](https://github.com/tenstorrent/tt-metal/issues/36541) - N300 hang issue
