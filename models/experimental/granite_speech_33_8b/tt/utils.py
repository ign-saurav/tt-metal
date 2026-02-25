from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from loguru import logger


def save_language_model_weights(weights_dir="granite_instruct_weights_from_speech"):
    """
    Instantiate the GraniteSpeech model and save the language_model weights.

    Args:
        weights_dir: Directory name where weights will be saved
    """
    logger.info(f"Loading GraniteSpeech model to extract language_model weights...")
    torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "ibm-granite/granite-speech-3.3-8b", torch_dtype=torch.bfloat16
    )
    torch_model.eval()

    processor = AutoProcessor.from_pretrained("ibm-granite/granite-speech-3.3-8b")
    tokenizer = processor.tokenizer

    if not hasattr(torch_model, "language_model"):
        raise AttributeError("Model does not have 'language_model' attribute")

    logger.info(f"Saving language_model weights to '{weights_dir}'...")
    torch_model.language_model.save_pretrained(weights_dir)
    tokenizer.save_pretrained(weights_dir)
    logger.info(f"Successfully saved language_model weights to '{weights_dir}'")


if __name__ == "__main__":
    save_language_model_weights()
