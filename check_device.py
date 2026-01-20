#!/usr/bin/env python3
"""Helper script to check if a PyTorch model is running on GPU or CPU."""

import torch


def check_model_device(model):
    """Check which device a PyTorch model is on."""
    if not hasattr(model, 'parameters'):
        return "Not a PyTorch model"
    
    try:
        # Get the device of the first parameter
        first_param = next(model.parameters())
        device = first_param.device
        
        if device.type == 'cuda':
            return f"GPU ({device})"
        else:
            return f"CPU ({device})"
    except StopIteration:
        return "Model has no parameters"


def check_cuda_status():
    """Check CUDA availability and status."""
    print("=" * 50)
    print("CUDA Status Check")
    print("=" * 50)
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    else:
        print("CUDA not available - using CPU")
    print("=" * 50)


if __name__ == "__main__":
    check_cuda_status()
    
    # Example: Check a model if you have one
    # from openpi.policies import policy_config
    # from openpi.training import config as _config
    # 
    # train_config = _config.get_config("pi0_aloha")
    # policy = policy_config.create_trained_policy(
    #     train_config, 
    #     "src/openpi/models_pytorch/checkpoint",
    #     pytorch_device="cuda"  # or "cpu"
    # )
    # 
    # print(f"\nModel device: {check_model_device(policy._model)}")
