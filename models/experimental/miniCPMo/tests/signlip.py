import torch
import sys


# Workaround for transformers version check issue
# The lerobot package expects a specific transformers version check module
# that doesn't exist in newer transformers versions. We'll monkey-patch it.
class MockCheck:
    @staticmethod
    def check_whether_transformers_replace_is_installed_correctly():
        return True


# Patch the import before it's used
sys.modules["transformers.models.siglip.check"] = MockCheck()

from lerobot.policies.pi0.modeling_pi0 import PI0Policy

# Load the π₀ model on CPU
print("Loading π₀ model (this may take a few minutes on CPU)...")
print("Downloading model files from HuggingFace...")

# PI0Policy is the concrete implementation class
policy = PI0Policy.from_pretrained("lerobot/pi0_base")
print("✓ Model loaded successfully")

# Access the vision encoder (SigLIP)
print("\nExtracting SigLIP vision encoder...")
# The vision tower is at: policy.model.paligemma_with_expert.paligemma.vision_tower
siglip_model = policy.model.paligemma_with_expert.paligemma.vision_tower
import pdb

pdb.set_trace()
print(f"✓ Found SigLIP vision encoder: {type(siglip_model).__name__}")

# Save just the SigLIP weights
print("\nSaving SigLIP weights...")
siglip_state_dict = siglip_model.state_dict()
torch.save(siglip_state_dict, "siglip_weights.pth")
print("✓ Saved SigLIP weights to siglip_weights.pth")

# Print some info about the extracted weights
total_params = sum(p.numel() for p in siglip_state_dict.values())
print(f"\nSigLIP Model Info:")
print(f"  Total parameters: {total_params:,}")
print(f"  Number of weight tensors: {len(siglip_state_dict)}")
print(f"  File size: ~{total_params * 4 / (1024**2):.1f} MB (float32)")

# Show some weight keys
print(f"\nSample weight keys (showing first 5):")
for i, key in enumerate(list(siglip_state_dict.keys())[:5]):
    shape = tuple(siglip_state_dict[key].shape)
    print(f"  {i+1}. {key}: {shape}")
