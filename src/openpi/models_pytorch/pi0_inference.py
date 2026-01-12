import torch
from safetensors.torch import load_file
from PIL import Image
import torchvision.transforms as T
import numpy as np

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
from openpi.models_pytorch.tokenizer_jax import PaligemmaTokenizer
from openpi.models_pytorch.pi0_config_jax import Pi0Config



pil_to_tensor = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),                     # (C, H, W)
])



class Observation:
    def __init__(self, images, prompt, state,
                 tokenized_prompt, tokenized_prompt_mask,
                 token_ar_mask, token_loss_mask,
                 image_masks):
        self.images = images
        self.prompt = prompt
        self.state = state
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.token_ar_mask = token_ar_mask
        self.token_loss_mask = token_loss_mask
        self.image_masks = image_masks



class ObsWrapper:
    def __init__(self, pre):
        self.images = pre.images
        self.image_masks = pre.image_masks
        self.tokenized_prompt = pre.tokenized_prompt
        self.tokenized_prompt_mask = pre.tokenized_prompt_mask
        self.token_ar_mask = pre.token_ar_mask
        self.token_loss_mask = pre.token_loss_mask
        self.state = pre.state



def load_checkpoint(model, ckpt_path):
    print(f"Loading checkpoint from {ckpt_path}")
    sd = load_file(ckpt_path, device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    print("Checkpoint loaded.\n")
    return model



def main():
    image_path = "cup.jpg"
    prompt = "pick up the cup"
    ckpt_path = "checkpoints/model.safetensors"

    device = torch.device("cpu")
    print("Using device:", device)

   
    config = Pi0Config()

    # MUST match training max token length
    tokenizer = PaligemmaTokenizer(config.max_token_len)

    model = PI0Pytorch(config)
    model.eval()

    # Load checkpoint on CPU
    load_checkpoint(model, ckpt_path)

    # Move to GPU
    model.to(device).float()

    raw_img = Image.open(image_path).convert("RGB")
    tensor_img = pil_to_tensor(raw_img).unsqueeze(0).to(device)

    # Image dict for π0 (required keys)
    images = {
        "base_0_rgb": tensor_img,
        "left_wrist_0_rgb": tensor_img,
        "right_wrist_0_rgb": tensor_img,
    }

    
    dummy_state_np = [
        # arm 1 joints (7)
        0.0, -0.5, 1.0, 0.0, -0.3, 0.0, 0.0,
        # arm 2 joints (7)
        0.0, -0.5, 1.0, 0.0, -0.3, 0.0, 0.0,
        # grippers / extra (18)
        *([0.0] * 18)
    ]
    dummy_state_pt = torch.tensor(dummy_state_np, dtype=torch.float32).unsqueeze(0).to(device)


    tok_ids, tok_mask = tokenizer.tokenize(prompt)

    tok_ids = torch.tensor(tok_ids, dtype=torch.long).unsqueeze(0).to(device)
    tok_mask = torch.tensor(tok_mask, dtype=torch.bool).unsqueeze(0).to(device)

    token_ar_mask = torch.ones_like(tok_mask, dtype=torch.bool).to(device)
    token_loss_mask = torch.ones_like(tok_mask, dtype=torch.bool).to(device)

    image_masks = {
        "base_0_rgb": torch.ones((1,), dtype=torch.bool).to(device),
        "left_wrist_0_rgb": torch.ones((1,), dtype=torch.bool).to(device),
        "right_wrist_0_rgb": torch.ones((1,), dtype=torch.bool).to(device),
    }

    obs = Observation(
        images=images,
        prompt=prompt,
        state=dummy_state_pt,
        tokenized_prompt=tok_ids,
        tokenized_prompt_mask=tok_mask,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
        image_masks=image_masks,
    )
    
    preprocessed = preprocess_observation_pytorch(
        obs,
        train=False
    )

    model_obs = ObsWrapper(preprocessed)
    np.random.seed(0)

    noise_np = np.random.randn(
        1,
        config.action_horizon,
        config.action_dim,
    ).astype(np.float32)
    noise_torch = torch.from_numpy(noise_np).to(device)
    print("\nRunning π0 inference...")

    with torch.no_grad():
        actions = model.sample_actions(
            device=device,
            observation=model_obs,
            noise=noise_torch,
            num_steps=30,
        )
    
    print("\n")
    print("shape:",actions.cpu().numpy().shape)
    print(actions.cpu().numpy())
    np.save("pytorch_actions.npy", actions.cpu().numpy())
    print("\nπ0 inference complete")

if __name__ == "__main__":
    main()