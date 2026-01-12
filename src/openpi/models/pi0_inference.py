import os
# Force CPU execution
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# (optional but helpful)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import download

from model import Observation
import model as _model
from flax import nnx
from tokenizer import PaligemmaTokenizer
from pi0_config import Pi0Config
from pi0 import Pi0    
          # same Observation type used by π0

def preprocess_image(image_path: str) -> jnp.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = (np.asarray(img).astype(np.float32) / 255.0) * 2.0 - 1.0 # (H, W, C)
    img = img[None, ...]                               
    return jnp.array(img)

def main():
    image_path = "cup.jpg"
    prompt = "pick up the cup"

    # -------------------------------
    # π0 config + tokenizer
    # -------------------------------
    config = Pi0Config()
    tokenizer = PaligemmaTokenizer(config.max_token_len)
    print("started main")
    # -------------------------------
    # Initialize model + RNG
    # -------------------------------
    rngs = nnx.Rngs(0)
    model = Pi0(config, rngs=rngs)
    # -------------------------------
    # Images (3 cameras required)
    # -------------------------------
    ckpt_dir = download.maybe_download(
        "gs://openpi-assets/checkpoints/pi0_base"
    )           
    
    print("ckpt_dir =", ckpt_dir)
    print("exists?", ckpt_dir.exists())

    params=_model.restore_params(ckpt_dir/"params", dtype=jnp.bfloat16)
    nnx.update(model,params)
    img = preprocess_image(image_path)

    images = {
        "base_0_rgb": img,
        "left_wrist_0_rgb": img,
        "right_wrist_0_rgb": img,
    }

    image_masks = {
        "base_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
        "left_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
        "right_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
    }

    # -------------------------------
    # Dummy robot state (batch=1)
    # -------------------------------
    dummy_state = jnp.array([
        # arm 1 joints (7)
        0.0, -0.5, 1.0, 0.0, -0.3, 0.0, 0.0,
        # arm 2 joints (7)
        0.0, -0.5, 1.0, 0.0, -0.3, 0.0, 0.0,
        # grippers / extra (18)
        *([0.0] * 18)
    ], dtype=jnp.float32)[None, :]   # (1, action_dim)

    # -------------------------------
    # Tokenization (language input)
    # -------------------------------
    tok_ids, tok_mask = tokenizer.tokenize(prompt)

    tok_ids = jnp.array(tok_ids, dtype=jnp.int32)[None, :]     # (1, T)
    tok_mask = jnp.array(tok_mask, dtype=jnp.bool_)[None, :]   # (1, T)

# masks
    token_ar_mask = jnp.ones_like(tok_ids, dtype=jnp.int32)
    token_loss_mask = jnp.ones_like(tok_mask, dtype=jnp.bool_)

    # -------------------------------
    # Build Observation
    # -------------------------------
    obs = Observation(
        images=images,
        state=dummy_state,
        tokenized_prompt=tok_ids,
        tokenized_prompt_mask=tok_mask,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
        image_masks=image_masks,
    )
    # -------------------------------
    # π0 action sampling
    # -------------------------------
    print("\nRunning π0 inference...")
    rng = jax.random.PRNGKey(0)
    rng, sample_rng = jax.random.split(rng)

    np.random.seed(0)

    noise_np = np.random.randn(
        1,
        config.action_horizon,
        config.action_dim,
    ).astype(np.float32)
    print(noise_np.mean())
    print(noise_np.std())
    print(noise_np.flatten()[:10])
    noise_jax = jnp.array(noise_np)
       
    print("before calling inference")
    actions = model.sample_actions(
        sample_rng,
        observation=obs,
        num_steps=30,
        noise=noise_jax
    )

    actions_np = jax.device_get(actions)
    np.save("jax_actions.npy", actions_np)
    print("\nπ0 inference complete")
    print("Action shape:", actions_np.shape)
    print(actions_np)

if __name__=="__main__":
    main()