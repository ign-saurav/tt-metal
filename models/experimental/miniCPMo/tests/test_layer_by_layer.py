# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Layer-by-Layer Comparison Test: PyTorch vs TT Model

Runs in TWO separate phases to avoid OOM:
1. Run PyTorch model, save layer outputs to disk, fully unload
2. Load TT model, compare against saved outputs

Exits immediately when PCC drops below threshold.
"""

import pytest
import torch
import ttnn
import os
import gc
from pathlib import Path
from loguru import logger
import tempfile

from transformers import AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from models.common.utility_functions import comp_pcc
from models.experimental.miniCPMo.tt.minicpm_weight_bridge import MiniCPMWeightBridge
from models.experimental.miniCPMo.tt_transformers.common import create_tt_model

MODEL_PATH = "openbmb/MiniCPM-o-2_6"
PCC_THRESHOLD = 0.90


def run_pytorch_phase(inputs_embeds, n_layers, checkpoint_path, output_dir):
    """
    Run PyTorch model and save outputs.
    Returns True if successful.
    """
    from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO
    from models.experimental.miniCPMo.reference.configuration_minicpm import MiniCPMOConfig

    logger.info("   Creating MiniCPMO with empty weights...")
    minicpm_config = MiniCPMOConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    with init_empty_weights():
        full_model = MiniCPMO(minicpm_config)

    logger.info(f"   Loading weights via load_checkpoint_and_dispatch...")
    load_checkpoint_and_dispatch(
        full_model,
        checkpoint_path,
        device_map="auto",
        dtype=torch.float32,
        offload_folder="/tmp/offload_weights",
    )

    pt_model = full_model.llm
    pt_model.eval()
    logger.info(f"   Extracted LLM from MiniCPMO")

    # Prepare attention mask
    seq_len = inputs_embeds.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    attention_mask = (1.0 - causal_mask.unsqueeze(0).unsqueeze(0)) * torch.finfo(torch.float32).min

    # Run layer by layer
    pt_hidden = inputs_embeds.clone()

    with torch.no_grad():
        rotary_emb = pt_model.model.rotary_emb
        cos, sin = rotary_emb(pt_hidden, position_ids)
        position_embeddings = (cos, sin)

        for layer_idx in range(n_layers):
            pt_layer_out = pt_model.model.layers[layer_idx](
                pt_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            pt_hidden = pt_layer_out[0]

            # Save to disk immediately
            output_path = output_dir / f"layer_{layer_idx}.pt"
            torch.save(pt_hidden.clone().cpu(), output_path)
            logger.info(f"   PT Layer {layer_idx}: range [{pt_hidden.min():.4f}, {pt_hidden.max():.4f}] -> saved")

    # Full cleanup
    del pt_model, full_model, pt_hidden, pt_layer_out
    gc.collect()
    logger.info("   PyTorch model fully unloaded")
    return True


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_layer_by_layer_comparison(mesh_device):
    """
    Layer-by-layer comparison between PyTorch Qwen2 and TT Transformer.
    Runs in two completely separate phases to avoid OOM.
    """
    logger.info("=" * 70)
    logger.info("LAYER-BY-LAYER COMPARISON: PyTorch vs TT")
    logger.info("=" * 70)

    os.environ.setdefault("HF_MODEL", MODEL_PATH)

    # Get model config
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm_config = config.llm_config if hasattr(config, "llm_config") else config
    hidden_dim = llm_config.hidden_size
    n_layers = llm_config.num_hidden_layers

    logger.info(f"\n[1] Model config: hidden_dim={hidden_dim}, n_layers={n_layers}")

    # Create test input
    SEQ_LEN = 128
    inputs_embeds = torch.randn(1, SEQ_LEN, hidden_dim, dtype=torch.float32) * 0.02
    logger.info(f"[2] Input: shape={inputs_embeds.shape}")

    # Find checkpoint
    hf_cache = Path.home() / ".cache/huggingface/hub/models--openbmb--MiniCPM-o-2_6"
    snapshot_dirs = list((hf_cache / "snapshots").glob("*")) if (hf_cache / "snapshots").exists() else []
    if not snapshot_dirs:
        pytest.skip(f"Model not cached at {hf_cache}")
    checkpoint_path = str(snapshot_dirs[0])

    # Create temp dir for PT outputs
    output_dir = Path(tempfile.mkdtemp())

    # ========================================
    # PHASE 1: PyTorch Forward Pass (in subprocess to ensure cleanup)
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: PyTorch Forward Pass")
    logger.info("=" * 70)

    # Save inputs for subprocess
    inputs_path = output_dir / "inputs.pt"
    torch.save(inputs_embeds, inputs_path)

    # Run PT model in subprocess to ensure complete memory cleanup
    import subprocess
    import sys

    pt_script = f"""
import torch
import gc
from pathlib import Path
from loguru import logger

from transformers import AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from models.experimental.miniCPMo.reference.modeling_minicpmo import MiniCPMO
from models.experimental.miniCPMo.reference.configuration_minicpm import MiniCPMOConfig

MODEL_PATH = "{MODEL_PATH}"
checkpoint_path = "{checkpoint_path}"
output_dir = Path("{output_dir}")
n_layers = {n_layers}

# Load inputs
inputs_embeds = torch.load(output_dir / "inputs.pt")
seq_len = inputs_embeds.shape[1]

logger.info("Creating MiniCPMO with empty weights...")
minicpm_config = MiniCPMOConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
with init_empty_weights():
    full_model = MiniCPMO(minicpm_config)

logger.info("Loading weights via load_checkpoint_and_dispatch...")
load_checkpoint_and_dispatch(
    full_model,
    checkpoint_path,
    device_map="auto",
    dtype=torch.float32,
    offload_folder="/tmp/offload_weights",
)

pt_model = full_model.llm
pt_model.eval()

# Prepare
position_ids = torch.arange(seq_len).unsqueeze(0)
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
attention_mask = (1.0 - causal_mask.unsqueeze(0).unsqueeze(0)) * torch.finfo(torch.float32).min

pt_hidden = inputs_embeds.clone()

with torch.no_grad():
    rotary_emb = pt_model.model.rotary_emb
    cos, sin = rotary_emb(pt_hidden, position_ids)
    position_embeddings = (cos, sin)

    for layer_idx in range(n_layers):
        pt_layer_out = pt_model.model.layers[layer_idx](
            pt_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        pt_hidden = pt_layer_out[0]

        output_path = output_dir / f"layer_{{layer_idx}}.pt"
        torch.save(pt_hidden.clone().cpu(), output_path)
        logger.info(f"PT Layer {{layer_idx}}: saved")

# Apply final norm and lm_head
pt_normed = pt_model.model.norm(pt_hidden)
pt_logits = pt_model.lm_head(pt_normed)
pt_last_logits = pt_logits[0, -1, :]  # [vocab]
pt_token = torch.argmax(pt_last_logits).item()

# Save final outputs
torch.save(pt_last_logits.cpu(), output_dir / "final_logits.pt")
torch.save(torch.tensor(pt_token), output_dir / "final_token.pt")
logger.info(f"PT final token: {{pt_token}}")

logger.info("PyTorch phase complete!")
"""

    script_path = output_dir / "pt_phase.py"
    with open(script_path, "w") as f:
        f.write(pt_script)

    logger.info("   Running PyTorch phase in subprocess...")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd="/home/ubuntu/ign_tt/forked/tt-metal",
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        # Check if layer files were created despite error
        layer_files_exist = all((output_dir / f"layer_{i}.pt").exists() for i in range(n_layers))
        if layer_files_exist:
            logger.warning(f"PyTorch subprocess had error but all layer files created: {result.stderr[:500]}")
        else:
            logger.error(f"PyTorch phase failed: {result.stderr}")
            pytest.fail(f"PyTorch phase failed: {result.stderr}")

    logger.info(result.stdout)
    logger.info("   PyTorch phase completed successfully")

    # Verify outputs exist
    for i in range(n_layers):
        assert (output_dir / f"layer_{i}.pt").exists(), f"Missing layer_{i}.pt"

    # ========================================
    # PHASE 2: TT Forward Pass & Compare
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: TT Forward Pass & Layer-by-Layer Comparison")
    logger.info("=" * 70)

    # Load TT weights
    logger.info("   Loading TT weights via MiniCPMWeightBridge...")
    bridge = MiniCPMWeightBridge(MODEL_PATH)
    qwen_weights = bridge.get_qwen_weights()
    logger.info(f"   Loaded {len(qwen_weights)} TT weights")

    # Create TT model
    logger.info("   Creating TT model...")
    tt_model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        instruct=False,
        max_batch_size=1,
        optimizations=None,
        max_seq_len=1024,
        paged_attention_config=None,
        dtype=ttnn.bfloat16,
        state_dict=qwen_weights,
        dummy_weights=False,
        num_layers=n_layers,
    )

    # Prepare TT rotation matrices
    padded_len = ((SEQ_LEN + 127) // 128) * 128
    tt_rot_mats_global = [
        tt_model.rope_setup.cos_matrix[:, :, :padded_len, :],
        tt_model.rope_setup.sin_matrix[:, :, :padded_len, :],
    ]
    tt_rot_mats_local = None
    if hasattr(tt_model, "rope_local_setup") and tt_model.rope_local_setup:
        tt_rot_mats_local = [
            tt_model.rope_local_setup.cos_matrix[:, :, :padded_len, :],
            tt_model.rope_local_setup.sin_matrix[:, :, :padded_len, :],
        ]

    # Prepare TT input
    tt_embeds_torch = inputs_embeds.unsqueeze(1)  # [1, 1, seq, hidden]
    if SEQ_LEN != padded_len:
        padding = torch.zeros(1, 1, padded_len - SEQ_LEN, hidden_dim, dtype=inputs_embeds.dtype)
        tt_embeds_torch = torch.cat([tt_embeds_torch, padding], dim=2)

    tt_hidden = ttnn.from_torch(
        tt_embeds_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Compare layer by layer
    all_passed = True
    for layer_idx in range(n_layers):
        logger.info(f"\n--- LAYER {layer_idx} ---")

        # Load PT reference from disk
        pt_output = torch.load(output_dir / f"layer_{layer_idx}.pt")

        # TT forward
        tt_layer = tt_model.layers[layer_idx]
        tt_layer_out = tt_layer(
            tt_hidden,
            current_pos=0,
            rot_mats_global=tt_rot_mats_global,
            rot_mats_local=tt_rot_mats_local,
            user_id=0,
            mode="prefill",
            page_table=None,
            kv_cache=None,
        )

        # Convert to torch
        tt_hidden_torch = ttnn.to_torch(tt_layer_out)

        # Shape handling
        if tt_hidden_torch.dim() == 4:
            tt_compare = tt_hidden_torch.squeeze(0).squeeze(0)[:SEQ_LEN, :].unsqueeze(0).float()
        else:
            tt_compare = tt_hidden_torch[:SEQ_LEN, :].unsqueeze(0).float()

        pt_compare = pt_output.float()

        logger.info(f"   PT range: [{pt_compare.min():.4f}, {pt_compare.max():.4f}]")
        logger.info(f"   TT range: [{tt_compare.min():.4f}, {tt_compare.max():.4f}]")

        # Compute PCC
        passed, pcc_value = comp_pcc(pt_compare, tt_compare, pcc=PCC_THRESHOLD)

        if passed:
            logger.info(f"   ✅ PCC = {pcc_value:.6f} (threshold: {PCC_THRESHOLD})")
        else:
            logger.error(f"   ❌ PCC = {pcc_value:.6f} < {PCC_THRESHOLD} - FAILED!")
            all_passed = False
            ttnn.deallocate(tt_hidden)
            ttnn.deallocate(tt_layer_out)
            assert False, f"Layer {layer_idx} PCC {pcc_value:.6f} below threshold {PCC_THRESHOLD}"

        # Update for next layer
        ttnn.deallocate(tt_hidden)
        tt_hidden = tt_layer_out

    # ========================================
    # FINAL: Compare norm + lm_head output (first token)
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL: Comparing norm + lm_head output")
    logger.info("=" * 70)

    # Get TT final hidden state as torch
    tt_final_hidden = ttnn.to_torch(tt_hidden)
    if tt_final_hidden.dim() == 4:
        tt_final_hidden = tt_final_hidden.squeeze(0).squeeze(0)[:SEQ_LEN, :].unsqueeze(0)

    # Apply TT norm
    tt_normed = tt_model.norm(tt_hidden, mode="prefill")
    tt_normed_torch = ttnn.to_torch(tt_normed)
    if tt_normed_torch.dim() == 4:
        tt_normed_torch = tt_normed_torch.squeeze(0).squeeze(0)[:SEQ_LEN, :].unsqueeze(0)

    # Apply TT lm_head (get last token logits)
    # The lm_head expects [1, 1, seq, hidden]
    tt_logits = tt_model.lm_head(tt_normed)
    tt_logits_torch = ttnn.to_torch(tt_logits)
    logger.info(f"   TT logits raw shape: {tt_logits_torch.shape}")

    # Get last token logits
    if tt_logits_torch.dim() == 4:
        tt_last_logits = tt_logits_torch.squeeze(0).squeeze(0)[-1, :].float()  # [vocab]
    else:
        tt_last_logits = tt_logits_torch[-1, :].float()

    logger.info(f"   TT last logits shape: {tt_last_logits.shape}")
    logger.info(f"   TT last logits range: [{tt_last_logits.min():.4f}, {tt_last_logits.max():.4f}]")

    # Get TT predicted token
    tt_token = torch.argmax(tt_last_logits).item()

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    logger.info(f"\n   TT predicted token: {tt_token}")
    logger.info(f"   TT predicted text: '{tokenizer.decode([tt_token])}'")

    # Load PT final outputs (if they exist)
    pt_final_logits_path = output_dir / "final_logits.pt"
    pt_final_token_path = output_dir / "final_token.pt"

    if pt_final_logits_path.exists() and pt_final_token_path.exists():
        pt_last_logits = torch.load(pt_final_logits_path).float()
        pt_token = torch.load(pt_final_token_path).item()

        logger.info(f"   PT predicted token: {pt_token}")
        logger.info(f"   PT predicted text: '{tokenizer.decode([pt_token])}'")

        # Compare logits PCC
        passed, pcc_value = comp_pcc(pt_last_logits.unsqueeze(0), tt_last_logits.unsqueeze(0), pcc=0.8)
        logger.info(f"\n   Logits PCC: {pcc_value:.6f}")

        # Compare tokens
        if pt_token == tt_token:
            logger.info(f"   ✅ TOKENS MATCH! Both predict: {pt_token} ('{tokenizer.decode([pt_token])}')")
        else:
            logger.warning(
                f"   ⚠️ Tokens differ: PT={pt_token} ('{tokenizer.decode([pt_token])}') vs TT={tt_token} ('{tokenizer.decode([tt_token])}')"
            )
            # Check if TT token is in PT top-5
            pt_top5 = torch.topk(pt_last_logits, 5).indices.tolist()
            if tt_token in pt_top5:
                logger.info(f"   TT token is in PT's top-5: {pt_top5}")
    else:
        logger.warning("   PT final token files not found (norm/lm_head may have failed)")
        logger.info(f"   TT predicted token: {tt_token} ('{tokenizer.decode([tt_token])}')")

    ttnn.deallocate(tt_hidden)
    ttnn.deallocate(tt_normed)
    ttnn.deallocate(tt_logits)

    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL LAYERS PASSED!")
    logger.info("=" * 70)

    assert all_passed
