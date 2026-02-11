# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Standalone vision model config for Qwen3-Omni-MoE TT vision attention tests.
No HF_MODEL or ModelArgs dependency.
"""
import math
import sys
from pathlib import Path
from typing import Tuple

from loguru import logger

import ttnn

from models.tt_transformers.tt.common import get_out_subblock_w, nearest_multiple
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

# qwen3-omni folder has hyphen; add parent so reference package is importable
_omni_root = Path(__file__).resolve().parents[1]
if str(_omni_root) not in sys.path:
    sys.path.insert(0, str(_omni_root))
from reference.configuration_qwen3_omni_moe import Qwen3OmniMoeVisionEncoderConfig


class _MinimalDecodersOptimizations:
    """Stub for DECODERS_OPTIMIZATIONS: returns concrete dtypes and HiFi4 kernel config."""

    def __init__(self, config):
        self._config = config

    def get_tensor_dtype(self, decoder_id, tensor: TensorGroup):
        # Return concrete dtypes so ttnn.typecast() etc. never receive None
        if tensor == TensorGroup.KV_CACHE:
            return ttnn.bfloat8_b
        if tensor == TensorGroup.WQKV:
            return ttnn.bfloat8_b
        if tensor == TensorGroup.WO:
            return ttnn.bfloat16
        if tensor == TensorGroup.ACTIVATION:
            return ttnn.bfloat16
        return ttnn.bfloat16

    def get_math_fidelity(self, decoder_id, op: OpGroup, configuration):
        return self._config.compute_kernel_config_hifi4


class Qwen3OmniVisionModelArgs:
    """Standalone config for Qwen3-Omni vision attention tests. No HF_MODEL required."""

    MAX_QKV_MM_SEQ_LEN = 2048

    def __init__(self, mesh_device, dummy_weights=True, max_batch_size=1, max_seq_len=2048, **kwargs):
        self.mesh_device = mesh_device
        self.dummy_weights = dummy_weights
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.tile_size = 32

        self.num_devices = mesh_device.get_num_devices() if mesh_device else 1
        self.cluster_shape = list(mesh_device.shape) if mesh_device else [1, 1]
        self.is_galaxy = self.num_devices == 32
        self.is_multichip = self.num_devices > 1
        self.ccl_dtype = ttnn.bfloat8_b

        vision_config = Qwen3OmniMoeVisionEncoderConfig()
        vision_config._attn_implementation = "eager"
        self.hf_config = type("HFConfig", (), {"vision_config": vision_config})()

        self.dim = vision_config.hidden_size
        self.head_dim = vision_config.hidden_size // vision_config.num_heads
        self.n_heads = vision_config.num_heads
        self.n_kv_heads = vision_config.num_heads
        self.padded_head_dim = math.ceil(self.head_dim / self.tile_size) * self.tile_size
        if self.padded_head_dim != self.head_dim:
            logger.info(f"padding head dim from {self.head_dim} to {self.padded_head_dim}")
        self.qkv_size = self.padded_head_dim * (2 * self.n_kv_heads + self.n_heads)
        self.norm_eps = 1e-6

        self.min_kv_prefill_shard_seqlen = (self.tile_size * 8 * 8) / max(1, self.n_kv_heads // self.cluster_shape[1])

        grid = mesh_device.compute_with_storage_grid_size() if mesh_device else None
        self.max_grid_size = ttnn.CoreGrid(x=grid.x, y=grid.y) if grid is not None else None

        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi2_fp16 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_lofi = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.unpadded_hidden_dim = vision_config.intermediate_size
        self.hidden_dim = nearest_multiple(self.unpadded_hidden_dim, self.tile_size * self.num_devices)
        if self.hidden_dim != self.unpadded_hidden_dim:
            logger.info(f"padding hidden dim from {self.unpadded_hidden_dim} to {self.hidden_dim}")
        self.optimizations = type("ModelOptimizations", (), {"bfp4_mlp": False})()

        self.model_config = {}
        self.model_config["DECODERS_OPTIMIZATIONS"] = _MinimalDecodersOptimizations(self)
        self.model_config["SDPA_PROGCFG"] = lambda seqlen, chunk_start_idx=None: ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=256 if seqlen >= 2048 else 64,
            k_chunk_size=256 if seqlen >= 2048 else 64,
        )
        num_rows = lambda seq_len: min(seq_len, 2048)
        k_dim = self.dim
        n_dim = self.dim
        self.model_config["VISION_WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
            m=num_rows(seq_len),
            k=k_dim,
            n=n_dim,
            grid_size=self.find_prefill_grid(num_rows(seq_len), n_dim // self.tile_size),
            in0_block_w=1,
            fuse_batch=seq_len <= 1024,
        )

        assert self.n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"

    def find_largest_divisor(self, n, max_divisor=8):
        for i in range(max_divisor, 0, -1):
            if n % i == 0:
                return i
        return 1

    def find_prefill_grid(self, row_tiles, col_tiles) -> Tuple[int, int]:
        max_rows, max_cols = 8, 8
        cols = next((i for i in range(max_cols, 0, -1) if col_tiles % i == 0), None)
        rows = next((i for i in range(max_rows, 0, -1) if row_tiles % i == 0), None)
        assert cols is not None and rows is not None
        return rows, cols

    def matmul_config(
        self,
        m: int,
        k: int,
        n: int,
        grid_size: Tuple[int, int],
        in0_block_w: int = None,
        fuse_batch: bool = False,
        fused_activation=None,
    ):
        per_core_M = math.ceil(m / (self.tile_size * grid_size[1]))
        per_core_N = math.ceil(n / (self.tile_size * grid_size[0]))
        out_subblock_h = 1
        out_subblock_w = get_out_subblock_w(per_core_N, out_subblock_h)
        if in0_block_w is None:
            in0_block_w = self.find_largest_divisor(k // (self.tile_size * grid_size[1]))
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=fused_activation,
            fuse_batch=fuse_batch,
        )

    def get_model_config(self):
        return self.model_config

    def ccl_topology(self):
        return None

    def get_state_dict_prefix(self, module_name, layer_num=None, **kwargs):
        prefix = f"blocks.{layer_num}." if layer_num is not None else "blocks.0."
        module_map = {"VisionAttention": "attn", "MLP": "mlp"}
        return prefix + module_map.get(module_name, "attn")

    def prepare_residual_tensor_prefill(self, x_bsh, force_replicated=False):
        x_1BSH = x_bsh.unsqueeze(0)
        return ttnn.from_torch(
            x_1BSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
        )

    def reference_attention(self):
        from reference.torch_modeling_qwen3_omni_moe import Qwen3OmniMoeVisionAttention

        return Qwen3OmniMoeVisionAttention(config=self.hf_config.vision_config)

    def reference_mlp(self):
        from reference.torch_modeling_qwen3_omni_moe import Qwen3OmniMoeVisionBlock

        block = Qwen3OmniMoeVisionBlock(config=self.hf_config.vision_config)
        return block.mlp
