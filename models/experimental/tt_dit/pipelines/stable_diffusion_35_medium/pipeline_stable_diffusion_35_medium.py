# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List
from PIL import Image

import torch
import tqdm
import ttnn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from safetensors.torch import load_file as safetensors_load_file
from huggingface_hub import hf_hub_download
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from contextlib import contextmanager, nullcontext

from ...encoders.clip.model_clip import CLIPEncoder, CLIPConfig
from ...encoders.t5.model_t5 import T5Encoder, T5Config

# NOTE: SD35Transformer is the new tt-dit implementation
from ...models.transformers.sd35_med.transformer_sd35_medium import SD35MediumMMDiTX
from ...models.vae.vae_sd35 import VAEDecoder
from ...parallel.manager import CCLManager
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig, ParallelFactor
from ...utils.padding import PaddingConfig
from ...utils.cache import save_cache_dict, load_cache_dict, cache_dict_exists, get_and_create_cache_path

TILE_SIZE = 32


@dataclass
class TimingData:
    clip_encoding_time: float = 0.0
    t5_encoding_time: float = 0.0
    total_encoding_time: float = 0.0
    denoising_step_times: List[float] = field(default_factory=list)
    vae_decoding_time: float = 0.0
    total_time: float = 0.0


class TimingCollector:
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.step_timings: Dict[str, List[float]] = {}

    @contextmanager
    def time_section(self, name: str):
        start = time.time()
        yield
        end = time.time()
        self.timings[name] = end - start

    @contextmanager
    def time_step(self, name: str):
        start = time.time()
        yield
        end = time.time()
        if name not in self.step_timings:
            self.step_timings[name] = []
        self.step_timings[name].append(end - start)

    def get_timing_data(self) -> TimingData:
        return TimingData(
            clip_encoding_time=self.timings.get("clip_encoding", 0.0),
            t5_encoding_time=self.timings.get("t5_encoding", 0.0),
            total_encoding_time=self.timings.get("total_encoding", 0.0),
            denoising_step_times=self.step_timings.get("denoising_step", []),
            vae_decoding_time=self.timings.get("vae_decoding", 0.0),
            total_time=self.timings.get("total", 0.0),
        )

    def reset(self):
        self.timings = {}
        self.step_timings = {}
        return self


@dataclass
class PipelineTrace:
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_projection_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    latents_output: ttnn.Tensor
    tid: int


class StableDiffusion3Pipeline:
    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        enable_t5_text_encoder: bool = True,
        guidance_cond: int,
        parallel_config: DiTParallelConfig,
        num_links: int,
        height: int,
        width: int,
        model_checkpoint_path: str,
        use_cache=False,
    ) -> None:
        self._mesh_device = mesh_device

        self.dit_parallel_config = parallel_config

        # Create submeshes
        submesh_shape = list(mesh_device.shape)
        if parallel_config.cfg_parallel.mesh_axis is not None:
            submesh_shape[parallel_config.cfg_parallel.mesh_axis] //= parallel_config.cfg_parallel.factor
        logger.info(f"Parallel config: {parallel_config}")
        logger.info(f"Original mesh shape: {mesh_device.shape}")
        logger.info(f"Creating submeshes with shape {submesh_shape}")
        self.submesh_devices = self._mesh_device.create_submeshes(ttnn.MeshShape(*submesh_shape))

        self.ccl_managers = [
            CCLManager(submesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
            for submesh_device in self.submesh_devices
        ]
        # Hacky submesh reshapes and assignment to parallelize encoders and VAE
        encoder_device = self.submesh_devices[0]
        self.original_submesh_shape = tuple(encoder_device.shape)
        self.desired_encoder_submesh_shape = tuple(encoder_device.shape)

        # Check if this is n150 (1, 1) - special handling needed
        is_n150 = tuple(encoder_device.shape) == (1, 1)

        if is_n150:
            # For n150, we can't reshape to (1, 4), so use (1, 1) with factor=1
            vae_submesh_idx = 0
            if enable_t5_text_encoder:
                logger.warning("n150 device detected: T5 enabled on single device - may have memory constraints")
            logger.info("n150 device detected: using (1, 1) submesh with no reshaping")
        elif encoder_device.shape[1] != 4:
            # If reshaping, vae_device must be on submesh 0. That means T5 can't fit, so disable it.
            vae_submesh_idx = 0
            if enable_t5_text_encoder:
                logger.warning(
                    "If VAE submesh must be reshaped, VAE must be on submesh 0, and T5 cannot fit. Disabling T5."
                )
                enable_t5_text_encoder = False

            cfg_shape = tuple(encoder_device.shape)
            assert cfg_shape[0] * cfg_shape[1] == 4, f"Cannot reshape {cfg_shape} to a 1x4 mesh"
            logger.info(f"Reshaping submesh device 0 from {cfg_shape} to (1, 4) for CLIP")
            self.desired_encoder_submesh_shape = (1, 4)

        else:
            # vae_device can only be on submesh 1 if submesh is not getting reshaped.
            vae_submesh_idx = 1
        vae_device = self.submesh_devices[vae_submesh_idx]

        # Create encoder parallel config
        # For n150, use factor=1; otherwise use factor=4 for 1x4 submesh
        encoder_parallel_factor = 1 if is_n150 else encoder_device.shape[1]
        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=encoder_parallel_factor, mesh_axis=1)
        )

        self.encoder_parallel_config = encoder_parallel_config
        self.encoder_device = encoder_device

        # VAE parallel config - same logic as encoder
        vae_parallel_factor = 1 if is_n150 else vae_device.shape[1]
        vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=vae_parallel_factor, mesh_axis=1))
        self.vae_parallel_config = vae_parallel_config
        self.vae_device = vae_device
        self.vae_submesh_idx = vae_submesh_idx

        logger.info("loading models...")
        self._tokenizer_1 = CLIPTokenizer.from_pretrained(model_checkpoint_path, subfolder="tokenizer")
        self._tokenizer_2 = CLIPTokenizer.from_pretrained(model_checkpoint_path, subfolder="tokenizer_2")
        self._tokenizer_3 = T5TokenizerFast.from_pretrained(model_checkpoint_path, subfolder="tokenizer_3")
        self._text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
            model_checkpoint_path, subfolder="text_encoder"
        )
        self._text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_checkpoint_path, subfolder="text_encoder_2"
        )
        if enable_t5_text_encoder:
            torch_text_encoder_3 = T5EncoderModel.from_pretrained(model_checkpoint_path, subfolder="text_encoder_3")
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_checkpoint_path, subfolder="scheduler")
        self._torch_vae = AutoencoderKL.from_pretrained(model_checkpoint_path, subfolder="vae")

        # Load reference transformer from test file instead of Hugging Face
        from ...tests.models.sd35_med.test_mmdit_sd35_medium import ReferenceMMDiTX

        # Load MMDiT weights from safetensors file (same as test)
        safetensors_filename = "sd3.5_medium.safetensors"
        logger.info(f"Loading MMDiT weights from safetensors file: {model_checkpoint_path}/{safetensors_filename}")

        safetensors_path = hf_hub_download(
            repo_id=model_checkpoint_path,
            filename=safetensors_filename,
            repo_type="model",
        )
        logger.info(f"Downloaded safetensors file to: {safetensors_path}")

        # Load the safetensors file
        all_weights = safetensors_load_file(safetensors_path)
        logger.info(f"Loaded {len(all_weights)} total keys from safetensors file")

        # Filter only MMDiT-related weights
        mmdit_weights = {}
        mmdit_prefix = "model.diffusion_model."
        mmdit_component_prefixes = [
            "model.diffusion_model.x_embedder",
            "model.diffusion_model.pos_embed",
            "model.diffusion_model.t_embedder",
            "model.diffusion_model.y_embedder",
            "model.diffusion_model.context_embedder",
            "model.diffusion_model.joint_blocks",
            "model.diffusion_model.final_layer",
        ]

        logger.info(f"Filtering MMDiT weights from safetensors (looking for '{mmdit_prefix}' prefix)...")
        for key, value in all_weights.items():
            if any(key.startswith(prefix) for prefix in mmdit_component_prefixes):
                if key.startswith(mmdit_prefix):
                    mmdit_key = key[len(mmdit_prefix) :]
                else:
                    mmdit_key = key
                mmdit_weights[mmdit_key] = value

        logger.info(f"Extracted {len(mmdit_weights)} MMDiT-related keys from safetensors")

        # Detect actual depth and pos_embed_max_size from weights
        actual_depth = 24  # Default for SD3.5 Medium
        joint_block_indices = set()
        for key in mmdit_weights.keys():
            if key.startswith("joint_blocks."):
                parts = key.split(".")
                if len(parts) >= 2:
                    try:
                        block_idx = int(parts[1])
                        joint_block_indices.add(block_idx)
                    except ValueError:
                        pass

        if joint_block_indices:
            actual_depth = max(joint_block_indices) + 1
            logger.info(f"Detected {actual_depth} joint blocks in safetensors")
        else:
            logger.warning(f"Could not detect joint_blocks from safetensors, using default depth={actual_depth}")

        pos_embed_max_size_from_weights = None
        if "pos_embed" in mmdit_weights:
            pos_embed_shape = mmdit_weights["pos_embed"].shape
            if len(pos_embed_shape) == 3:
                num_patches_in_weights = pos_embed_shape[1]
                pos_embed_max_size_from_weights = int(num_patches_in_weights**0.5)
                logger.info(f"Detected pos_embed_max_size={pos_embed_max_size_from_weights} from weights")

        actual_pos_embed_max_size = pos_embed_max_size_from_weights if pos_embed_max_size_from_weights else 384
        actual_num_patches = (
            actual_pos_embed_max_size * actual_pos_embed_max_size if pos_embed_max_size_from_weights else None
        )

        # Create transformer config object with detected values
        class TransformerConfig:
            def __init__(self):
                self.sample_size = 128
                self.patch_size = 2
                self.in_channels = 16
                self.num_layers = actual_depth
                self.attention_head_dim = 64
                self.num_attention_heads = 24
                self.joint_attention_dim = 4096
                self.out_channels = 16
                self.pos_embed_max_size = actual_pos_embed_max_size

        transformer_config = TransformerConfig()

        # Map safetensors keys to reference model format
        def map_hf_to_ttnn_keys(hf_state_dict):
            """Map safetensors/Hugging Face state dict keys to reference model keys"""
            ttnn_state_dict = {}
            for key, value in hf_state_dict.items():
                if key.startswith("pos_embed."):
                    new_key = key.replace("pos_embed.", "x_embedder.")
                    ttnn_state_dict[new_key] = value
                elif key == "pos_embed":
                    ttnn_state_dict[key] = value
                elif key.startswith("time_text_embed.timestep_embed."):
                    new_key = key.replace("time_text_embed.timestep_embed.", "t_embedder.")
                    if "linear_1" in new_key:
                        new_key = new_key.replace("linear_1", "mlp.0")
                    elif "linear_2" in new_key:
                        new_key = new_key.replace("linear_2", "mlp.2")
                    ttnn_state_dict[new_key] = value
                elif key.startswith("time_text_embed.text_embed."):
                    new_key = key.replace("time_text_embed.text_embed.", "y_embedder.")
                    if "linear_1" in new_key:
                        new_key = new_key.replace("linear_1", "mlp.0")
                    elif "linear_2" in new_key:
                        new_key = new_key.replace("linear_2", "mlp.2")
                    ttnn_state_dict[new_key] = value
                elif key.startswith("transformer_blocks."):
                    new_key = key.replace("transformer_blocks.", "joint_blocks.")
                    ttnn_state_dict[new_key] = value
                elif key.startswith("norm_out."):
                    new_key = key.replace("norm_out.", "final_layer.norm_final.")
                    ttnn_state_dict[new_key] = value
                elif key.startswith("proj_out."):
                    new_key = key.replace("proj_out.", "final_layer.linear.")
                    ttnn_state_dict[new_key] = value
                else:
                    ttnn_state_dict[key] = value
            return ttnn_state_dict

        logger.info("Mapping safetensors keys to reference model format...")
        mapped_weights = map_hf_to_ttnn_keys(mmdit_weights)

        # Create reference model
        sample_size = 128  # Default, will be updated from config if available
        input_size = sample_size // 2  # Assuming patch_size=2

        # Create reference transformer model
        # Pass actual_num_patches so pos_embed is registered as a buffer
        ref_transformer = ReferenceMMDiTX(
            input_size=input_size,
            patch_size=2,
            in_channels=16,
            depth=actual_depth,
            mlp_ratio=4.0,
            learn_sigma=False,
            out_channels=16,
            pos_embed_max_size=actual_pos_embed_max_size,
            num_patches=actual_num_patches,  # Pass actual_num_patches so pos_embed buffer is created
        )
        ref_transformer.eval()

        # Load weights into reference model
        logger.info("Loading weights into reference model...")
        # Check if pos_embed is in mapped_weights
        if "pos_embed" in mapped_weights:
            logger.info(f"Found pos_embed in mapped weights with shape: {mapped_weights['pos_embed'].shape}")
        else:
            logger.warning("pos_embed not found in mapped weights")

        ref_transformer.load_state_dict(mapped_weights, strict=False)

        # Ensure pos_embed is loaded if it exists in mapped_weights
        if "pos_embed" in mapped_weights and ref_transformer.pos_embed is not None:
            # pos_embed is a buffer, so we need to load it explicitly
            ref_transformer.pos_embed.data.copy_(mapped_weights["pos_embed"])
            logger.info("Loaded pos_embed into reference model buffer")
        elif "pos_embed" in mapped_weights:
            logger.warning("pos_embed found in weights but reference model pos_embed is None (num_patches was None)")

        # Get state dict from reference model for TTNN transformer
        # Include buffers in state_dict
        torch_transformer_state_dict = ref_transformer.state_dict()

        # Verify pos_embed is in the state dict
        if "pos_embed" in torch_transformer_state_dict:
            logger.info(
                f"pos_embed found in reference model state_dict with shape: {torch_transformer_state_dict['pos_embed'].shape}"
            )
        else:
            logger.warning("pos_embed not found in reference model state_dict")

        assert isinstance(self._tokenizer_1, CLIPTokenizer)
        assert isinstance(self._tokenizer_2, CLIPTokenizer)
        assert isinstance(self._tokenizer_3, T5TokenizerFast)
        assert isinstance(self._text_encoder_1, CLIPTextModelWithProjection)
        assert isinstance(self._text_encoder_2, CLIPTextModelWithProjection)
        assert isinstance(self._scheduler, FlowMatchEulerDiscreteScheduler)
        assert isinstance(self._torch_vae, AutoencoderKL)

        logger.info("creating TT-NN transformer...")

        # Detect model variant from checkpoint path
        model_checkpoint_str = str(model_checkpoint_path)
        is_medium = "medium" in model_checkpoint_str.lower()
        is_large = "large" in model_checkpoint_str.lower()
        assert is_medium or is_large, f"Model checkpoint path must contain 'medium' or 'large': {model_checkpoint_path}"

        # Determine model name for cache
        if is_medium:
            model_name = "stable-diffusion-3.5-medium"
        else:
            model_name = "stable-diffusion-3.5-large"

        if transformer_config.num_attention_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                transformer_config.num_attention_heads,
                transformer_config.attention_head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        # Get embedding dimensions from config
        # For medium: embedding_dim=1536, pooled_projection_dim=1152, caption_projection_dim=1536
        # For large: embedding_dim=2432, pooled_projection_dim=2048, caption_projection_dim=2432
        embedding_dim = transformer_config.num_attention_heads * transformer_config.attention_head_dim

        # Get pooled_projection_dim from text_proj config if available, otherwise infer
        # For SD3.5, this is typically in the text_proj config
        if hasattr(transformer_config, "pooled_projection_dim"):
            pooled_projection_dim = transformer_config.pooled_projection_dim
        elif is_medium:
            pooled_projection_dim = 1152  # SD3.5 Medium default
        else:
            pooled_projection_dim = 2048  # SD3.5 Large default

        # Get caption_projection_dim (same as embedding_dim for SD3.5)
        caption_projection_dim = embedding_dim

        self.transformers = []
        for i, submesh_device in enumerate(self.submesh_devices):
            # Calculate input_size from sample_size (assuming square images)
            sample_size = transformer_config.sample_size if hasattr(transformer_config, "sample_size") else 128
            input_size = (
                sample_size // transformer_config.patch_size
                if hasattr(transformer_config, "patch_size")
                else sample_size // 2
            )

            tt_transformer = SD35MediumMMDiTX(
                input_size=input_size,
                patch_size=transformer_config.patch_size if hasattr(transformer_config, "patch_size") else 2,
                in_channels=transformer_config.in_channels,
                depth=transformer_config.num_layers,
                out_channels=transformer_config.out_channels
                if hasattr(transformer_config, "out_channels")
                else transformer_config.in_channels,
                pos_embed_max_size=transformer_config.pos_embed_max_size
                if hasattr(transformer_config, "pos_embed_max_size")
                else 192,
                mesh_device=submesh_device,
                ccl_manager=self.ccl_managers[i],
                parallel_config=self.dit_parallel_config,
            )

            if use_cache:
                cache_path = get_and_create_cache_path(
                    model_name=model_name,
                    subfolder="transformer",
                    parallel_config=self.dit_parallel_config,
                    mesh_shape=tuple(submesh_device.shape),
                    dtype="bf16",
                )
                # create cache if it doesn't exist or is invalid
                cache_exists = cache_dict_exists(cache_path)
                cache_valid = True
                if cache_exists:
                    cache_dict = load_cache_dict(cache_path)
                    cache_valid = SD35MediumMMDiTX.is_cache_valid(cache_dict)
                    if not cache_valid:
                        logger.warning(
                            f"Cache exists but is invalid (missing required keys, likely created before joint block caching was added). "
                            f"Recreating cache: {cache_path}"
                        )

                if not cache_exists or not cache_valid:
                    logger.info(f"Creating cache: {cache_path} and loading transformer weights from PyTorch state dict")
                    tt_transformer.load_state_dict(torch_transformer_state_dict)
                    save_cache_dict(tt_transformer.to_cached_state_dict(cache_path), cache_path)
                else:
                    logger.info(f"Loading transformer weights from cache: {cache_path}")
                    cache_dict = load_cache_dict(cache_path)
                    tt_transformer.from_cached_state_dict(cache_dict)
            else:
                logger.info("Loading transformer weights from reference model state dict")
                tt_transformer.load_state_dict(torch_transformer_state_dict)

            self.transformers.append(tt_transformer)
            ttnn.synchronize_device(submesh_device)

        self._num_channels_latents = transformer_config.in_channels
        self._joint_attention_dim = transformer_config.joint_attention_dim
        self.patch_size = transformer_config.patch_size if hasattr(transformer_config, "patch_size") else 2

        self._block_out_channels = self._torch_vae.config.block_out_channels
        self._torch_vae_scaling_factor = self._torch_vae.config.scaling_factor
        self._torch_vae_shift_factor = self._torch_vae.config.shift_factor

        self._torch_vae_scale_factor = 2 ** (len(self._block_out_channels) - 1)
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._torch_vae_scale_factor)

        if self.desired_encoder_submesh_shape != self.original_submesh_shape:
            # HACK: reshape submesh device 0 to 1D
            self.encoder_device.reshape(ttnn.MeshShape(*self.desired_encoder_submesh_shape))

        logger.info("creating TT-NN CLIP text encoder...")

        # Create CLIP config for encoder 1
        clip_config_1 = CLIPConfig(
            vocab_size=self._text_encoder_1.config.vocab_size,
            embed_dim=self._text_encoder_1.config.hidden_size,
            ff_dim=self._text_encoder_1.config.intermediate_size,
            num_heads=self._text_encoder_1.config.num_attention_heads,
            num_hidden_layers=self._text_encoder_1.config.num_hidden_layers,
            max_prompt_length=77,
            layer_norm_eps=self._text_encoder_1.config.layer_norm_eps,
            attention_dropout=self._text_encoder_1.config.attention_dropout,
            hidden_act=self._text_encoder_1.config.hidden_act,
        )

        # Create CLIP config for encoder 2
        clip_config_2 = CLIPConfig(
            vocab_size=self._text_encoder_2.config.vocab_size,
            embed_dim=self._text_encoder_2.config.hidden_size,
            ff_dim=self._text_encoder_2.config.intermediate_size,
            num_heads=self._text_encoder_2.config.num_attention_heads,
            num_hidden_layers=self._text_encoder_2.config.num_hidden_layers,
            max_prompt_length=77,
            layer_norm_eps=self._text_encoder_2.config.layer_norm_eps,
            attention_dropout=self._text_encoder_2.config.attention_dropout,
            hidden_act=self._text_encoder_2.config.hidden_act,
        )

        # Store original state dicts before creating new encoders
        text_encoder_1_state_dict = self._text_encoder_1.state_dict()
        text_encoder_2_state_dict = self._text_encoder_2.state_dict()

        # Create new CLIP encoders
        self._text_encoder_1 = CLIPEncoder(
            config=clip_config_1,
            mesh_device=encoder_device,
            ccl_manager=self.ccl_managers[0],  # use CCL manager for submesh 0
            parallel_config=encoder_parallel_config,
            eos_token_id=2,  # default EOS token ID for CLIP
        )

        self._text_encoder_2 = CLIPEncoder(
            config=clip_config_2,
            mesh_device=encoder_device,
            ccl_manager=self.ccl_managers[0],  # Use CCL manager for submesh 0
            parallel_config=encoder_parallel_config,
            eos_token_id=2,  # default EOS token ID for CLIP
        )

        # Load state dicts into new encoders
        self._text_encoder_1.load_torch_state_dict(text_encoder_1_state_dict)
        self._text_encoder_2.load_torch_state_dict(text_encoder_2_state_dict)

        if enable_t5_text_encoder:
            logger.info("creating TT-NN T5 text encoder...")

            # Create T5 config
            t5_config = T5Config(
                vocab_size=torch_text_encoder_3.config.vocab_size,
                embed_dim=torch_text_encoder_3.config.d_model,
                ff_dim=torch_text_encoder_3.config.d_ff,
                kv_dim=torch_text_encoder_3.config.d_kv,
                num_heads=torch_text_encoder_3.config.num_heads,
                num_hidden_layers=torch_text_encoder_3.config.num_layers,
                max_prompt_length=256,  # default T5 max prompt length
                layer_norm_eps=torch_text_encoder_3.config.layer_norm_epsilon,
                relative_attention_num_buckets=torch_text_encoder_3.config.relative_attention_num_buckets,
                relative_attention_max_distance=torch_text_encoder_3.config.relative_attention_max_distance,
            )

            # Store original state dict before creating new encoder
            torch_text_encoder_3_state_dict = torch_text_encoder_3.state_dict()

            # Create new T5 encoder
            self._text_encoder_3 = T5Encoder(
                config=t5_config,
                mesh_device=encoder_device,
                ccl_manager=self.ccl_managers[0],  # use CCL manager for submesh 0
                parallel_config=encoder_parallel_config,
            )

            # Load state dict into new encoder
            self._text_encoder_3.load_torch_state_dict(torch_text_encoder_3_state_dict)
        else:
            self._text_encoder_3 = None

        self.timing_collector = None  # Set externally when timing is needed

        self._trace = None

        # intermediate buffers for safe tracing
        self._intermediate_noise_list = []
        self._sigma_difference_list = []
        self._vae_input_latents = None

        ttnn.synchronize_device(self.encoder_device)

        self._vae_decoder = VAEDecoder.from_torch(
            torch_ref=self._torch_vae.decoder,
            mesh_device=self.vae_device,
            parallel_config=self.vae_parallel_config,
            ccl_manager=self.ccl_managers[vae_submesh_idx],
        )

        if self.desired_encoder_submesh_shape != self.original_submesh_shape:
            # HACK: reshape submesh device 0 to 1D
            # If reshaping, vae device is same as encoder device
            self.encoder_device.reshape(ttnn.MeshShape(*self.original_submesh_shape))

    def prepare(
        self,
        *,
        batch_size: int,
        num_images_per_prompt: int = 1,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 4.5,
        max_t5_sequence_length: int = 256,
        prompt_sequence_length: int = 333,
        spatial_sequence_length: int = 4096,
    ) -> None:
        self._prepared_batch_size = batch_size
        self._prepared_num_images_per_prompt = num_images_per_prompt
        self._prepared_width = width
        self._prepared_height = height
        self._prepared_guidance_scale = guidance_scale
        self._prepared_max_t5_sequence_length = max_t5_sequence_length
        self._prepared_prompt_sequence_length = prompt_sequence_length

    @staticmethod
    def create_pipeline(
        mesh_device,
        batch_size=1,
        image_w=1024,
        image_h=1024,
        guidance_scale=4.5,
        num_images_per_prompt=1,
        max_t5_sequence_length=256,
        prompt_sequence_length=333,
        spatial_sequence_length=4096,
        cfg_config=None,
        sp_config=None,
        tp_config=None,
        num_links=None,
        model_checkpoint_path=f"stabilityai/stable-diffusion-3.5-medium",
        use_cache=False,
        enable_t5_text_encoder=None,  # None = auto-detect, True/False = explicit
    ):
        # default config per mesh shape
        default_config = {
            (1, 1): {"cfg_config": (1, None), "sp_config": (1, None), "tp_config": (1, None), "num_links": 1},
        }

        # get config from user or default if not provided
        mesh_shape = tuple(mesh_device.shape)
        if mesh_shape not in default_config and (cfg_config is None or sp_config is None or tp_config is None):
            raise ValueError(
                f"Mesh shape {mesh_shape} not in default_config. Please provide cfg_config, sp_config, and tp_config explicitly."
            )

        cfg_factor, cfg_axis = cfg_config or default_config[mesh_shape]["cfg_config"]
        sp_factor, sp_axis = sp_config or default_config[mesh_shape]["sp_config"]
        tp_factor, tp_axis = tp_config or default_config[mesh_shape]["tp_config"]
        num_links = num_links or default_config[mesh_shape]["num_links"]

        parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=cfg_factor, mesh_axis=cfg_axis),
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
        )

        guidance_cond = 2 if (guidance_scale > 1 and cfg_factor == 1) else 1

        # Enable T5 based on device configuration
        # If enable_t5_text_encoder is explicitly set, use that value
        # Otherwise, auto-detect based on device configuration
        mesh_shape = tuple(mesh_device.shape)
        if enable_t5_text_encoder is None:
            # Auto-detect: T5 is disabled if mesh needs reshaping for CLIP encoder
            if mesh_shape == (1, 1):
                # n150: T5 disabled by default due to memory constraints, but can be enabled explicitly
                enable_t5_text_encoder = False
                submesh_shape = [1, 1]  # For logging
            else:
                submesh_shape = list(mesh_device.shape)
                submesh_shape[cfg_axis] //= cfg_factor
                enable_t5_text_encoder = submesh_shape[1] == 4  # T5 only works if submesh doesn't need reshaping
        else:
            # User explicitly set enable_t5_text_encoder
            if mesh_shape == (1, 1):
                submesh_shape = [1, 1]
                if enable_t5_text_encoder:
                    logger.warning("T5 explicitly enabled on single device (n150) - may have memory constraints")
            else:
                submesh_shape = list(mesh_device.shape)
                submesh_shape[cfg_axis] //= cfg_factor
                if enable_t5_text_encoder and submesh_shape[1] != 4:
                    logger.warning(
                        f"T5 enabled but submesh shape {submesh_shape} may require reshaping - T5 may not fit"
                    )

        logger.info(f"Mesh device shape: {mesh_device.shape}")
        logger.info(f"Submesh shape: {submesh_shape}")
        logger.info(f"Parallel config: {parallel_config}")
        logger.info(f"T5 enabled: {enable_t5_text_encoder}")

        # Create pipeline
        pipeline = StableDiffusion3Pipeline(
            mesh_device=mesh_device,
            enable_t5_text_encoder=enable_t5_text_encoder,
            guidance_cond=guidance_cond,
            parallel_config=parallel_config,
            num_links=num_links,
            height=image_h,
            width=image_w,
            model_checkpoint_path=model_checkpoint_path,
            use_cache=use_cache,
        )

        pipeline.prepare(
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            width=image_w,
            height=image_h,
            guidance_scale=guidance_scale,
            max_t5_sequence_length=max_t5_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
            spatial_sequence_length=spatial_sequence_length,
        )

        return pipeline

    def run_single_prompt(self, prompt, negative_prompt="", num_inference_steps=40, seed=None):
        return self.__call__(
            prompt_1=[prompt],
            prompt_2=[prompt],
            prompt_3=[prompt],
            negative_prompt_1=[negative_prompt or ""],
            negative_prompt_2=[negative_prompt or ""],
            negative_prompt_3=[negative_prompt or ""],
            num_inference_steps=num_inference_steps,
            seed=seed,
            traced=True,
        )

    def __call__(
        self,
        *,
        prompt_1: list[str],
        prompt_2: list[str],
        prompt_3: list[str],
        negative_prompt_1: list[str],
        negative_prompt_2: list[str],
        negative_prompt_3: list[str],
        num_inference_steps: int = 40,
        seed: int | None = None,
        traced: bool = False,
        clip_skip: int | None = None,
    ) -> List[Image.Image]:
        timer = self.timing_collector.reset() if self.timing_collector else None

        with timer.time_section("total") if timer else nullcontext():
            start_time = time.time()

            batch_size = self._prepared_batch_size
            num_images_per_prompt = self._prepared_num_images_per_prompt
            width = self._prepared_width
            height = self._prepared_height
            guidance_scale = self._prepared_guidance_scale
            max_t5_sequence_length = self._prepared_max_t5_sequence_length

            assert height % (self._torch_vae_scale_factor * self.patch_size) == 0
            assert width % (self._torch_vae_scale_factor * self.patch_size) == 0
            assert max_t5_sequence_length <= 512  # noqa: PLR2004
            assert batch_size == len(prompt_1)

            do_classifier_free_guidance = guidance_scale > 1
            # TODO: pass the patch_size value
            patch_size = 2
            # Latents shape: (B, H, W, C) in NHWC format for VAE compatibility
            # But new transformer expects (B, C, H, W) in NCHW format
            latents_shape = (
                batch_size * num_images_per_prompt,
                height // self._torch_vae_scale_factor,
                width // self._torch_vae_scale_factor,
                self._num_channels_latents,
            )

            print(f"Latents shape: {latents_shape}")

            logger.info("encoding prompts...")

            with timer.time_section("total_encoding") if timer else nullcontext():
                if self.desired_encoder_submesh_shape != self.original_submesh_shape:
                    # HACK: reshape submesh device 0 from 2D to 1D
                    self.encoder_device.reshape(ttnn.MeshShape(*self.desired_encoder_submesh_shape))
                prompt_encoding_start_time = time.time()
                prompt_embeds, pooled_prompt_embeds = self._encode_prompts(
                    prompt_1=prompt_1,
                    prompt_2=prompt_2,
                    prompt_3=prompt_3,
                    negative_prompt_1=negative_prompt_1,
                    negative_prompt_2=negative_prompt_2,
                    negative_prompt_3=negative_prompt_3,
                    num_images_per_prompt=num_images_per_prompt,
                    max_t5_sequence_length=max_t5_sequence_length,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    clip_skip=clip_skip,
                )
                if self.desired_encoder_submesh_shape != self.original_submesh_shape:
                    # HACK: reshape submesh device 0 from 1D to 2D
                    self.encoder_device.reshape(ttnn.MeshShape(*self.original_submesh_shape))
                prompt_encoding_end_time = time.time()
                logger.info("preparing timesteps...")

            self._scheduler.set_timesteps(num_inference_steps)
            timesteps = self._scheduler.timesteps

            logger.info("preparing latents...")

            if seed is not None:
                torch.manual_seed(seed)
            latents = torch.randn(latents_shape, dtype=prompt_embeds.dtype)  # Shape: [B, C, H, W] in NCHW format
            # Note: New transformer handles patching internally, so we don't call patchify here

            tt_prompt_embeds_list = []
            tt_pooled_prompt_embeds_list = []
            tt_latents_step_list = []
            for i, submesh_device in enumerate(self.submesh_devices):
                # For n150 (1,1) or when dims are [None, None], use replication
                is_n150 = tuple(submesh_device.shape) == (1, 1)
                if is_n150:
                    prompt_mesh_mapper = ttnn.ReplicateTensorToMesh(submesh_device)
                else:
                    prompt_mesh_mapper = ttnn.ShardTensor2dMesh(
                        submesh_device,
                        tuple(submesh_device.shape),
                        dims=[None, None],
                    )

                tt_prompt_embeds = ttnn.from_torch(
                    prompt_embeds[i].unsqueeze(0).unsqueeze(0)
                    if self.dit_parallel_config.cfg_parallel.factor == 2
                    else prompt_embeds,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=prompt_mesh_mapper,
                )

                tt_pooled_prompt_embeds = ttnn.from_torch(
                    pooled_prompt_embeds[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    if self.dit_parallel_config.cfg_parallel.factor == 2
                    else pooled_prompt_embeds,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=prompt_mesh_mapper,
                )

                # For latents: check if sequence_parallel factor is 1 (n150 case)
                if is_n150 or self.dit_parallel_config.sequence_parallel.factor == 1:
                    latents_mesh_mapper = ttnn.ReplicateTensorToMesh(submesh_device)
                else:
                    shard_latents_dims = [None, None]
                    shard_latents_dims[self.dit_parallel_config.sequence_parallel.mesh_axis] = 2
                    latents_mesh_mapper = ttnn.ShardTensor2dMesh(
                        submesh_device,
                        tuple(submesh_device.shape),
                        dims=shard_latents_dims,
                    )

                # Convert latents from NHWC (B, H, W, C) to NCHW (B, C, H, W) for new transformer
                latents_nchw = latents.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

                tt_initial_latents = ttnn.from_torch(
                    latents_nchw,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=latents_mesh_mapper,
                )

                if len(self._intermediate_noise_list) <= i:
                    self._intermediate_noise_list.append(
                        ttnn.from_torch(
                            latents_nchw,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16,
                            device=submesh_device,
                            mesh_mapper=latents_mesh_mapper,
                        )
                    )
                    self._sigma_difference_list.append(ttnn.clone(self._intermediate_noise_list[-1]))
                if traced:
                    if self._trace is None:
                        # Push inputs to device
                        tt_initial_latents = tt_initial_latents.to(submesh_device)
                        tt_prompt_embeds = tt_prompt_embeds.to(submesh_device)
                        tt_pooled_prompt_embeds = tt_pooled_prompt_embeds.to(submesh_device)
                    else:
                        # Copy inputs to trace
                        ttnn.copy_host_to_device_tensor(tt_initial_latents, self._trace[i].spatial_input)
                        ttnn.copy_host_to_device_tensor(tt_prompt_embeds, self._trace[i].prompt_input)
                        ttnn.copy_host_to_device_tensor(tt_pooled_prompt_embeds, self._trace[i].pooled_projection_input)
                        # Ensure trace inputs are passed to function
                        tt_initial_latents = self._trace[i].spatial_input
                        tt_prompt_embeds = self._trace[i].prompt_input
                        tt_pooled_prompt_embeds = self._trace[i].pooled_projection_input

                tt_prompt_embeds_list.append(tt_prompt_embeds)
                tt_pooled_prompt_embeds_list.append(tt_pooled_prompt_embeds)
                tt_latents_step_list.append(tt_initial_latents)

            logger.info("denoising...")
            denoising_start_time = time.time()

            for i, t in enumerate(tqdm.tqdm(timesteps)):
                with timer.time_step("denoising_step") if timer else nullcontext():
                    sigma_difference = self._scheduler.sigmas[i + 1] - self._scheduler.sigmas[i]

                    tt_timestep_list = []
                    tt_sigma_difference_list = []
                    for submesh_device in self.submesh_devices:
                        # New transformer expects timestep as [B] tensor, not [1, 1, 1, 1]
                        # Create as [batch_size] tensor
                        batch_size = (
                            tt_latents_step_list[0].shape[0] if hasattr(tt_latents_step_list[0], "shape") else 1
                        )
                        tt_timestep = ttnn.full(
                            [batch_size],
                            fill_value=t,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.float32,
                            device=submesh_device if not traced else None,
                        )
                        tt_timestep_list.append(tt_timestep)

                        tt_sigma_difference_list.append(
                            ttnn.full(
                                tt_latents_step_list[0].shape,
                                fill_value=sigma_difference,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16,
                                device=None,  # We'll copy to device when needed
                            )
                        )

                    tt_latents_step_list = self._step(
                        timestep=tt_timestep_list,
                        latents=tt_latents_step_list,  # tt_latents,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        prompt_embeds=tt_prompt_embeds_list,
                        pooled_prompt_embeds=tt_pooled_prompt_embeds_list,
                        guidance_scale=guidance_scale,
                        sigma_difference=tt_sigma_difference_list,
                        prompt_sequence_length=333,
                        spatial_sequence_length=4096,
                        traced=traced,
                    )

            denoising_end_time = time.time()

            logger.info("decoding image...")

            with timer.time_section("vae_decoding") if timer else nullcontext():
                image_decoding_start_time = time.time()
                decoded_output = self._vae_decode(tt_latents_step_list[self.vae_submesh_idx], width, height)
                decoded_output = ttnn.to_torch(ttnn.get_device_tensors(decoded_output)[0]).permute(0, 3, 1, 2)

                # HACK: reshape submesh device 0 from 1D to 2D
                if self.desired_encoder_submesh_shape != self.original_submesh_shape:
                    # If reshaping, vae device is same as encoder device
                    self.encoder_device.reshape(ttnn.MeshShape(*self.original_submesh_shape))
                # image = self._torch_vae.decoder(tt_latents)
                image = self._image_processor.postprocess(decoded_output, output_type="pt")
                print(f"postprocessed image shape: {image.shape}")
                assert isinstance(image, torch.Tensor)
                image_decoding_end_time = time.time()

                output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

                end_time = time.time()

                logger.info(f"prompt encoding duration: {prompt_encoding_end_time - prompt_encoding_start_time}")
                logger.info(f"denoising duration: {denoising_end_time - denoising_start_time}")
                logger.info(f"image decoding duration: {image_decoding_end_time - image_decoding_start_time}")
                logger.info(f"total runtime: {end_time - start_time}")

        return output

    def _step(
        self,
        *,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        latents: List[ttnn.Tensor],  # device tensor
        timestep: List[ttnn.Tensor],  # host tensor
        pooled_prompt_embeds: List[ttnn.Tensor],  # device tensor
        prompt_embeds: List[ttnn.Tensor],  # device tensor
        sigma_difference: List[ttnn.Tensor],  # device tensor
        prompt_sequence_length: int,
        spatial_sequence_length: int,
        traced: bool,
    ) -> List[ttnn.Tensor]:
        def inner(latent, prompt, pooled_projection, timestep, cfg_index):
            if do_classifier_free_guidance and not self.dit_parallel_config.cfg_parallel.factor > 1:
                latent_model_input = ttnn.concat([latent, latent])
            else:
                latent_model_input = latent

            # New transformer API: (x, t, y, context) where:
            # x: (B, C, H, W) in NCHW format (already in this format)
            # t: (B,) timesteps - need to extract from timestep tensor
            # y: (B, D) class embeddings from pooled_projections
            # context: (B, L, D') context embeddings from prompt_embed

            # Extract batch dimension and reshape inputs
            # latent_model_input: [1, B, C, H, W] or [B, C, H, W] -> [B, C, H, W]
            if len(latent_model_input.shape) == 5:  # [1, B, C, H, W]
                x = ttnn.reshape(
                    latent_model_input,
                    [
                        latent_model_input.shape[1],
                        latent_model_input.shape[2],
                        latent_model_input.shape[3],
                        latent_model_input.shape[4],
                    ],
                )
            else:
                x = latent_model_input

            # timestep: [1, B] or [B] -> [B]
            if len(timestep.shape) == 2:  # [1, B]
                t = ttnn.reshape(timestep, [timestep.shape[1]])
            else:
                t = timestep

            # prompt: [1, B, L, D'] or [B, L, D'] -> [B, L, D']
            if len(prompt.shape) == 4:  # [1, B, L, D']
                context = ttnn.reshape(prompt, [prompt.shape[1], prompt.shape[2], prompt.shape[3]])
            else:
                context = prompt

            # pooled_projection: [1, B, D''] or [B, D''] -> [B, D'']
            if len(pooled_projection.shape) == 3:  # [1, B, D'']
                y = ttnn.reshape(pooled_projection, [pooled_projection.shape[1], pooled_projection.shape[2]])
            else:
                y = pooled_projection

            return self.transformers[cfg_index](
                x=x,
                t=t,
                y=y,
                context=context,
            )

        if traced and self._trace is None:
            print(f"Tracing...")
            self._trace = [None for _ in self.submesh_devices]
            for submesh_id, submesh_device in enumerate(self.submesh_devices):
                print(f"Tracing submesh {submesh_id}")
                latent_device = latents[submesh_id]  # already on device
                prompt_device = prompt_embeds[submesh_id]  # already on device
                pooled_projection_device = pooled_prompt_embeds[submesh_id]  # already on device
                timestep_device = timestep[submesh_id].to(submesh_device)

                print("compile run")
                pred = inner(
                    latent_device,
                    prompt_device,
                    pooled_projection_device,
                    timestep_device,
                    submesh_id,
                )

                if submesh_id == self.vae_submesh_idx:
                    print("Initializing VAE buffers for safe tracing...")
                    self._vae_decode(latent_device, self._prepared_width, self._prepared_height)
                    if self.desired_encoder_submesh_shape != self.original_submesh_shape:
                        self.encoder_device.reshape(ttnn.MeshShape(*self.original_submesh_shape))

                ttnn.synchronize_device(submesh_device)
                # ttnn.synchronize_device(self.submesh_devices[0])
                # ttnn.synchronize_device(self.submesh_devices[1])

                print("begin trace capture")
                trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
                pred = inner(
                    latent_device,
                    prompt_device,
                    pooled_projection_device,
                    timestep_device,
                    submesh_id,
                )
                ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)
                ttnn.synchronize_device(submesh_device)
                # ttnn.synchronize_device(self.submesh_devices[0])
                # ttnn.synchronize_device(self.submesh_devices[1])
                print("done sync after trace capture")

                self._trace[submesh_id] = PipelineTrace(
                    spatial_input=latent_device,
                    prompt_input=prompt_device,
                    pooled_projection_input=pooled_projection_device,
                    timestep_input=timestep_device,
                    latents_output=pred,
                    tid=trace_id,
                )

        noise_pred_list = []
        if traced:
            for submesh_id, submesh_device in enumerate(self.submesh_devices):
                ttnn.copy_host_to_device_tensor(timestep[submesh_id], self._trace[submesh_id].timestep_input)
                ttnn.execute_trace(submesh_device, self._trace[submesh_id].tid, cq_id=0, blocking=False)
                noise_pred_list.append(self._trace[submesh_id].latents_output)
        else:
            for submesh_id, submesh_device in enumerate(self.submesh_devices):
                noise_pred = inner(
                    latents[submesh_id],
                    prompt_embeds[submesh_id],
                    pooled_prompt_embeds[submesh_id],
                    timestep[submesh_id],
                    submesh_id,
                )
                noise_pred_list.append(noise_pred)

        if do_classifier_free_guidance:
            if not self.dit_parallel_config.cfg_parallel.factor > 1:
                split_pos = noise_pred_list[0].shape[0] // 2
                uncond = noise_pred_list[0][0:split_pos]
                cond = noise_pred_list[0][split_pos:]
                noise_pred_list[0] = uncond + guidance_scale * (cond - uncond)
            else:
                # uncond and cond are replicated, so it is fine to get a single tensor from each
                uncond = ttnn.to_torch(ttnn.get_device_tensors(noise_pred_list[0])[0].cpu(blocking=True)).to(
                    torch.float32
                )
                cond = ttnn.to_torch(ttnn.get_device_tensors(noise_pred_list[1])[0].cpu(blocking=True)).to(
                    torch.float32
                )

                torch_noise_pred = uncond + guidance_scale * (cond - uncond)

                # For n150 or when sequence_parallel factor is 1, use replication
                is_n150_0 = tuple(self.submesh_devices[0].shape) == (1, 1)
                is_n150_1 = tuple(self.submesh_devices[1].shape) == (1, 1)
                if is_n150_0 or self.dit_parallel_config.sequence_parallel.factor == 1:
                    noise_mesh_mapper_0 = ttnn.ReplicateTensorToMesh(self.submesh_devices[0])
                else:
                    shard_latents_dims = [None, None]
                    shard_latents_dims[self.dit_parallel_config.sequence_parallel.mesh_axis] = 2
                    noise_mesh_mapper_0 = ttnn.ShardTensor2dMesh(
                        self.submesh_devices[0],
                        tuple(self.submesh_devices[0].shape),
                        dims=shard_latents_dims,
                    )

                if is_n150_1 or self.dit_parallel_config.sequence_parallel.factor == 1:
                    noise_mesh_mapper_1 = ttnn.ReplicateTensorToMesh(self.submesh_devices[1])
                else:
                    shard_latents_dims = [None, None]
                    shard_latents_dims[self.dit_parallel_config.sequence_parallel.mesh_axis] = 2
                    noise_mesh_mapper_1 = ttnn.ShardTensor2dMesh(
                        self.submesh_devices[1],
                        tuple(self.submesh_devices[1].shape),
                        dims=shard_latents_dims,
                    )

                noise_pred_list[0] = ttnn.from_torch(
                    torch_noise_pred,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=None,  # self.submesh_devices[0],
                    mesh_mapper=noise_mesh_mapper_0,
                )

                noise_pred_list[1] = ttnn.from_torch(
                    torch_noise_pred,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=None,  # self.submesh_devices[1],
                    mesh_mapper=noise_mesh_mapper_1,
                )

        for submesh_id, submesh_device in enumerate(self.submesh_devices):
            # Convert noise_pred to host tensor if it's on device, then copy to the pre-allocated device buffer
            noise_pred = noise_pred_list[submesh_id]
            # Check if tensor is on device (when cfg_parallel.factor is not > 1)
            if hasattr(noise_pred, "storage_type") and noise_pred.storage_type() == ttnn.StorageType.DEVICE:
                # Allocate a host tensor and copy from device to host, then to device buffer
                noise_pred_host = ttnn.allocate_tensor_on_host(
                    noise_pred.shape,
                    noise_pred.dtype,
                    noise_pred.layout,
                    submesh_device,
                )
                ttnn.copy_device_to_host_tensor(noise_pred, noise_pred_host, blocking=True)
                ttnn.copy_host_to_device_tensor(noise_pred_host, self._intermediate_noise_list[submesh_id])
            else:
                # Already a host tensor
                ttnn.copy_host_to_device_tensor(noise_pred, self._intermediate_noise_list[submesh_id])

            ttnn.copy_host_to_device_tensor(sigma_difference[submesh_id], self._sigma_difference_list[submesh_id])
            ttnn.multiply_(
                self._sigma_difference_list[submesh_id], self._intermediate_noise_list[submesh_id]
            )  # This allocates during trace. Need to investigate
            ttnn.add_(latents[submesh_id], self._sigma_difference_list[submesh_id])

        return latents

    def _vae_decode(self, tt_latents, width, height):
        ttnn.synchronize_device(self.vae_device)

        # Skip all_gather if on single device (sequence_parallel factor is 1)
        if self.dit_parallel_config.sequence_parallel.factor > 1:
            tt_latents = self.ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
                tt_latents, dim=2, mesh_axis=self.dit_parallel_config.sequence_parallel.mesh_axis
            )

        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
        torch_latents = (torch_latents / self._torch_vae_scaling_factor) + self._torch_vae_shift_factor
        # New transformer already unpatchifies and outputs in NCHW format [B, C, H, W]
        # VAE expects NHWC format [B, H, W, C], so convert
        torch_latents = torch_latents.permute(0, 2, 3, 1)  # NCHW -> NHWC

        if self.desired_encoder_submesh_shape != self.original_submesh_shape:
            # HACK: reshape submesh device 0 from 2D to 1D
            # If reshaping, vae device is same as encoder device
            self.encoder_device.reshape(ttnn.MeshShape(*self.desired_encoder_submesh_shape))

        tt_latents = ttnn.from_torch(
            torch_latents,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=None,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.vae_device),
        )

        if self._vae_input_latents is None:
            self._vae_input_latents = tt_latents.to(self.vae_device)
        else:
            ttnn.copy_host_to_device_tensor(tt_latents, self._vae_input_latents)

        decoded_output = self._vae_decoder(self._vae_input_latents)
        return decoded_output

    def _encode_prompts(
        self,
        *,
        prompt_1: list[str],
        prompt_2: list[str],
        prompt_3: list[str],
        negative_prompt_1: list[str],
        negative_prompt_2: list[str],
        negative_prompt_3: list[str],
        num_images_per_prompt: int,
        max_t5_sequence_length: int,
        do_classifier_free_guidance: bool,
        clip_skip: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timer = self.timing_collector

        tokenizer_max_length = self._tokenizer_1.model_max_length

        with timer.time_section("clip_encoding") if timer else nullcontext():
            prompt_embed, pooled_prompt_embed = _get_clip_prompt_embeds(
                prompt=prompt_1,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_1,
                text_encoder=self._text_encoder_1,
                tokenizer_max_length=tokenizer_max_length,
                ttnn_device=self.encoder_device,
                encoder_parallel_config=self.encoder_parallel_config,
                clip_skip=clip_skip,
            )

            prompt_2_embed, pooled_prompt_2_embed = _get_clip_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_2,
                text_encoder=self._text_encoder_2,
                tokenizer_max_length=tokenizer_max_length,
                ttnn_device=self.encoder_device,
                encoder_parallel_config=self.encoder_parallel_config,
                clip_skip=clip_skip,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        with timer.time_section("t5_encoding") if timer else nullcontext():
            t5_prompt_embed = _get_t5_prompt_embeds(
                device=self.encoder_device,
                encoder_parallel_config=self.encoder_parallel_config,
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_t5_sequence_length,
                tokenizer=self._tokenizer_3,
                text_encoder=self._text_encoder_3,
                tokenizer_max_length=tokenizer_max_length,
                joint_attention_dim=self._joint_attention_dim,
            )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if not do_classifier_free_guidance:
            return prompt_embeds, pooled_prompt_embeds

        with timer.time_section("clip_encoding") if timer else nullcontext():
            negative_prompt_embed, negative_pooled_prompt_embed = _get_clip_prompt_embeds(
                prompt=negative_prompt_1,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_1,
                text_encoder=self._text_encoder_1,
                tokenizer_max_length=tokenizer_max_length,
                encoder_parallel_config=self.encoder_parallel_config,
                ttnn_device=self.encoder_device,
                clip_skip=clip_skip,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = _get_clip_prompt_embeds(
                prompt=negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_2,
                text_encoder=self._text_encoder_2,
                tokenizer_max_length=tokenizer_max_length,
                encoder_parallel_config=self.encoder_parallel_config,
                ttnn_device=self.encoder_device,
                clip_skip=clip_skip,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

        with timer.time_section("t5_encoding") if timer else nullcontext():
            t5_negative_prompt_embed = _get_t5_prompt_embeds(
                device=self.encoder_device,
                encoder_parallel_config=self.encoder_parallel_config,
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_t5_sequence_length,
                tokenizer=self._tokenizer_3,
                text_encoder=self._text_encoder_3,
                tokenizer_max_length=tokenizer_max_length,
                joint_attention_dim=self._joint_attention_dim,
            )

        negative_clip_prompt_embeds = torch.nn.functional.pad(
            negative_clip_prompt_embeds,
            (
                0,
                t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1],
            ),
        )

        negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
        negative_pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        return prompt_embeds, pooled_prompt_embeds

    def t5_enabled(self):
        return self._text_encoder_3 is not None


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_clip_prompt_embeds(
    *,
    clip_skip: int | None = None,
    device: torch.device | None = None,
    ttnn_device: ttnn.Device | None = None,
    encoder_parallel_config: EncoderParallelConfig | None = None,
    num_images_per_prompt: int,
    prompt: list[str],
    text_encoder: CLIPEncoder,
    tokenizer_max_length: int,
    tokenizer: CLIPTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer_max_length} tokens: {removed_text}"
        )

    tt_text_input_ids = ttnn.from_torch(
        text_input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_device),
    )

    # Call the new CLIPEncoder with projection enabled
    encoder_output, projected_output = text_encoder(
        prompt_tokenized=tt_text_input_ids,
        mesh_device=ttnn_device,
        with_projection=True,
    )

    # Handle clip_skip by selecting the appropriate hidden state layer
    if clip_skip is None:
        # Use the second-to-last layer (like the original implementation)
        sequence_embeddings = encoder_output[-2]
    else:
        layer_index = -(clip_skip + 2)
        if abs(layer_index) > len(encoder_output):
            layer_index = -2
        sequence_embeddings = encoder_output[layer_index]

    prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(sequence_embeddings)[0])

    pooled_prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(projected_output)[0])

    prompt_embeds = prompt_embeds.to(device=device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds, pooled_prompt_embeds


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_t5_prompt_embeds(
    prompt: list[str],
    *,
    torch_device: torch.device | None = None,
    device: ttnn.Device,
    encoder_parallel_config: EncoderParallelConfig | None = None,
    joint_attention_dim: int,
    max_sequence_length: int,
    num_images_per_prompt: int,
    text_encoder: T5Encoder | None,
    tokenizer_max_length: int,
    tokenizer: T5TokenizerFast,
) -> torch.Tensor:
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if text_encoder is None:
        return torch.zeros(
            (
                batch_size * num_images_per_prompt,
                tokenizer_max_length,
                joint_attention_dim,
            ),
            device=torch_device,
            dtype=torch.bfloat16,
        )

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    tt_text_input_ids = ttnn.from_torch(
        text_input_ids,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    # Call the new T5Encoder
    hidden_states = text_encoder(prompt=tt_text_input_ids, device=device)

    # Use the final layer output (last element in the list)
    tt_prompt_embeds = hidden_states[-1]
    tt_prompt_embeds = ttnn.get_device_tensors(tt_prompt_embeds)[0]
    prompt_embeds = ttnn.to_torch(tt_prompt_embeds)

    prompt_embeds = prompt_embeds.to(device=torch_device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    return prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
