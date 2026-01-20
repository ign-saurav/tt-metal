# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Union, List
from PIL import Image

import os
import tqdm
import ttnn
import torch
import numpy as np
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from loguru import logger
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from models.perf.benchmarking_utils import BenchmarkProfiler

from ...encoders.clip.model_clip import CLIPConfig, CLIPEncoder
from ...encoders.t5.model_t5 import T5Config, T5Encoder
from ...models.transformers.transformer_flux1 import Flux1Transformer

from ...models.vae.vae_flux1 import VAEEncoder, VAEDecoder
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils.padding import PaddingConfig
from ...utils import cache


# NOTE: Ttnn VAE encoder and decoder only supports 1024x1024 image resolutions currently.
PREFERRED_KONTEXT_RESOLUTIONS = [
    (1024, 1024),
]


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


@dataclass
class PipelineTrace:
    tid: int
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    guidance_input: ttnn.Tensor
    spatial_rope_cos: ttnn.Tensor
    spatial_rope_sin: ttnn.Tensor
    prompt_rope_cos: ttnn.Tensor
    prompt_rope_sin: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor
    latents_output: ttnn.Tensor


class Flux1KontextPipeline:
    T5_SEQUENCE_LENGTH = 512

    def __init__(
        self,
        *,
        checkpoint_name: str,
        mesh_device: ttnn.MeshDevice,
        use_torch_t5_text_encoder: bool = False,
        use_torch_clip_text_encoder: bool = False,
        use_torch_vae: bool = False,
        parallel_config: DiTParallelConfig,
        topology: ttnn.Topology,
        num_links: int,
    ) -> None:
        self._mesh_device = mesh_device
        self._parallel_config = parallel_config
        self.sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        # Create submeshes based on CFG parallel factor
        submesh_shape = list(mesh_device.shape)
        submesh_shape[parallel_config.cfg_parallel.mesh_axis] //= parallel_config.cfg_parallel.factor
        logger.info(f"Parallel config: {parallel_config}")
        logger.info(f"Original mesh shape: {mesh_device.shape}")
        logger.info(f"Creating submeshes with shape {submesh_shape}")
        self._submesh_devices = self._mesh_device.create_submeshes(ttnn.MeshShape(*submesh_shape))
        self._ccl_managers = [
            CCLManager(submesh_device, num_links=num_links, topology=topology)
            for submesh_device in self._submesh_devices
        ]

        self.encoder_submesh_idx = 0  # Use submesh 0 for encoder
        encoder_device = self._submesh_devices[self.encoder_submesh_idx]

        self.vae_submesh_idx = 1
        if len(self._submesh_devices) == 1:
            self.vae_submesh_idx = 0  # Only one sub mesh device is present
        vae_device = self._submesh_devices[self.vae_submesh_idx]

        # Create encoder parallel config
        encoder_parallel_config = EncoderParallelConfig(tensor_parallel=parallel_config.tensor_parallel)
        self.encoder_parallel_config = encoder_parallel_config
        self.encoder_device = encoder_device

        vae_parallel_config = VAEParallelConfig(tensor_parallel=parallel_config.tensor_parallel)
        self.vae_parallel_config = vae_parallel_config
        self.vae_device = vae_device

        logger.info("loading models...")
        self._tokenizer_1 = CLIPTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")
        self._tokenizer_2 = T5TokenizerFast.from_pretrained(checkpoint_name, subfolder="tokenizer_2")
        torch_text_encoder_1 = CLIPTextModel.from_pretrained(checkpoint_name, subfolder="text_encoder")
        torch_text_encoder_2 = T5EncoderModel.from_pretrained(checkpoint_name, subfolder="text_encoder_2")
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")
        self._torch_vae = AutoencoderKL.from_pretrained(checkpoint_name, subfolder="vae")

        torch_transformer = FluxTransformer2DModel.from_pretrained(
            checkpoint_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,  # bfloat16 is the native datatype of the model
        )
        torch_transformer.eval()

        assert isinstance(self._tokenizer_1, CLIPTokenizer)
        assert isinstance(self._tokenizer_2, T5TokenizerFast)
        assert isinstance(torch_text_encoder_1, CLIPTextModel)
        assert isinstance(torch_text_encoder_2, T5EncoderModel)
        assert isinstance(self._scheduler, FlowMatchEulerDiscreteScheduler)
        assert isinstance(self._torch_vae, AutoencoderKL)
        assert isinstance(torch_transformer, FluxTransformer2DModel)

        logger.info("creating TT-NN transformer...")

        if torch_transformer.config.num_attention_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                torch_transformer.config.num_attention_heads,
                torch_transformer.config.attention_head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        self.transformers = []
        for i, submesh_device in enumerate(self._submesh_devices):
            tt_transformer = Flux1Transformer(
                patch_size=torch_transformer.config.patch_size,
                in_channels=torch_transformer.config.in_channels,
                num_layers=torch_transformer.config.num_layers,
                num_single_layers=torch_transformer.config.num_single_layers,
                attention_head_dim=torch_transformer.config.attention_head_dim,
                num_attention_heads=torch_transformer.config.num_attention_heads,
                joint_attention_dim=torch_transformer.config.joint_attention_dim,
                pooled_projection_dim=torch_transformer.config.pooled_projection_dim,
                out_channels=torch_transformer.out_channels,
                axes_dims_rope=torch_transformer.config.axes_dims_rope,
                with_guidance_embeds=torch_transformer.config.guidance_embeds,
                mesh_device=submesh_device,
                ccl_manager=self._ccl_managers[i],
                parallel_config=parallel_config,
                padding_config=padding_config,
            )

            model_name = os.path.basename(checkpoint_name)
            if not cache.initialize_from_cache(
                tt_transformer,
                torch_transformer,
                model_name,
                "transformer",
                parallel_config,
                tuple(submesh_device.shape),
            ):
                logger.info(f"Loading transformer weights from PyTorch state dict")
                tt_transformer.load_torch_state_dict(torch_transformer.state_dict())
                logger.info(f"Successfully loaded transformer weights")

            self.transformers.append(tt_transformer)
            ttnn.synchronize_device(submesh_device)

        self._pos_embed = torch_transformer.pos_embed

        self._num_channels_latents = torch_transformer.config.in_channels // 4
        self._joint_attention_dim = torch_transformer.config.joint_attention_dim
        self._patch_size = torch_transformer.config.patch_size
        self._with_guidance_embeds = torch_transformer.config.guidance_embeds

        self._block_out_channels = self._torch_vae.config.block_out_channels
        self._latent_channels = self._torch_vae.config.latent_channels
        self._latents_scaling = self._torch_vae.config.scaling_factor
        self._latents_shift = self._torch_vae.config.shift_factor

        self._vae_scale_factor = 2 ** (len(self._block_out_channels) - 1)
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._vae_scale_factor * 2)
        self.default_sample_size = 128

        if use_torch_clip_text_encoder:
            self._text_encoder_1 = torch_text_encoder_1.eval()
        else:
            logger.info("creating TT-NN CLIP text encoder...")
            clip_config_1 = CLIPConfig(
                vocab_size=torch_text_encoder_1.config.vocab_size,
                embed_dim=torch_text_encoder_1.config.hidden_size,
                ff_dim=torch_text_encoder_1.config.intermediate_size,
                num_heads=torch_text_encoder_1.config.num_attention_heads,
                num_hidden_layers=torch_text_encoder_1.config.num_hidden_layers,
                max_prompt_length=77,
                layer_norm_eps=torch_text_encoder_1.config.layer_norm_eps,
                attention_dropout=torch_text_encoder_1.config.attention_dropout,
                hidden_act=torch_text_encoder_1.config.hidden_act,
            )

            self._text_encoder_1 = CLIPEncoder(
                config=clip_config_1,
                mesh_device=self.encoder_device,
                ccl_manager=self._ccl_managers[self.encoder_submesh_idx],
                parallel_config=encoder_parallel_config,
                eos_token_id=2,  # default EOS token ID for CLIP
            )

            self._text_encoder_1.load_torch_state_dict(torch_text_encoder_1.state_dict())

        if use_torch_t5_text_encoder:
            self._text_encoder_2 = torch_text_encoder_2.eval()
        else:
            logger.info("creating TT-NN T5 text encoder...")
            t5_config = T5Config(
                vocab_size=torch_text_encoder_2.config.vocab_size,
                embed_dim=torch_text_encoder_2.config.d_model,
                ff_dim=torch_text_encoder_2.config.d_ff,
                kv_dim=torch_text_encoder_2.config.d_kv,
                num_heads=torch_text_encoder_2.config.num_heads,
                num_hidden_layers=torch_text_encoder_2.config.num_layers,
                max_prompt_length=self.T5_SEQUENCE_LENGTH,
                layer_norm_eps=torch_text_encoder_2.config.layer_norm_epsilon,
                relative_attention_num_buckets=torch_text_encoder_2.config.relative_attention_num_buckets,
                relative_attention_max_distance=torch_text_encoder_2.config.relative_attention_max_distance,
            )

            self._text_encoder_2 = T5Encoder(
                config=t5_config,
                mesh_device=self.encoder_device,
                ccl_manager=self._ccl_managers[self.encoder_submesh_idx],
                parallel_config=encoder_parallel_config,
            )

            if not cache.initialize_from_cache(
                self._text_encoder_2,
                torch_text_encoder_2,
                model_name,
                "t5_text_encoder",
                encoder_parallel_config,
                tuple(self.encoder_device.shape),
            ):
                logger.info(f"Loading T5 text encoder weights from PyTorch state dict")
                self._text_encoder_2.load_torch_state_dict(torch_text_encoder_2.state_dict())
                logger.info(f"Successfully loaded T5 text encoder weights")

        self._traces = None

        # intermediate buffers for safe tracing
        self._vae_input_latents = None

        ttnn.synchronize_device(self.encoder_device)

        self.use_torch_vae = use_torch_vae
        if use_torch_vae:
            self._vae_encoder = self._torch_vae.encoder
            self._vae_decoder = self._torch_vae.decoder
        else:
            self._vae_encoder = VAEEncoder.from_torch(
                torch_ref=self._torch_vae.encoder,
                mesh_device=self.vae_device,
                parallel_config=self.vae_parallel_config,
                ccl_manager=self._ccl_managers[self.vae_submesh_idx],
            )
            self._vae_decoder = VAEDecoder.from_torch(
                torch_ref=self._torch_vae.decoder,
                mesh_device=self.vae_device,
                parallel_config=self.vae_parallel_config,
                ccl_manager=self._ccl_managers[self.vae_submesh_idx],
            )

        self.synchronize_devices()

    @staticmethod
    def create_pipeline(
        checkpoint_name: str,
        mesh_device: ttnn.MeshDevice,
        sp_config: Optional[tuple] = None,
        tp_config: Optional[tuple] = None,
        cfg_config: Optional[tuple] = None,
        use_torch_t5_text_encoder: bool = False,
        use_torch_clip_text_encoder: bool = False,
        use_torch_vae: bool = False,
        num_links: Optional[int] = None,
        topology: ttnn.Topology = ttnn.Topology.Linear,
    ):
        # defatult config per mesh shape
        default_config = {
            (1, 4): {"sp": (1, 0), "tp": (4, 1), "cfg_config": (1, 0), "num_links": 1},
            (2, 4): {"sp": (2, 0), "tp": (4, 1), "cfg_config": (2, 1), "num_links": 1},
        }

        # get config from user or default if not provided
        sp_factor, sp_axis = sp_config or default_config[tuple(mesh_device.shape)]["sp"]
        tp_factor, tp_axis = tp_config or default_config[tuple(mesh_device.shape)]["tp"]
        cfg_factor, cfg_axis = cfg_config or default_config[tuple(mesh_device.shape)]["cfg_config"]
        num_links = num_links or default_config[tuple(mesh_device.shape)]["num_links"]

        dit_parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=cfg_factor, mesh_axis=cfg_axis),
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
        )

        pipeline = Flux1KontextPipeline(
            checkpoint_name=checkpoint_name,
            mesh_device=mesh_device,
            use_torch_t5_text_encoder=use_torch_t5_text_encoder,
            use_torch_clip_text_encoder=use_torch_clip_text_encoder,
            use_torch_vae=use_torch_vae,
            parallel_config=dit_parallel_config,
            topology=topology,
            num_links=num_links,
        )

        return pipeline

    def run_single_prompt(
        self,
        *,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = "",
        negative_prompt: Union[str, List[str]] = None,
        cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        traced: bool = True,
        timer: BenchmarkProfiler = None,
        timer_iteration: int = 0,
    ) -> List[Image.Image]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if negative_prompt is not None:
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        return self(
            image=image,
            width=width,
            height=height,
            prompt_1=prompt,
            prompt_2=prompt,
            negative_prompt_1=negative_prompt,
            negative_prompt_2=negative_prompt,
            cfg_scale=cfg_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            traced=traced,
            timer=timer,
            timer_iteration=timer_iteration,
        )

    # adapted from https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/flux/pipeline_flux_kontext.py
    def _get_t5_prompt_embeds(
        self,
        *,
        prompts: Union[str, List[str]],
        text_encoder: Union[T5Encoder, T5EncoderModel],
        tokenizer: T5TokenizerFast,
        max_sequence_length: int = 512,
        num_images_per_prompt: int = 1,
        mesh_device: Optional[ttnn.MeshDevice] = None,
    ) -> torch.Tensor:
        prompts = [prompts] if isinstance(prompts, str) else prompts
        batch_size = len(prompts)

        tokens = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
        ).input_ids

        untruncated_tokens = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
        ).input_ids

        if untruncated_tokens.shape[-1] >= tokens.shape[-1] and not torch.equal(tokens, untruncated_tokens):
            logger.warning("T5 input text was truncated")

        if isinstance(text_encoder, T5Encoder):
            assert mesh_device is not None

            tt_tokens = ttnn.from_torch(
                tokens,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.uint32,
                device=mesh_device,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
            )
            tt_hidden_states = text_encoder(prompt=tt_tokens, device=mesh_device)
            tt_prompt_embeds = tt_hidden_states[-1]

            prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
        else:
            tokens = tokens.to(device=text_encoder.device)
            with torch.no_grad():
                output = text_encoder.forward(tokens)
            prompt_embeds = output.last_hidden_state.to("cpu")

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # adapted from https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/flux/pipeline_flux_kontext.py
    def _get_clip_prompt_embeds(
        self,
        *,
        prompts: Union[str, List[str]],
        text_encoder: Union[CLIPEncoder, CLIPTextModel],
        tokenizer: CLIPTokenizer,
        sequence_length: int,
        num_images_per_prompt: int = 1,
        mesh_device: Optional[ttnn.MeshDevice] = None,
    ) -> torch.Tensor:
        prompts = [prompts] if isinstance(prompts, str) else prompts
        batch_size = len(prompts)

        tokens = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
        ).input_ids

        untruncated_tokens = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
        ).input_ids

        if untruncated_tokens.shape[-1] >= tokens.shape[-1] and not torch.equal(tokens, untruncated_tokens):
            logger.warning("CLIP input text was truncated")

        if isinstance(text_encoder, CLIPEncoder):
            assert mesh_device is not None

            tt_tokens = ttnn.from_torch(
                tokens,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
            )

            _, tt_pooled_prompt_embeds = text_encoder(
                prompt_tokenized=tt_tokens,
                mesh_device=mesh_device,
            )

            pooled_prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_pooled_prompt_embeds)[0])
        else:
            tokens = tokens.to(device=text_encoder.device)
            with torch.no_grad():
                output = text_encoder.forward(tokens, output_hidden_states=True)
            pooled_prompt_embeds = output.pooler_output.to("cpu")

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return pooled_prompt_embeds

    def _encode_prompts_partial(
        self,
        *,
        prompt_1: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        timer: BenchmarkProfiler = None,
        timer_iteration: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer_max_length = self._tokenizer_1.model_max_length

        prompt_1 = [prompt_1] if isinstance(prompt_1, str) else prompt_1
        prompt_2 = prompt_2 or prompt_1
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # We only use the pooled prompt output from the CLIPTextModel
        with timer("clip_encoding", timer_iteration) if timer else nullcontext():
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompts=prompt_1,
                text_encoder=self._text_encoder_1,
                tokenizer=self._tokenizer_1,
                sequence_length=tokenizer_max_length,
                num_images_per_prompt=num_images_per_prompt,
                mesh_device=self.encoder_device,
            )

        with timer("t5_encoding", timer_iteration) if timer else nullcontext():
            prompt_embeds = self._get_t5_prompt_embeds(
                prompts=prompt_2,
                text_encoder=self._text_encoder_2,
                tokenizer=self._tokenizer_2,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
                mesh_device=self.encoder_device,
            )

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompts(
        self,
        *,
        prompt_1: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_1: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        cfg_enabled: bool = False,
        timer: BenchmarkProfiler = None,
        timer_iteration: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prompt_embeds is None:
            prompt_embeds, pooled_prompt_embeds = self._encode_prompts_partial(
                prompt_1=prompt_1,
                prompt_2=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                timer=timer,
                timer_iteration=timer_iteration,
            )

        if cfg_enabled:
            if negative_prompt_embeds is None:
                negative_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompts_partial(
                    prompt_1=negative_prompt_1,
                    prompt_2=negative_prompt_2,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    timer=timer,
                    timer_iteration=timer_iteration,
                )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        text_ids = torch.zeros(prompt_embeds.shape[1], 3)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    @staticmethod
    # adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
    def _prepare_latent_image_ids(height: int, width: int) -> torch.Tensor:
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        return latent_image_ids.reshape(latent_image_id_height * latent_image_id_width, latent_image_id_channels)

    @staticmethod
    # adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
    def _pack_latents(
        latents: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        # B, C, H * P, W * Q -> B, H * W, C * P * Q
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    @staticmethod
    # adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
    def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
        # B, H * W, C * P * Q -> B, C, H * P, W * Q
        batch_size, num_patches, channels = latents.shape

        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        return latents.reshape(batch_size, channels // (2 * 2), height, width)

    def _vae_decode(self, tt_latents: ttnn.Tensor, width: int, height: int) -> torch.Tensor:
        ttnn.synchronize_device(self.vae_device)

        tt_latents = self._ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
            tt_latents,
            dim=1,
            mesh_axis=self.sp_axis,
            use_hyperparams=True,
        )

        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])

        # We need to slice the output latents since TTNN persistant buffer is created for combined latents
        if self.img_to_img:
            torch_latents = torch_latents[:, : self.output_latents_seq_length]

        torch_latents = self._unpack_latents(torch_latents, height, width, self._vae_scale_factor)
        torch_latents = (torch_latents / self._latents_scaling) + self._latents_shift

        if self.use_torch_vae:
            torch_latents = torch_latents.to(torch.float32)
            with torch.no_grad():
                decoded_output = self._vae_decoder(torch_latents)
        else:
            tt_latents = ttnn.from_torch(
                torch_latents.permute(0, 2, 3, 1),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                device=None,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.vae_device),
            )

            if self._vae_input_latents is None:
                self._vae_input_latents = tt_latents.to(self.vae_device)
            else:
                ttnn.copy_host_to_device_tensor(tt_latents, self._vae_input_latents)

            tt_decoded_output = self._vae_decoder(self._vae_input_latents)
            decoded_output = ttnn.to_torch(ttnn.get_device_tensors(tt_decoded_output)[0]).permute(0, 3, 1, 2)

        return decoded_output

    def _vae_encode(self, image: torch.Tensor) -> torch.Tensor:
        if self.use_torch_vae:
            with torch.no_grad():
                encoded_image = self._vae_encoder(image)
        else:
            tt_image = ttnn.from_torch(
                image.permute(0, 2, 3, 1),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                device=self.vae_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.vae_device),
            )
            encoded_image = self._vae_encoder(tt_image)
            encoded_image = ttnn.to_torch(ttnn.get_device_tensors(encoded_image)[0]).permute(0, 3, 1, 2)
        image_latents, _ = torch.chunk(encoded_image, 2, dim=1)

        image_latents = (image_latents - self._torch_vae.config.shift_factor) * self._torch_vae.config.scaling_factor

        return image_latents

    # adapted from https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/flux/pipeline_flux_kontext.py
    def prepare_latents(
        self,
        image: Optional[torch.Tensor],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        timer: BenchmarkProfiler = None,
        timer_iteration: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self._vae_scale_factor * 2))
        width = 2 * (int(width) // (self._vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        image_latents = image_ids = None
        if image is not None:
            image = image.to(dtype=torch.float32)
            if image.shape[1] != self._latent_channels:
                with timer("vae_encoding", timer_iteration) if timer else nullcontext():
                    image_latents = self._vae_encode(image=image)
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[2:]
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )
            image_ids = self._prepare_latent_image_ids(image_latent_height // 2, image_latent_width // 2)
            # image ids are the same as latent ids with the first dimension set to 1 instead of 0
            image_ids[..., 0] = 1

        latent_ids = self._prepare_latent_image_ids(height // 2, width // 2)

        if latents is None:
            if isinstance(generator, list):  # support for batched prompts with different seeds
                shape = (1,) + shape[1:]
                latents = [torch.randn(shape, generator=generator[i], dtype=torch.float32) for i in range(batch_size)]
                latents = torch.cat(latents, dim=0)
            else:
                latents = torch.randn(shape, generator=generator, dtype=torch.float32)
                latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(dtype=torch.float32)

        return latents, image_latents, latent_ids, image_ids

    def __call__(
        self,
        *,
        image: Optional[PipelineImageInput] = None,
        prompt_1: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_1: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        max_area: int = 1024**2,
        traced: bool = False,
        _auto_resize: bool = True,
        timer: BenchmarkProfiler = None,
        timer_iteration: int = 0,
    ) -> List[Image.Image]:
        if prompt_1 is not None and isinstance(prompt_1, str):
            prompt_count = 1
        elif prompt_1 is not None and isinstance(prompt_1, list):
            prompt_count = len(prompt_1)
        else:
            prompt_count = prompt_embeds.shape[0]

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        has_neg_prompt = negative_prompt_1 is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        cfg_enabled = cfg_scale > 1 and has_neg_prompt

        with timer("total", timer_iteration) if timer else nullcontext():
            height = height or self.default_sample_size * self._vae_scale_factor
            width = width or self.default_sample_size * self._vae_scale_factor
            assert height % (self._vae_scale_factor * self._patch_size) == 0
            assert width % (self._vae_scale_factor * self._patch_size) == 0

            original_height, original_width = height, width
            aspect_ratio = width / height
            width = round((max_area * aspect_ratio) ** 0.5)
            height = round((max_area / aspect_ratio) ** 0.5)

            multiple_of = self._vae_scale_factor * 2
            self.generation_width = width // multiple_of * multiple_of
            self.generation_height = height // multiple_of * multiple_of

            assert (
                self.generation_width,
                self.generation_height,
            ) in PREFERRED_KONTEXT_RESOLUTIONS, (
                f"Only {PREFERRED_KONTEXT_RESOLUTIONS} image resolutions are currently supported."
            )

            if self.generation_height != original_height or self.generation_width != original_width:
                logger.warning(
                    f"Generation `height` and `width` have been adjusted to {self.generation_height} and {self.generation_width} to fit the model requirements."
                )

            with timer("total_encoding", timer_iteration) if timer else nullcontext():
                logger.info("encoding prompts...")
                prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompts(
                    prompt_1=prompt_1,
                    prompt_2=prompt_2,
                    negative_prompt_1=negative_prompt_1,
                    negative_prompt_2=negative_prompt_2,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    cfg_enabled=cfg_enabled,
                    num_images_per_prompt=num_images_per_prompt,
                    timer=timer,
                    timer_iteration=timer_iteration,
                )
                _, prompt_sequence_length, _ = prompt_embeds.shape

            self.img_to_img = image is not None
            with timer("input_image_preprocessing", timer_iteration) if timer else nullcontext():
                logger.info("preprocessing image prompt...")
                if image is not None and not (
                    isinstance(image, torch.Tensor) and image.size(1) == self._latent_channels
                ):
                    img = image[0] if isinstance(image, list) else image
                    image_height, image_width = self._image_processor.get_default_height_width(img)
                    aspect_ratio = image_width / image_height
                    if _auto_resize:
                        # Kontext is trained on specific resolutions, using one of them is recommended
                        _, image_width, image_height = min(
                            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                        )
                    image_width = image_width // multiple_of * multiple_of
                    image_height = image_height // multiple_of * multiple_of

                    logger.info(f"resizing image to ({image_width}, {image_height})")
                    image = self._image_processor.resize(image, image_height, image_width)
                    image = self._image_processor.preprocess(image, image_height, image_width)

            with timer("preparing_latents", timer_iteration) if timer else nullcontext():
                logger.info("preparing_latents...")
                generator = torch.Generator().manual_seed(seed) if seed is not None else None
                latents, image_latents, latent_ids, image_ids = self.prepare_latents(
                    image,
                    prompt_count * num_images_per_prompt,
                    self._num_channels_latents,
                    self.generation_height,
                    self.generation_width,
                    generator,
                    latents,
                    timer,
                    timer_iteration,
                )

            self.output_latents_shape = latents.shape
            self.output_latents_seq_length = latents.shape[1]
            if image_ids is not None:
                latents = torch.cat([latents, image_latents], dim=1)
                latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension
            spatial_seq_length = latents.shape[1]

            logger.info("preparing timesteps...")
            self._scheduler.set_timesteps(
                sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps),
                mu=_calculate_shift(
                    self.output_latents_seq_length,
                    self._scheduler.config.get("base_image_seq_len", 256),
                    self._scheduler.config.get("max_image_seq_len", 4096),
                    self._scheduler.config.get("base_shift", 0.5),
                    self._scheduler.config.get("max_shift", 1.15),
                ),
            )

            guidance = (
                torch.full([prompt_count * num_images_per_prompt], fill_value=guidance_scale)
                if self._with_guidance_embeds
                else None
            )
            # Add guidance value for negative promts as well
            if cfg_enabled:
                guidance = torch.concat([guidance, guidance])

            ids = torch.cat((text_ids, latent_ids), dim=0)
            rope_cos, rope_sin = self._pos_embed.forward(ids)

            tt_prompt_embeds_list = []
            tt_pooled_prompt_embeds_list = []
            tt_latents_step_list = []
            tt_guidance_list = []
            tt_spatial_rope_cos_list = []
            tt_spatial_rope_sin_list = []
            tt_prompt_rope_cos_list = []
            tt_prompt_rope_sin_list = []
            for i, submesh_device in enumerate(self._submesh_devices):
                tt_prompt_embeds = ttnn.from_torch(
                    prompt_embeds[i].unsqueeze(0)
                    if ((self._parallel_config.cfg_parallel.factor > 1) and cfg_enabled)
                    else prompt_embeds,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        submesh_device,
                        tuple(submesh_device.shape),
                        dims=(None, None),
                    ),
                )

                tt_pooled_prompt_embeds = ttnn.from_torch(
                    pooled_prompt_embeds[i].unsqueeze(0)
                    if ((self._parallel_config.cfg_parallel.factor > 1) and cfg_enabled)
                    else pooled_prompt_embeds,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        submesh_device,
                        tuple(submesh_device.shape),
                        dims=(None, None),
                    ),
                )

                shard_latents_dims = [None, None]
                shard_latents_dims[self.sp_axis] = 1  # height of latents
                tt_initial_latents = ttnn.from_torch(
                    latents,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        submesh_device,
                        tuple(submesh_device.shape),
                        dims=tuple(shard_latents_dims),
                    ),
                )

                if guidance is not None:
                    if (self._parallel_config.cfg_parallel.factor > 1) and cfg_enabled:
                        guidance_tensor = guidance[
                            i * prompt_count * num_images_per_prompt : (i + 1) * prompt_count * num_images_per_prompt
                        ].unsqueeze(-1)
                    else:
                        guidance_tensor = guidance.unsqueeze(-1)
                    tt_guidance = ttnn.from_torch(
                        guidance_tensor,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                        device=submesh_device if not traced else None,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh_device),
                    )
                else:
                    tt_guidance = None

                shard_rope_dims = [None, None]
                shard_rope_dims[self.sp_axis] = 0
                rope_mesh_mapper = ttnn.ShardTensor2dMesh(
                    submesh_device,
                    tuple(submesh_device.shape),
                    dims=tuple(shard_rope_dims),
                )

                tt_spatial_rope_cos = ttnn.from_torch(
                    rope_cos[prompt_sequence_length:],
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=rope_mesh_mapper,
                )
                tt_spatial_rope_sin = ttnn.from_torch(
                    rope_sin[prompt_sequence_length:],
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=rope_mesh_mapper,
                )
                tt_prompt_rope_cos = ttnn.from_torch(
                    rope_cos[:prompt_sequence_length],
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh_device),
                )
                tt_prompt_rope_sin = ttnn.from_torch(
                    rope_sin[:prompt_sequence_length],
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh_device),
                )

                if traced:
                    if self._traces is None:
                        tt_initial_latents = tt_initial_latents.to(submesh_device)
                        tt_prompt_embeds = tt_prompt_embeds.to(submesh_device)
                        tt_pooled_prompt_embeds = tt_pooled_prompt_embeds.to(submesh_device)
                        tt_spatial_rope_cos = tt_spatial_rope_cos.to(submesh_device)
                        tt_spatial_rope_sin = tt_spatial_rope_sin.to(submesh_device)
                        tt_prompt_rope_cos = tt_prompt_rope_cos.to(submesh_device)
                        tt_prompt_rope_sin = tt_prompt_rope_sin.to(submesh_device)

                        if tt_guidance is not None:
                            tt_guidance = tt_guidance.to(submesh_device)
                    else:
                        ttnn.copy_host_to_device_tensor(tt_initial_latents, self._traces[i].spatial_input)
                        ttnn.copy_host_to_device_tensor(tt_prompt_embeds, self._traces[i].prompt_input)
                        ttnn.copy_host_to_device_tensor(tt_pooled_prompt_embeds, self._traces[i].pooled_input)
                        ttnn.copy_host_to_device_tensor(tt_spatial_rope_cos, self._traces[i].spatial_rope_cos)
                        ttnn.copy_host_to_device_tensor(tt_spatial_rope_sin, self._traces[i].spatial_rope_sin)
                        ttnn.copy_host_to_device_tensor(tt_prompt_rope_cos, self._traces[i].prompt_rope_cos)
                        ttnn.copy_host_to_device_tensor(tt_prompt_rope_sin, self._traces[i].prompt_rope_sin)

                        tt_initial_latents = self._traces[i].spatial_input
                        tt_prompt_embeds = self._traces[i].prompt_input
                        tt_pooled_prompt_embeds = self._traces[i].pooled_input
                        tt_spatial_rope_cos = self._traces[i].spatial_rope_cos
                        tt_spatial_rope_sin = self._traces[i].spatial_rope_sin
                        tt_prompt_rope_cos = self._traces[i].prompt_rope_cos
                        tt_prompt_rope_sin = self._traces[i].prompt_rope_sin

                        if tt_guidance is not None:
                            ttnn.copy_host_to_device_tensor(tt_guidance, self._traces[i].guidance_input)
                            tt_guidance = self._traces[i].guidance_input

                tt_prompt_embeds_list.append(tt_prompt_embeds)
                tt_pooled_prompt_embeds_list.append(tt_pooled_prompt_embeds)
                tt_latents_step_list.append(tt_initial_latents)
                tt_guidance_list.append(tt_guidance)
                tt_spatial_rope_cos_list.append(tt_spatial_rope_cos)
                tt_spatial_rope_sin_list.append(tt_spatial_rope_sin)
                tt_prompt_rope_cos_list.append(tt_prompt_rope_cos)
                tt_prompt_rope_sin_list.append(tt_prompt_rope_sin)

            logger.info("denoising...")
            with timer("denoising", timer_iteration) if timer else nullcontext():
                for i, t in enumerate(tqdm.tqdm(self._scheduler.timesteps)):
                    with timer(f"denoising_step_{i}", timer_iteration) if timer else nullcontext():
                        sigma_difference = self._scheduler.sigmas[i + 1] - self._scheduler.sigmas[i]

                        tt_timestep_list = []
                        tt_sigma_difference_list = []
                        for i, submesh_device in enumerate(self._submesh_devices):
                            tt_timestep = ttnn.full(
                                [tt_pooled_prompt_embeds_list[i].shape[0], 1],
                                fill_value=t,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.float32,
                                device=submesh_device if not traced else None,
                            )
                            tt_timestep_list.append(tt_timestep)

                            tt_sigma_difference = ttnn.full(
                                self.output_latents_shape,
                                fill_value=sigma_difference,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16,
                                device=submesh_device
                                if not traced
                                else None,  # Not used in trace region, can be on device always.
                            )
                            tt_sigma_difference_list.append(tt_sigma_difference)

                        tt_latents_step_list = self._step(
                            timestep=tt_timestep_list,
                            latents=tt_latents_step_list,
                            cfg_enabled=cfg_enabled,
                            prompt_embeds=tt_prompt_embeds_list,
                            pooled_prompt_embeds=tt_pooled_prompt_embeds_list,
                            cfg_scale=cfg_scale,
                            sigma_difference=tt_sigma_difference_list,
                            guidance=tt_guidance_list,
                            spatial_rope_cos=tt_spatial_rope_cos_list,
                            spatial_rope_sin=tt_spatial_rope_sin_list,
                            prompt_rope_cos=tt_prompt_rope_cos_list,
                            prompt_rope_sin=tt_prompt_rope_sin_list,
                            spatial_sequence_length=spatial_seq_length,
                            prompt_sequence_length=prompt_sequence_length,
                            traced=traced,
                        )

            logger.info("decoding image...")

            if output_type == "latent":
                ttnn.synchronize_device(self.vae_device)
                tt_latents = self._ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
                    tt_latents,
                    dim=1,
                    mesh_axis=self.sp_axis,
                    use_hyperparams=True,
                )

                torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
                # We need to slice the output latents since TTNN persistant buffer is created for combined latents
                if self.img_to_img:
                    torch_latents = torch_latents[:, : self.output_latents_seq_length]
                image = torch_latents
            else:
                with timer("vae_decoding", timer_iteration) if timer else nullcontext():
                    decoded_output = self._vae_decode(
                        tt_latents_step_list[self.vae_submesh_idx], self.generation_width, self.generation_height
                    )
                    image = self._image_processor.postprocess(decoded_output, output_type=output_type)

        return image

    def synchronize_devices(self) -> None:
        for device in self._submesh_devices:
            ttnn.synchronize_device(device)

    def _step_inner(
        self,
        *,
        cfg_enabled: bool,
        latent: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled: ttnn.Tensor,
        timestep: ttnn.Tensor,
        guidance: ttnn.Tensor | None,
        spatial_rope_cos: ttnn.Tensor,
        spatial_rope_sin: ttnn.Tensor,
        prompt_rope_cos: ttnn.Tensor,
        prompt_rope_sin: ttnn.Tensor,
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        submesh_index: int,
    ) -> ttnn.Tensor:
        if cfg_enabled and not self._parallel_config.cfg_parallel.factor > 1:
            latents_model_input = ttnn.concat([latent, latent])
        else:
            latents_model_input = latent
        noise_pred = self.transformers[submesh_index].forward(
            spatial=latents_model_input,
            prompt=prompt,
            pooled=pooled,
            timestep=timestep,
            guidance=guidance,
            spatial_rope=(spatial_rope_cos, spatial_rope_sin),
            prompt_rope=(prompt_rope_cos, prompt_rope_sin),
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )

        # HACK: Gathering across sp_axis since random_latents and image_latents are concatenated across dim 1
        if tuple(self._submesh_devices[submesh_index].shape)[0] > 1:
            # Collects shards from all 8 devices along mesh_axis=0 in 2x4 mesh
            noise_pred = ttnn.all_gather(
                noise_pred,
                dim=1,
                cluster_axis=self._parallel_config.sequence_parallel.mesh_axis,  # axis=0 for sequence parallel
            )

        if self.img_to_img:
            noise_pred = noise_pred[:, : self.output_latents_seq_length]
        return noise_pred

    def _step(
        self,
        *,
        cfg_enabled: bool,
        cfg_scale: float,
        latents: list[ttnn.Tensor],  # device tensor\
        timestep: list[ttnn.Tensor],  # host tensor
        pooled_prompt_embeds: list[ttnn.Tensor],  # device tensor
        prompt_embeds: list[ttnn.Tensor],  # device tensor
        sigma_difference: list[ttnn.Tensor],  # device tensor
        guidance: list[ttnn.Tensor | None],
        spatial_rope_cos: list[ttnn.Tensor],
        spatial_rope_sin: list[ttnn.Tensor],
        prompt_rope_cos: list[ttnn.Tensor],
        prompt_rope_sin: list[ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        traced: bool,
    ) -> list[ttnn.Tensor]:
        if traced and self._traces is None:
            self._traces = []
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                logger.info(f"Tracing for Device ID : {submesh_id}...")
                latent_device = latents[submesh_id]  # already on device
                prompt_device = prompt_embeds[submesh_id]  # already on device
                pooled_projection_device = pooled_prompt_embeds[submesh_id]  # already on device
                timestep_device = timestep[submesh_id].to(submesh_device)
                sigma_difference_device = sigma_difference[submesh_id].to(submesh_device)

                logger.info("Compile run for tracing...")
                pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latent_device,
                    prompt=prompt_device,
                    pooled=pooled_projection_device,
                    timestep=timestep_device,
                    guidance=guidance[submesh_id],
                    spatial_rope_cos=spatial_rope_cos[submesh_id],
                    spatial_rope_sin=spatial_rope_sin[submesh_id],
                    prompt_rope_cos=prompt_rope_cos[submesh_id],
                    prompt_rope_sin=prompt_rope_sin[submesh_id],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
                    submesh_index=submesh_id,
                )

                if submesh_id == self.vae_submesh_idx:
                    logger.info("Initializing VAE buffers for safe tracing...")
                    self._vae_decode(latent_device, self.generation_width, self.generation_height)

                logger.info("Capturing trace...")
                ttnn.synchronize_device(submesh_device)
                trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
                pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latent_device,
                    prompt=prompt_device,
                    pooled=pooled_projection_device,
                    timestep=timestep_device,
                    guidance=guidance[submesh_id],
                    spatial_rope_cos=spatial_rope_cos[submesh_id],
                    spatial_rope_sin=spatial_rope_sin[submesh_id],
                    prompt_rope_cos=prompt_rope_cos[submesh_id],
                    prompt_rope_sin=prompt_rope_sin[submesh_id],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
                    submesh_index=submesh_id,
                )
                ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)
                ttnn.synchronize_device(submesh_device)
                logger.info("Trace captured sucessfully...")

                self._traces.append(
                    PipelineTrace(
                        spatial_input=latents[submesh_id],
                        prompt_input=prompt_embeds[submesh_id],
                        pooled_input=pooled_prompt_embeds[submesh_id],
                        timestep_input=timestep_device,
                        guidance_input=guidance[submesh_id],
                        latents_output=pred,
                        spatial_rope_cos=spatial_rope_cos[submesh_id],
                        spatial_rope_sin=spatial_rope_sin[submesh_id],
                        prompt_rope_cos=prompt_rope_cos[submesh_id],
                        prompt_rope_sin=prompt_rope_sin[submesh_id],
                        sigma_difference_input=sigma_difference_device,
                        tid=trace_id,
                    )
                )

        noise_pred_list = []
        sigma_difference_device_list = []
        if traced:
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                ttnn.copy_host_to_device_tensor(timestep[submesh_id], self._traces[submesh_id].timestep_input)
                ttnn.copy_host_to_device_tensor(
                    sigma_difference[submesh_id], self._traces[submesh_id].sigma_difference_input
                )
                sigma_difference_device_list.append(self._traces[submesh_id].sigma_difference_input)
                ttnn.execute_trace(submesh_device, self._traces[submesh_id].tid, cq_id=0, blocking=False)
                noise_pred_list.append(self._traces[submesh_id].latents_output)
        else:
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                noise_pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    pooled=pooled_prompt_embeds[submesh_id],
                    timestep=timestep[submesh_id],
                    guidance=guidance[submesh_id],
                    spatial_rope_cos=spatial_rope_cos[submesh_id],
                    spatial_rope_sin=spatial_rope_sin[submesh_id],
                    prompt_rope_cos=prompt_rope_cos[submesh_id],
                    prompt_rope_sin=prompt_rope_sin[submesh_id],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
                    submesh_index=submesh_id,
                )
                noise_pred_list.append(noise_pred)
                sigma_difference_device_list.append(sigma_difference[submesh_id])

        if cfg_enabled:
            if not self._parallel_config.cfg_parallel.factor > 1:
                split_pos = noise_pred_list[0].shape[0] // 2
                uncond = noise_pred_list[0][0:split_pos]
                cond = noise_pred_list[0][split_pos:]
                noise_pred_list[0] = uncond + cfg_scale * (cond - uncond)
            else:
                # uncond and cond are replicated, so it is fine to get a single tensor from each
                uncond = ttnn.to_torch(ttnn.get_device_tensors(noise_pred_list[0])[0].cpu(blocking=True)).to(
                    torch.float32
                )
                cond = ttnn.to_torch(ttnn.get_device_tensors(noise_pred_list[1])[0].cpu(blocking=True)).to(
                    torch.float32
                )

                torch_noise_pred = uncond + cfg_scale * (cond - uncond)

                shard_latents_dims = [None, None]
                shard_latents_dims[self._parallel_config.sequence_parallel.mesh_axis] = 1  # height of latents
                noise_pred_list[0] = ttnn.from_torch(
                    torch_noise_pred,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=self._submesh_devices[0],
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self._submesh_devices[0],
                        tuple(self._submesh_devices[0].shape),
                        dims=tuple(shard_latents_dims),
                    ),
                )

                noise_pred_list[1] = ttnn.from_torch(
                    torch_noise_pred,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=self._submesh_devices[1],
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self._submesh_devices[1],
                        tuple(self._submesh_devices[1].shape),
                        dims=tuple(shard_latents_dims),
                    ),
                )

        for submesh_id, submesh_device in enumerate(self._submesh_devices):
            ttnn.synchronize_device(submesh_device)  # Helps with accurate time profiling.
            sigma_difference_device = sigma_difference_device_list[submesh_id]
            ttnn.multiply_(sigma_difference_device, noise_pred_list[submesh_id])

            # HACK: Gathering across sp_axis since random_latents and image_latents are concatenated across dim 1
            if tuple(submesh_device.shape)[0] > 1:
                # Collects shards from all 8 devices along mesh_axis=0 in 2x4 mesh
                ttnn.synchronize_device(submesh_device)  # Helps with accurate time profiling.
                latents[submesh_id] = ttnn.all_gather(
                    latents[submesh_id],
                    dim=1,
                    cluster_axis=self._parallel_config.sequence_parallel.mesh_axis,  # axis=0 for sequence parallel
                    topology=self._ccl_managers[submesh_id].topology,
                )

            if self.img_to_img:
                randn_latents = latents[submesh_id][:, : self.output_latents_seq_length]
                spatial_latents = latents[submesh_id][:, self.output_latents_seq_length :]
                ttnn.add_(randn_latents, sigma_difference_device)
                latents[submesh_id] = ttnn.concat([randn_latents, spatial_latents], dim=1)
            else:
                ttnn.add_(latents[submesh_id], sigma_difference_device)

            # HACK: Resharding to match the inital conditions, funtionality needs to be checked for sub_meshing and sp_axis dim > 1
            # Undo the gather operation
            if tuple(submesh_device.shape)[0] > 1:
                latents[submesh_id] = ttnn.mesh_partition(
                    latents[submesh_id],
                    dim=1,
                    cluster_axis=self._parallel_config.sequence_parallel.mesh_axis,
                )

            # Copy the updated latents tensor to presistant buffer
            if traced:
                ttnn.copy(latents[submesh_id], self._traces[submesh_id].spatial_input)
                latents[submesh_id] = self._traces[submesh_id].spatial_input

        return latents
