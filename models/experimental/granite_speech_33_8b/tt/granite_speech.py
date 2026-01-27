import ttnn
import torch
from loguru import logger

from models.tt_transformers.tt.common import (
    sample_host,
)
from models.tt_transformers.tt.model_config import DecodersPrecision
from models.experimental.granite_speech_33_8b.tt.generator import (
    Generator,
    prepare_generator_args,
)
from typing import List
from models.experimental.granite_speech_33_8b.tt.ttnn_encoder_block import GraniteSpeechCTCEncoderTTNN
from models.experimental.granite_speech_33_8b.tt.ttnn_projector_block import GraniteSpeechEncoderProjectorTTNN
from models.tt_transformers.tt.generator import SamplingParams


class GraniteSpeechTTNN:
    """TTNN implementation of GraniteSpeech."""

    def __init__(
        self,
        mesh_device,
        config,
        tokenizer=None,
        torch_ref=None,
        use_torch_audio_feat=False,
        include_conformer_layernorm=True,
        use_optimized_attention_projector=True,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.torch_ref = torch_ref
        self.tokenizer = tokenizer
        self.use_torch_audio_feat = use_torch_audio_feat

        if self.use_torch_audio_feat:
            self.encoder = self.torch_ref.encoder
            self.projector = self.torch_ref.projector
        else:
            self.encoder = GraniteSpeechCTCEncoderTTNN(mesh_device, config.encoder_config, include_conformer_layernorm)
            self.projector = GraniteSpeechEncoderProjectorTTNN(mesh_device, config, use_optimized_attention_projector)
            self.encoder.prepare_weights(self.torch_ref.encoder)
            self.projector.prepare_weights(self.torch_ref.projector)
            self.encoder = self.encoder.forward
            self.projector = self.projector.forward

        self.batch_size = 1
        self.data_parallel = 1
        self.repeat_batches = 1
        self.max_seq_len = 1024
        self.max_generated_tokens = 200
        self.num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
        self.global_batch_size = (
            self.batch_size * self.data_parallel
        )  # input batch_size is interpreted as size per DP group
        optimisations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)
        page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 256}
        (
            self.model_args,
            self.model,
            self.page_table,
            self.tt_kv_cache,
            self.tokenizer,
            self.processor,
        ) = prepare_generator_args(
            num_devices=self.num_devices,
            data_parallel=self.data_parallel,
            mesh_device=self.mesh_device,
            instruct=True,
            global_batch_size=self.global_batch_size,
            optimizations=optimisations,
            max_seq_len=self.max_seq_len,
            page_params=page_params,
            paged_attention=True,
            num_layers=None,
        )

        self.generator = Generator(
            self.model, self.model_args, self.mesh_device, processor=self.processor, tokenizer=self.tokenizer
        )

        self.paged_cache_max_seq_len = (
            page_params["page_block_size"] * page_params["page_max_num_blocks_per_dp"] / self.batch_size
        )

    def forward(self, input_ids, input_features, input_features_mask):
        if not self.use_torch_audio_feat:
            input_features = ttnn.from_torch(
                input_features, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
            )
        else:
            input_features = input_features.to(torch.bfloat16)

        """Get the audio features to merged into the multimodal embeddings."""
        encoder_embeds = self.encoder(input_features)
        audio_features = self.projector(encoder_embeds)

        if not self.use_torch_audio_feat:
            composer = ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=-1)
            audio_features = ttnn.to_torch(audio_features, mesh_composer=composer)

        is_audio_index = input_ids == self.config.audio_token_id
        llm_input_ids = torch.where(is_audio_index, 0, input_ids)
        inputs_embeds = self.torch_ref.language_model.get_input_embeddings()(
            llm_input_ids
        )  # [bsz, # features, hidden size]

        # Mask the audio features into the text embeddings
        special_audio_mask = is_audio_index.unsqueeze(-1)
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        if input_features_mask is not None:
            if torch.all(is_audio_index.int().sum(dim=1) != input_features_mask.int().sum(dim=1)).item():
                raise ValueError("Number of audio tokens does not match number of audio features")

            audio_features = audio_features[input_features_mask]

        inputs_embeds = inputs_embeds.masked_scatter(
            special_audio_mask,
            audio_features,
        )
        inputs_embeds = inputs_embeds * self.config.text_config.embedding_multiplier  # embedding multiplier

        input_tokens_prefill_pt = inputs_embeds
        decoding_pos = [inputs_embeds.shape[-2]]
        max_encoded_prompt_len = inputs_embeds.shape[-2]
        assert self.max_generated_tokens + max_encoded_prompt_len <= self.max_seq_len

        assert (
            self.max_generated_tokens + max_encoded_prompt_len <= self.paged_cache_max_seq_len
        ), f"max_generated_tokens ({self.max_generated_tokens}) needs to be <= than paged_cache_max_seq_len ({self.paged_cache_max_seq_len})"

        # logger.info("Starting prefill warmup...")
        # logits = self.generator.prefill_forward_text(
        #     input_tokens_prefill_pt,  # Prefill warmup for all users, in case some users have different seqlens than others
        #     page_table=self.page_table,
        #     kv_cache=self.tt_kv_cache,
        #     prompt_lens=decoding_pos,
        #     enable_trace=False,
        # )
        # logger.info("Finished prefill warmup")

        logger.info(f"Starting prefill...")
        logits = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=decoding_pos,
            enable_trace=False,
        )
        prefilled_token = torch.argmax(logits, dim=-1)
        logger.info(f"Prefill finished")

        # Keep track of generated outputs to print out every iteration
        all_outputs = [[]] * self.global_batch_size
        for user in range(self.global_batch_size):
            user_tok = int(prefilled_token[user].item())
            all_outputs[user].append(user_tok)

        user_done = [False] * self.global_batch_size  # Keeps track when a user reaches EoD token

        sampling_params = {"temperature": 0, "top_p": 0.08, "top_k": 32}
        device_sampling_params = (
            SamplingParams(
                temperature=sampling_params["temperature"],
                top_k=sampling_params["top_k"],
                top_p=sampling_params["top_p"],
                frequency_penalty=(
                    sampling_params["frequency_penalty"] if "frequency_penalty" in sampling_params else 0.0
                ),
                presence_penalty=sampling_params["presence_penalty"] if "presence_penalty" in sampling_params else 0.0,
                repetition_penalty=(
                    sampling_params["repetition_penalty"] if "repetition_penalty" in sampling_params else 1.0
                ),
            )
            if self.model[0]._supports_on_device_sampling
            else None
        )
        if device_sampling_params is None and isinstance(sampling_params["temperature"], List):
            # host sampling only supports single sample param for all users in a batch
            sampling_params["temperature"] = sampling_params["temperature"][0]
            sampling_params["top_p"] = sampling_params["top_p"][0]

        # Initial positions
        current_pos = torch.tensor([decoding_pos[b] for b in range(self.global_batch_size)])

        user_done = [False] * self.global_batch_size  # Keeps track when a user reaches EoD token
        stop_at_eos = True

        # Start decoding
        iteration = 0
        users_decoding = True

        out_tok = prefilled_token

        logger.info(f"Starting decode loop...")

        while users_decoding:
            # Run decode forward
            logits, log_probs = self.generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=False,
                page_table=self.page_table,
                kv_cache=self.tt_kv_cache,
                sampling_params=device_sampling_params,
                prompt_tokens=input_tokens_prefill_pt,
                output_tokens=out_tok,
            )

            # Get the next token
            if device_sampling_params is not None:
                out_tok = logits.unsqueeze(1)

            else:
                # TODO Fix use case with temperature > 0
                _, out_tok = sample_host(
                    logits,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    on_host=True,
                )

            current_pos += 1

            # Save output token to print out later
            for user in range(self.global_batch_size):
                user_tok = out_tok[user].item()
                if (
                    user_tok not in self.tokenizer.stop_tokens and user_done[user] == False
                ):  # Read until an eos token (e.g. <|eot_id|>); create_tokenizer adds stop_tokens to HF tokenizers
                    all_outputs[user].append(user_tok)
                else:
                    if (
                        stop_at_eos
                    ):  # For performance gathering in CI, we want to sometimes force decoding for a fixed number of iterations
                        user_done[user] = True
                        logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                        if all(user_done):
                            users_decoding = False

            # Print out generated outputs for each user at the end of every iteration
            for user in range(self.global_batch_size):
                text = "".join(self.tokenizer.decode(all_outputs[user]))
                if len(text) > 100:
                    text = "..." + text[-97:]
                text = text.replace("\n", " ")
                logger.debug("[User {}] {}".format(user, text))

            iteration += 1

            # Upper limit of generated tokens for each user
            if iteration >= self.max_generated_tokens:
                users_decoding = False

        # Final print
        if not users_decoding:
            logger.info("Finished decoding, printing the final outputs...\n")
            for i, output in enumerate(all_outputs):
                text = self.tokenizer.decode(output)
                logger.info(f"\nSTT OUTPUT: {text}\n")
