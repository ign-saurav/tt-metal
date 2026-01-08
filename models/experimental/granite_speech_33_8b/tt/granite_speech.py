import ttnn
import torch
from loguru import logger

from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
    sample_host,
)
from models.tt_transformers.tt.model_config import DecodersPrecision
from models.experimental.granite_speech_33_8b.tt.generator import Generator


def create_tt_page_table(global_batch_size, data_parallel, paged_attention_config: PagedAttentionConfig):
    page_table = None

    if paged_attention_config:
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
        page_table = reverse_permutation.reshape(
            global_batch_size, paged_attention_config.max_num_blocks // (global_batch_size // data_parallel)
        )

    return page_table


def create_tt_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    paged_attention_config: PagedAttentionConfig = None,
    dtype=ttnn.bfloat8_b,
    state_dict=None,
    num_layers=None,
):
    # from models.tt_transformers.tt.model import Transformer
    from models.experimental.granite_speech_33_8b.tt.granite_transformer import GraniteSpeech
    from models.tt_transformers.tt.model_config import ModelArgs

    tt_model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    if num_layers is not None:
        tt_model_args.n_layers = num_layers

    # Avoid loading state_dict for every DP model
    if not state_dict:
        state_dict = tt_model_args.load_state_dict()

    model = GraniteSpeech(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    tt_kv_cache = [l.attention.layer_past for l in model.layers] if paged_attention_config else None

    return tt_model_args, model, tt_kv_cache, state_dict


class GraniteSpeech:
    """TTNN implementation of GraniteSpeech."""

    def __init__(self, device, config, tokenizer=None, torch_ref=None, use_torch_audio_feat=True):
        self.device = device
        self.config = config
        self.torch_ref = torch_ref
        self.tokenizer = tokenizer
        self.use_torch_audio_feat = use_torch_audio_feat

        if self.use_torch_audio_feat:
            self.encoder = self.torch_ref.encoder
            self.projector = self.torch_ref.projector
        else:
            self.encoder = self.encoder
            self.projector = self.projector

        paged_attention_config = PagedAttentionConfig(
            block_size=32,  # page_params["page_block_size"],
            max_num_blocks=256,  # page_params["page_max_num_blocks_per_dp"],
        )

        self.page_table = create_tt_page_table(
            global_batch_size=1,
            data_parallel=1,
            paged_attention_config=paged_attention_config,
        )

        self.model_args, self.tt_model, self.tt_kv_cache, self.state_dict = create_tt_model(
            self.device,
            instruct=True,
            max_batch_size=1,
            optimizations=lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
            max_seq_len=512,
            paged_attention_config=paged_attention_config,
        )

        self.tokenizer = self.model_args.tokenizer
        self.generator = Generator(
            [self.tt_model],
            [self.model_args],
            self.device,
            processor=self.model_args.processor,
            tokenizer=self.model_args.tokenizer,
        )

    def forward(self, input_ids, input_features, input_features_mask):
        global_batch_size = 1
        max_generated_tokens = 200

        """Get the audio features to merged into the multimodal embeddings."""
        encoder_embeds = self.encoder(input_features)
        audio_features = self.projector(encoder_embeds)

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
        prompt_lens = inputs_embeds.shape[1]

        # Run TT model
        logger.info(f"Running TT model...")
        tt_output_torch = self.generator.prefill_forward_text(
            inputs_embeds,
            page_table=self.page_table,
            kv_cache=[self.tt_kv_cache],
            prompt_lens=[prompt_lens],
            enable_trace=False,
        )
        logger.info(f"Finished running TT model.")
        prefilled_token = torch.argmax(tt_output_torch, dim=-1)

        # Initial positions
        current_pos = torch.tensor([prompt_lens])

        # Start decoding
        iteration = 0
        users_decoding = True
        device_sampling_params = None
        stress_test = False
        user_done = [False] * global_batch_size  # Keeps track when a user reaches EoD token
        all_outputs = []
        stop_at_eos = True

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
                # prompt_tokens=input_tokens_prefill_pt,
                output_tokens=out_tok,
            )

            # Get the next token
            if device_sampling_params is not None:
                out_tok = logits.unsqueeze(1)

            else:
                # TODO Fix use case with temperature > 0
                _, out_tok = sample_host(
                    logits,
                    temperature=0,  # sampling_params["temperature"],
                    top_p=0.08,  # sampling_params["top_p"],
                    on_host=True,
                )

            if not stress_test:  # During stress test runs we will iterate over the same position for X iterations
                current_pos += 1
            # Save output token to print out later
            for user in range(global_batch_size):
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
            for user in range(global_batch_size):
                text = "".join(self.tokenizer.decode(all_outputs[user]))
                if len(text) > 100:
                    text = "..." + text[-97:]
                text = text.replace("\n", " ")
                logger.debug("[User {}] {}".format(user, text))

            iteration += 1

            # Upper limit of generated tokens for each user
            if iteration >= max_generated_tokens:
                users_decoding = False
