import ttnn
import torch
from loguru import logger
from models.experimental.granite_speech_33_8b.tt.ttnn_encoder_block import GraniteSpeechCTCEncoderTTNN
from models.experimental.granite_speech_33_8b.tt.ttnn_projector_block import GraniteSpeechEncoderProjectorTTNN

from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator

class GraniteEncoderAndProjector:
    """TTNN implementation of Encoder+Projector."""

    def __init__(self, device, config, include_conformer_layernorm=False, use_optimized_attention=True):
        self.device = device
        self.encoder = GraniteSpeechCTCEncoderTTNN(device, config.encoder_config, include_conformer_layernorm)
        self.projector = GraniteSpeechEncoderProjectorTTNN(device, config, use_optimized_attention)

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration for high accuracy."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,  # High fidelity for 0.99 PCC
            math_approx_mode=False,
            fp32_dest_acc_en=True,  # Enable FP32 accumulation for accuracy
            packer_l1_acc=False,
        )

    def prepare_weights(
        self, model
    ):
        """Load and convert PyTorch weights to TTNN format."""
        self.encoder.prepare_weights(model.encoder)
        self.projector.prepare_weights(model.projector)

    def forward(self, input_features):
        """Get the audio features to merged into the multimodal embeddings."""
        encoder_embeds = self.encoder.forward(input_features)
        projected_embeds = self.projector.forward(encoder_embeds)

        return projected_embeds


def sample_host(tt_input, temperature=0.6, top_p=0.08, on_host=True):
    vocab_size = tt_input.shape[-1]
    pt_input = tt_input[..., :vocab_size]

    if temperature > 0:
        probs = torch.softmax(pt_input / temperature, dim=-1)
        pt_out = sample_top_p(probs.squeeze(), top_p)
    else:
        pt_out = torch.argmax(pt_input, dim=-1)

    if pt_out.dim() == 1:  # if sampling a single token re-add the batch dim to the tensor
        pt_out = pt_out.unsqueeze(0)
    return None, pt_out


class GraniteSpeech:
    """TTNN implementation of GraniteSpeech."""

    def __init__(self, device, config, tokenizer=None, torch_ref=None, use_torch_audio_feat=True):
        self.device = device
        self.config = config
        self.torch_ref = torch_ref
        self.tokenizer = tokenizer
        self.use_torch_audio_feat = use_torch_audio_feat
        self.encoder = GraniteSpeechCTCEncoderTTNN(device, config, include_conformer_layernorm=False)
        self.projector = GraniteSpeechEncoderProjectorTTNN(device, config)

        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.text_config_hidden_size, config.pad_token_id)

        if self.use_torch_audio_feat:
            self.encoder = self.torch_ref.encoder
            self.projector = self.torch_ref.projector
        else:
            self.encoder = self.encoder
            self.projector = self.projector

        paged_attention_config = (
            PagedAttentionConfig(
                block_size=32,#page_params["page_block_size"],
                max_num_blocks=1024,#page_params["page_max_num_blocks_per_dp"],
            )
        )

        self.model_args, self.tt_model, self.tt_kv_cache, self.state_dict = create_tt_model(
            self.device,
            instruct=True,
            max_batch_size=1,
            optimizations=None,
            max_seq_len=4*1024,
            paged_attention_config=paged_attention_config,
        )

        self.generator = Generator([self.tt_model], [self.model_args], self.device, processor=self.model_args.processor, tokenizer=self.model_args.tokenizer)

        self._setup_compute_config()

    def _setup_compute_config(self):
        """Setup compute kernel configuration for high accuracy."""
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,  # High fidelity for 0.99 PCC
            math_approx_mode=False,
            fp32_dest_acc_en=True,  # Enable FP32 accumulation for accuracy
            packer_l1_acc=False,
        )

    def prepare_weights(
        self, model
    ):
        if not self.use_torch_audio_feat:
            """Load and convert PyTorch weights to TTNN format."""
            self.encoder.prepare_weights(model.encoder.state_dict())
            self.projector.prepare_weights(model.projector)

    def forward(self, input_ids, input_features, input_features_mask):

        """Get the audio features to merged into the multimodal embeddings."""
        encoder_embeds = self.encoder.forward(input_features)
        audio_features = self.projector.forward(encoder_embeds)

        is_audio_index = input_ids == self.config.audio_token_id
        llm_input_ids = torch.where(is_audio_index, 0, input_ids)
        inputs_embeds = self.torch_ref.language_model.get_input_embeddings()(llm_input_ids)  # [bsz, # features, hidden size]

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
            prompt_lens=prompt_lens,
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
        global_batch_size = 1
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
                    temperature=0,#sampling_params["temperature"],
                    top_p=0.08,#sampling_params["top_p"],
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
                text = "".join(tokenizer.decode(all_outputs[user]))
                if len(text) > 100:
                    text = "..." + text[-97:]
                text = text.replace("\n", " ")
                logger.debug("[User {}] {}".format(user, text))

            iteration += 1

            # Upper limit of generated tokens for each user
            if iteration >= max_generated_tokens:
                users_decoding = False
