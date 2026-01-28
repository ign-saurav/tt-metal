import ttnn
import torch
import os
from loguru import logger

from models.tt_transformers.tt.common import (
    sample_host,
)
from models.tt_transformers.tt.model_config import DecodersPrecision, determine_device_name
from models.experimental.granite_speech_33_8b.tt.generator import (
    Generator,
    prepare_generator_args,
)
from typing import List
from models.experimental.granite_speech_33_8b.tt.ttnn_encoder_block import GraniteSpeechCTCEncoderTTNN
from models.experimental.granite_speech_33_8b.tt.ttnn_projector_block import GraniteSpeechEncoderProjectorTTNN
from models.experimental.granite_speech_33_8b.tt.utils import save_language_model_weights
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.demos.utils.llm_demo_utils import create_benchmark_data


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

        # Check if language_model weights exist, if not save them
        weights_dir = "granite_instruct_weights_from_speech"
        if not os.path.exists(weights_dir):
            logger.info(f"Weights directory '{weights_dir}' not found. Saving language_model weights...")
            if self.torch_ref is not None and hasattr(self.torch_ref, "language_model"):
                self.torch_ref.language_model.save_pretrained(weights_dir)
                self.tokenizer.save_pretrained(weights_dir)
                logger.info(f"Successfully saved language_model weights to '{weights_dir}'")
            else:
                logger.warning(
                    f"torch_ref not available or missing language_model. Calling save_language_model_weights()..."
                )
                save_language_model_weights(weights_dir)

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
        # Start profiler
        logger.info(f"Start profiler")
        profiler = BenchmarkProfiler()
        profiler.start("run")

        """Get the audio features to merged into the multimodal embeddings."""
        profiler.start("audio_features_time")
        encoder_embeds = self.encoder(input_features)
        audio_features = self.projector(encoder_embeds)
        profiler.end("audio_features_time")

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

        logger.info("Starting prefill warmup...")
        profiler.start(f"compile_prefill", iteration=0)
        logits = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,  # Prefill warmup for all users, in case some users have different seqlens than others
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=decoding_pos,
            enable_trace=True,
        )
        profiler.end(f"compile_prefill", iteration=0)
        logger.info("Finished prefill warmup")

        logger.info(f"Starting prefill...")
        profiler.start(f"inference_prefill", iteration=0)
        logits = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=decoding_pos,
            enable_trace=True,
        )
        prefilled_token = torch.argmax(logits, dim=-1)
        profiler.end(f"inference_prefill", iteration=0)
        logger.info(f"Prefill finished")

        # Keep track of generated outputs to print out every iteration
        all_outputs = [[]] * self.global_batch_size
        for user in range(self.global_batch_size):
            user_tok = int(prefilled_token[user].item())
            all_outputs[user].append(user_tok)

        user_done = [False] * self.global_batch_size  # Keeps track when a user reaches EoD token

        sampling_params = {"temperature": 0, "top_p": 0.08, "top_k": 32}
        device_sampling_params = None
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
        num_tokens_generated_decode = []

        out_tok = prefilled_token

        logger.info(f"Starting decode loop...")
        profiler.start(f"inference_decode", iteration=0)

        while users_decoding:
            if iteration == 0:  # First iteration also accounts for compile time
                profiler.start(f"compile_decode", iteration=0)
            else:
                profiler.start(f"inference_decode_time_{iteration}", iteration=0)
            # Run decode forward
            logits, log_probs = self.generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=True,
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
            if iteration == 0:  # First iteration will account the compile time
                profiler.end(f"compile_decode", iteration=0)
                decode_iteration_time = profiler.get_duration("compile_decode", iteration=0)
            else:
                profiler.end(f"inference_decode_time_{iteration}", iteration=0)
                decode_iteration_time = profiler.get_duration(f"inference_decode_time_{iteration}", iteration=0)

            # Print perf after every iteration (skip in CI to avoid performance overhead)
            tokens_per_second_per_user = 1 / decode_iteration_time
            logger.debug(
                f"Iteration {iteration}: {1000 * decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({self.global_batch_size * tokens_per_second_per_user:.1f} tok/s throughput)"
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

        num_tokens_generated_decode.append(iteration)  # Save the number of tokens generated for each repeat batch

        # Final print
        if not users_decoding:
            logger.info("Finished decoding, printing the final outputs...\n")
            for i, output in enumerate(all_outputs):
                text = self.tokenizer.decode(output)
                logger.info(f"\nSTT OUTPUT: {text}\n")

        profiler.end(f"inference_decode", iteration=0)

        # Finish profiling at the end of inference for all repeated batches
        profiler.end("run")

        # Prepare profile benchmark metrics for the first repeat batch only
        compile_prefill_time = profiler.get_duration("compile_prefill")
        compile_decode_time = profiler.get_duration("compile_decode")

        total_inference_prefill_time = profiler.get_duration("inference_prefill")
        audio_features_time = profiler.get_duration("audio_features_time")
        total_inference_decode_time = 0
        for i in range(1, num_tokens_generated_decode[0]):  # Iteration 0 is the compile time
            total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

        # Average prefill time for each user
        avg_time_to_first_token = total_inference_prefill_time / self.global_batch_size

        # Average decode time per batch iteration
        avg_decode_iteration_time = (
            total_inference_decode_time / (num_tokens_generated_decode[0] - 1) if iteration > 1 else 0
        )
        prefill_lens = [input_tokens_prefill_pt.shape[-2]]

        prefill_tok_s = prefill_lens[0] / total_inference_prefill_time * self.global_batch_size
        decode_tok_s_user = (
            (num_tokens_generated_decode[0] - 1) / total_inference_decode_time if iteration > 1 else 0
        )  # Remove the compile time
        decode_tok_s = (
            ((num_tokens_generated_decode[0] - 1) / total_inference_decode_time * self.global_batch_size)
            if iteration > 1
            else 0
        )  # Remove the compile time

        measurements = {
            # Required measurements
            "compile_prefill": compile_prefill_time,
            "compile_decode": compile_decode_time,
            "inference_prefill": total_inference_prefill_time,
            "inference_decode": total_inference_decode_time,
            "audio_features_time": audio_features_time,
            "prefill_time_to_token": avg_time_to_first_token,
            "prefill_t/s": prefill_tok_s,  # tokens/s
            "decode_t/s/u": decode_tok_s_user,  # tokens/s/u
            "decode_t/s": decode_tok_s,  # tokens/s
            # Optional measurements
            "Total compile time": compile_prefill_time + compile_decode_time,
            "Full demo runtime": profiler.get_duration("run"),
        }

        # Decode performance for some specific tokens
        tok_1_perf = (
            profiler.get_duration(f"inference_decode_time_{1}") if 1 < num_tokens_generated_decode[0] else 0
        )  # Iteration 0 is compile time
        tok_128_perf = (
            profiler.get_duration(f"inference_decode_time_{127}") if 127 < num_tokens_generated_decode[0] else 0
        )
        tok_1024_perf = (
            profiler.get_duration(f"inference_decode_time_{1023}") if 1023 < num_tokens_generated_decode[0] else 0
        )
        tok_4096_perf = (
            profiler.get_duration(f"inference_decode_time_{4095}") if 4095 < num_tokens_generated_decode[0] else 0
        )

        if not stop_at_eos:
            logger.info(f"Please note that 'stop_at_eos' is disabled. Output repetition is expected.")

        logger.info("")
        logger.info(f"=== Performance metrics ===")
        if tok_1_perf > 0:
            logger.info(
                f"1st token decode time: {tok_1_perf * 1000:.2f}ms [{round(1 / tok_1_perf, 2)} t/s/u, {round((1 / tok_1_perf) * self.global_batch_size, 2)} t/s]"
            )
        if tok_128_perf > 0:
            logger.info(
                f"128th token decode time: {tok_128_perf * 1000:.2f}ms [{round(1 / tok_128_perf, 2)} t/s/u, {round((1 / tok_128_perf) * self.global_batch_size, 2)} t/s]"
            )
        if tok_1024_perf > 0:
            logger.info(
                f"1024th token decode time: {tok_1024_perf * 1000:.2f}ms [{round(1 / tok_1024_perf, 2)} t/s/u, {round((1 / tok_1024_perf) * self.global_batch_size, 2)} t/s]"
            )

        # Print some of the perf metrics
        logger.info("==")
        logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
        logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
        logger.info("")
        logger.info(f"Audio features time: {round(audio_features_time, 2)}s")
        logger.info(f"Average Time to First Token (TTFT): {round(avg_time_to_first_token * 1000, 2)}ms")
        logger.info(
            f"Average speed: {round(avg_decode_iteration_time * 1000, 2)}ms @ {round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
        )

        # Save benchmark data for CI dashboard
        # Instead of running warmup iterations, the demo profiles the initial compile iteration
        targets = {}
        tt_device_name = determine_device_name(self.mesh_device)  # submesh device should not decide performance target
        model_name = "Granite-speech-3.3-8b"
        bench_n_warmup_iter = {"inference_prefill": 0, "inference_decode": 1}
        benchmark_data = create_benchmark_data(profiler, measurements, bench_n_warmup_iter, targets)

        # Save the decode performance of every iteration for plotting in superset
        for i in range(1, num_tokens_generated_decode[0]):
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                f"time_to_token_{i}",
                profiler.get_duration(f"inference_decode_time_{i}") * 1000,
                step_warm_up_num_iterations=None,
                target=None,
            )

        # Also save the avg decode performance for the 128 iterations (excluding the compile time)
        num_iterations_for_avg = min(128, num_tokens_generated_decode[0])
        inference_decode_time_first_128 = sum(
            profiler.get_duration(f"inference_decode_time_{i}") for i in range(1, num_iterations_for_avg)
        )
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference_decode",
            "avg_decode_time_first_128",
            inference_decode_time_first_128 * 1000 / max(1, num_iterations_for_avg - 1),
            step_warm_up_num_iterations=None,
            target=None,
        )
        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=model_name,
            ml_model_type="speech",
            num_layers=self.model_args[0].n_layers,
            batch_size=self.global_batch_size,
            config_params={
                "data_parallel": self.data_parallel,
                "tensor_parallel": self.num_devices // self.data_parallel,
            },
            input_sequence_length=max(prefill_lens),
            output_sequence_length=num_tokens_generated_decode[0],
        )
