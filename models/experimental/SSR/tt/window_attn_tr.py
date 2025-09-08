import ttnn
import torch
from models.common.lightweightmodule import LightweightModule


class TTWindowAttentionTR(LightweightModule):
    def __init__(self, device, parameters, dim, window_size, num_heads, memory_config=None):
        super().__init__()
        self.device = device
        self.memory_config = memory_config or ttnn.L1_MEMORY_CONFIG
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Extract preprocessed parameters
        self.qkv_weight = parameters["qkv"]["weight"]
        self.qkv_bias = parameters["qkv"]["bias"] if "bias" in parameters["qkv"] else None
        self.proj_weight = parameters["proj"]["weight"]
        self.proj_bias = parameters["proj"]["bias"] if "bias" in parameters["proj"] else None
        self.relative_position_bias = parameters["relative_position_bias"]

        # Scale factor
        self.scale = self.head_dim**-0.5

    def forward(self, x, rpi, mask=None):
        b_, n, c = x.shape
        if x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        self.memory_config = ttnn.L1_MEMORY_CONFIG if b_ * n * c < 1_100_000 else ttnn.DRAM_MEMORY_CONFIG
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=self.qkv_bias,
            memory_config=self.memory_config,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(x)
        tile_size = 32
        # for 180 dim, 540 qkvshape[-1] :- head_size = 30, padded_head_size = 32
        head_size = qkv.shape[-1] // (3 * self.num_heads)
        padded_head_size = ((head_size + tile_size - 1) // tile_size) * tile_size
        pad = padded_head_size != head_size
        if pad:  # add padding
            qkv_torch = ttnn.to_torch(qkv)
            input_tensor_heads = torch.split(qkv_torch, head_size, dim=-1)
            input_tensor_heads = [
                torch.nn.functional.pad(head, (0, padded_head_size - head_size), "constant", 0)
                for head in input_tensor_heads
            ]
            qkv = torch.cat(input_tensor_heads, dim=-1)
            qkv = ttnn.from_torch(
                qkv,
                device=self.device,
                dtype=ttnn.bfloat16,
                memory_config=self.memory_config,
                layout=ttnn.TILE_LAYOUT,
            )
        (
            q,
            k,
            v,
        ) = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, memory_config=ttnn.L1_MEMORY_CONFIG, num_heads=self.num_heads
        )

        ttnn.deallocate(qkv)
        if pad:  # remove padding
            q = ttnn.to_torch(q)[..., :head_size]
            k = ttnn.to_torch(k)[..., :head_size, :]
            v = ttnn.to_torch(v)[..., :head_size]

            q = ttnn.from_torch(
                q,
                device=self.device,
                dtype=ttnn.bfloat16,
                memory_config=self.memory_config,
                layout=ttnn.TILE_LAYOUT,
            )
            k = ttnn.from_torch(
                k,
                device=self.device,
                dtype=ttnn.bfloat16,
                memory_config=self.memory_config,
                layout=ttnn.TILE_LAYOUT,
            )
            v = ttnn.from_torch(
                v,
                device=self.device,
                dtype=ttnn.bfloat16,
                memory_config=self.memory_config,
                layout=ttnn.TILE_LAYOUT,
            )

        # Remove the first dimension
        q = ttnn.squeeze(q, 0)
        k = ttnn.squeeze(k, 0)
        v = ttnn.squeeze(v, 0)

        # Scale Q
        q = ttnn.multiply(q, self.scale, memory_config=ttnn.L1_MEMORY_CONFIG)

        attn = ttnn.matmul(
            q,
            k,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        attn = ttnn.add(attn, self.relative_position_bias, memory_config=self.memory_config)

        # Apply mask if provided
        if mask is not None:
            nw = mask.shape[0]
            attn = ttnn.reshape(attn, [b_ // nw, nw, self.num_heads, n, n])
            mask_expanded = ttnn.unsqueeze(ttnn.unsqueeze(mask, 1), 0)
            attn = ttnn.add(attn, mask_expanded)
            attn = ttnn.reshape(attn, [-1, self.num_heads, n, n])

        # Softmax
        attn = ttnn.softmax(attn, dim=-1, memory_config=self.memory_config)

        # Apply attention to values
        x = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )  # [b_, num_heads, n, head_dim]

        # import pdb; pdb.set_trace()
        # Transpose and reshape back
        # x = ttnn.transpose(x, 1, 2, memory_config=self.memory_config)  # [b_, n, num_heads, head_dim]
        # x = ttnn.reshape(x, [b_, n, c], memory_config=self.memory_config)

        # x = ttnn.pad(x, ((0, 0), (0, 0), (0, 0), (0, 2)), value=0.0)
        # # TODO: fix this
        # x = ttnn.transformer.concatenate_heads( x, memory_config=self.memory_config)
        # original_final_dim = 180
        # start_indices = [0, 0, 0]
        # end_indices = [x.shape[0], x.shape[1], original_final_dim]
        # x = ttnn.slice(x, start_indices, end_indices, memory_config=self.memory_config)
        # program_config = matmul_config(x.shape[-2], x.shape[-1], self.proj_bias.shape[-1])

        # TODO: find a better way to do padding, maybe pad the weights of the prev matmul
        if pad:
            x = ttnn.to_torch(x)
            padded_head_size = 32
            head_size = 30

            x = torch.nn.functional.pad(x, (0, padded_head_size - head_size), "constant", 0)
            input_tensor = ttnn.from_torch(
                x,
                device=self.device,
                dtype=ttnn.bfloat16,
                memory_config=self.memory_config,
                layout=ttnn.TILE_LAYOUT,
            )
        output_tensor = ttnn.transformer.concatenate_heads(input_tensor)

        if pad:
            output_tensor = ttnn.to_torch(output_tensor)

            # Remove the padding
            output_tensor = torch.cat(
                [chunk[..., :head_size] for chunk in torch.split(output_tensor, padded_head_size, dim=-1)], dim=-1
            )
        x = ttnn.from_torch(
            output_tensor,
            device=self.device,
            dtype=ttnn.bfloat16,
            memory_config=self.memory_config,
            layout=ttnn.TILE_LAYOUT,
        )

        x = ttnn.linear(
            x,
            self.proj_weight,
            bias=self.proj_bias,
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            # program_config=program_config,
        )
        return x
