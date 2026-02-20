import ttnn
import torch
from typing import Optional, Tuple, List, Dict
from models.experimental.parakeet.tt.tt_lstm import TtLSTMCell


class TT_LSTMDropout:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        device=None,
        memory_config=None,
        dtype=ttnn.float32,
        weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        ttnn-based LSTMDropout. Dropout uses ttnn.experimental.dropout.
        LSTM core is not yet available in ttnn; keep a placeholder or custom implementation.
        """
        self.device = device
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.dtype = dtype
        # Placeholder: ttnn does not provide an LSTM primitive yet.
        # You could replace this with a custom ttnn LSTM implementation or keep torch.nn.LSTM for now.
        self.lstm = TtLSTMCell(input_size, hidden_size, device, num_layers, dtype, memory_config, weights)

    def forward(
        self,
        x: ttnn.Tensor,
        h: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
    ) -> Tuple[ttnn.Tensor, Tuple[ttnn.Tensor, ttnn.Tensor]]:
        # Placeholder: run torch LSTM on host; replace with ttnn LSTM when available.
        # For now, we assume x is on torch device or we transfer back/forth.
        if isinstance(x, ttnn.Tensor):
            x_torch = ttnn.to_torch(x)
        else:
            x_torch = x

        h_torch = None
        if h is not None:
            h_torch = (ttnn.to_torch(h[0]), ttnn.to_torch(h[1]))

        out_torch, h_torch = self.lstm(x_torch, h_torch)

        out = ttnn.from_torch(out_torch, dtype=x.dtype, layout=x.layout, device=self.device)
        h_out = (
            ttnn.from_torch(h_torch[0], dtype=h[0].dtype, layout=h[0].layout, device=self.device),
            ttnn.from_torch(h_torch[1], dtype=h[1].dtype, layout=h[1].layout, device=self.device),
        )
        return out, h_out
