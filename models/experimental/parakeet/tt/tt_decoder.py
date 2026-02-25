import torch
from models.experimental.parakeet.tt.tt_lstm import TtLSTMCell
import ttnn
from typing import Any, Dict, Optional, Tuple, List


class TtRNNTDecoder:
    """
    TTNN version of NeMo RNNTDecoder (prediction network).

    Matches NeMo semantics:
    - predict(y, state, add_sos, batch_size) -> (g, (h, c))
      where g is [B, U(+1), H] and h,c are [L, B, H]
    - initialize_state(y) -> (h, c) each [L, B, H]
    - batch_replace_states_mask / batch_split_states helpers
    """

    def __init__(
        self,
        device,
        prednet: Dict[str, Any],
        vocab_size: int,
        normalization_mode: Optional[str] = None,  # not implemented (kept for API parity)
        random_state_sampling: bool = False,
        blank_as_pad: bool = True,
        dtype=ttnn.float32,
        memory_config=None,
        # Optional real weights (recommended for accuracy parity)
        embedding_weight_torch: Optional[torch.Tensor] = None,  # [V(+1), H]
        lstm_weights_torch: Optional[
            List[Dict[str, torch.Tensor]]
        ] = None,  # len=L; each has weight_ih, weight_hh, bias_ih, bias_hh
    ):
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        self.pred_hidden = int(prednet["pred_hidden"])
        self.pred_rnn_layers = int(prednet["pred_rnn_layers"])
        self.blank_idx = vocab_size
        self.blank_as_pad = blank_as_pad
        self.random_state_sampling = random_state_sampling

        # Keep a training flag for parity with NeMo behavior.
        # (You can toggle this externally if you want random_state_sampling during "training".)
        self.training = False

        # Build modules (embed + LSTM cell stack)
        self.prediction = self._predict_modules(
            vocab_size=vocab_size,
            pred_n_hidden=self.pred_hidden,
            pred_rnn_layers=self.pred_rnn_layers,
            normalization_mode=normalization_mode,  # ignored for now
            embedding_weight_torch=embedding_weight_torch,
            lstm_weights_torch=lstm_weights_torch,
        )

    def _predict_modules(
        self,
        vocab_size: int,
        pred_n_hidden: int,
        pred_rnn_layers: int,
        normalization_mode: Optional[str],
        embedding_weight_torch: Optional[torch.Tensor],
        lstm_weights_torch: Optional[List[Dict[str, torch.Tensor]]],
    ) -> Dict[str, Any]:
        # ---- Embedding weight ----
        total_vocab = vocab_size + 1 if self.blank_as_pad else vocab_size

        if embedding_weight_torch is None:
            # fallback init (for correctness tests you should load from checkpoint)
            embedding_weight_torch = torch.randn(total_vocab, pred_n_hidden) * (1.0 / (pred_n_hidden**0.5))

        if self.blank_as_pad:
            embedding_weight_torch = embedding_weight_torch.clone()
            embedding_weight_torch[self.blank_idx].zero_()  # blank/pad embedding -> zeros (NeMo pad behavior)

        embed_weight = ttnn.from_torch(
            embedding_weight_torch,
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # embedding tables typically ROW_MAJOR
            memory_config=self.memory_config,
        )

        # ---- LSTM cell stack ----
        # Reuse your existing multi-layer TTNN LSTM-cell implementation

        lstm_cell = TtLSTMCell(
            input_size=pred_n_hidden,
            hidden_size=pred_n_hidden,
            device=self.device,
            num_layers=pred_rnn_layers,
            dtype=self.dtype,
            memory_config=self.memory_config,
            weights=lstm_weights_torch,  # pass real weights extracted from checkpoint (recommended)
        )

        return {"embed_weight": embed_weight, "dec_rnn_cell": lstm_cell}

    def initialize_state(self, y: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        NeMo parity: input y is used to infer batch size (batch = y.size(0) in NeMo).

        Here we assume y is [B, T, H] (post-embedding), so batch = y.shape[0].
        Returns (h, c) each [L, B, H] in TILE_LAYOUT.
        """
        B = int(y.shape[0])

        if self.random_state_sampling and self.training:
            h_torch = torch.randn(self.pred_rnn_layers, B, self.pred_hidden) * 0.1
            c_torch = torch.randn(self.pred_rnn_layers, B, self.pred_hidden) * 0.1
            h = ttnn.from_torch(
                h_torch, dtype=self.dtype, device=self.device, layout=ttnn.TILE_LAYOUT, memory_config=self.memory_config
            )
            c = ttnn.from_torch(
                c_torch, dtype=self.dtype, device=self.device, layout=ttnn.TILE_LAYOUT, memory_config=self.memory_config
            )
        else:
            h = ttnn.zeros(
                [self.pred_rnn_layers, B, self.pred_hidden],
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.memory_config,
            )
            c = ttnn.zeros(
                [self.pred_rnn_layers, B, self.pred_hidden],
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.memory_config,
            )

        return h, c

    def _embed(self, y_tokens: ttnn.Tensor) -> ttnn.Tensor:
        # y_tokens: [B, U] uint32 -> returns [B, U, H]
        return ttnn.embedding(y_tokens, self.prediction["embed_weight"])

    def predict(
        self,
        y: Optional[ttnn.Tensor] = None,  # [B, U] uint32
        state: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,  # (h, c) each [L, B, H]
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[ttnn.Tensor, Tuple[ttnn.Tensor, ttnn.Tensor]]:
        """
        Mirrors NeMo RNNTDecoder.predict.

        Returns:
          g: [B, U(+1), H]
          hid: (h, c) each [L, B, H]
        """
        # ---- Build embedded y: [B, U, H] or [B, 1, H] ----
        if y is not None:
            embedded = self._embed(y)
        else:
            # NeMo behavior: if y is None -> zeros [B, 1, H]
            if batch_size is None:
                if state is None:
                    raise ValueError("batch_size cannot be None when y is None and state is None")
                B = int(state[0].shape[1])
            else:
                B = int(batch_size)

            embedded = ttnn.zeros(
                [B, 1, self.pred_hidden],
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.memory_config,
            )

        # Ensure TILE for concat / LSTM math
        embedded = ttnn.to_layout(embedded, ttnn.TILE_LAYOUT, memory_config=self.memory_config)

        # ---- add_sos: prepend zeros [B, 1, H] -> [B, U+1, H] ----
        if add_sos:
            B, U, H = embedded.shape
            start = ttnn.zeros(
                [B, 1, H],
                dtype=self.dtype,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.memory_config,
            )
            embedded = ttnn.concat([start, embedded], dim=1)

        # ---- Initialize state (NeMo: only in training + random_state_sampling) ----
        if state is None:
            if self.random_state_sampling and self.training:
                state = self.initialize_state(embedded)

        h, c = state if state is not None else (None, None)

        # ---- LSTM sequence forward: emulate torch.nn.LSTM over time ----
        # embedded: [B, T, H] -> y_seq: [T, B, H]
        y_seq = ttnn.transpose(embedded, 0, 1)

        # Split along time into [1, B, H] chunks, then run cell step-by-step
        timesteps = ttnn.split(y_seq, 1, dim=0)

        outputs = []
        cell = self.prediction["dec_rnn_cell"]

        for x_t in timesteps:
            # x_t: [1, B, H]
            out_t, (h, c) = cell(x_t, h, c)  # out_t: [1, B, H]; h,c: [L, B, H]
            outputs.append(out_t)

        g_seq = ttnn.concat(outputs, dim=0)  # [T, B, H]
        g = ttnn.transpose(g_seq, 0, 1)  # [B, T, H]

        return g, (h, c)

    @classmethod
    def batch_replace_states_mask(
        cls,
        src_states: Tuple[ttnn.Tensor, ttnn.Tensor],
        dst_states: Tuple[ttnn.Tensor, ttnn.Tensor],
        mask,  # torch.Tensor or ttnn.Tensor, shape [B]
        other_src_states: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        TTNN equivalent of NeMo batch_replace_states_mask.
        Replaces dst state for batch elements where mask==True using src state.

        Shapes:
          src/dst/other: (h, c) each [L, B, H]
          mask: [B]
        """
        other = other_src_states if other_src_states is not None else dst_states

        h_dst, c_dst = dst_states
        h_src, c_src = src_states
        h_other, c_other = other

        # Build condition tensor [L, B, H]
        if torch.is_tensor(mask):
            cond_torch = mask.to(torch.int32).reshape(1, -1, 1)  # [1, B, 1]
            cond = ttnn.from_torch(cond_torch, dtype=ttnn.int32, device=h_dst.device(), layout=ttnn.TILE_LAYOUT)
        else:
            # assume TTNN tensor [B] or [1,B,1]
            cond = mask
            if len(cond.shape) == 1:
                cond = ttnn.reshape(cond, [1, cond.shape[0], 1])

        cond = ttnn.to_layout(cond, ttnn.TILE_LAYOUT)
        cond = ttnn.experimental.broadcast_to(cond, ttnn.Shape(h_dst.shape))

        # where(cond, true, false)
        new_h = ttnn.where(cond, h_src, h_other)
        new_c = ttnn.where(cond, c_src, c_other)
        return new_h, new_c

    @classmethod
    def batch_split_states(cls, batch_states: Tuple[ttnn.Tensor, ttnn.Tensor]) -> List[Tuple[ttnn.Tensor, ttnn.Tensor]]:
        """
        TTNN equivalent of NeMo batch_split_states.

        Input:
          batch_states: (h, c) each [L, B, H]
        Output:
          list length B: [(h_i, c_i)] where each is [L, H]
        """
        h, c = batch_states
        h_list = ttnn.split(h, 1, dim=1)  # each [L, 1, H]
        c_list = ttnn.split(c, 1, dim=1)

        out: List[Tuple[ttnn.Tensor, ttnn.Tensor]] = []
        for h_i, c_i in zip(h_list, c_list):
            out.append((ttnn.squeeze(h_i, 1), ttnn.squeeze(c_i, 1)))  # [L, H]
        return out
