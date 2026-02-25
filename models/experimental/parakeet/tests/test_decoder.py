"""
Compare NeMo RNNTDecoder vs TTNN TtRNNTDecoder (models/experimental/parakeet/tt/tt_decoder.py)
using real weights from the pretrained Parakeet v2 0.6 model.

What this script does:
- Loads NeMo ASRModel from pretrained name/path (default: parakeet v2 0.6)
- Finds the RNNTDecoder module
- Extracts embedding + per-layer LSTM weights from NeMo decoder
- Loads those weights into TTNN decoder implementation
- Runs inference-style decoder.predict() in 3 cases:
  1) tokens (y provided)
  2) blank (y=None)
  3) stateful (use state from blank)
- Normalizes output/state shapes and compares with PCC + abs error

Usage:
  python models/experimental/parakeet/tests/test_decoder_parakeet_v2_0_6.py --model parakeet-v2-0.6

If NeMo import fails with "No usable temporary directory", make sure /tmp exists or set TMPDIR.
This script sets TMPDIR=/tmp automatically before importing torch/nemo.
"""

import os
import sys

# ---- Must be set before importing torch/nemo (dill -> tempfile.gettempdir()) ----
os.environ.setdefault("TMPDIR", "/tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# Disable any pdb.set_trace() left in the repo / site-packages
import pdb  # noqa: E402

pdb.set_trace = lambda *args, **kwargs: None  # type: ignore

import argparse  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import torch  # noqa: E402
import ttnn  # noqa: E402

import nemo.collections.asr as nemo_asr  # noqa: E402
from nemo.collections.asr.modules.rnnt import RNNTDecoder  # noqa: E402

from models.common.metrics import compute_pcc, compute_max_abs_error, compute_mean_abs_error  # noqa: E402
from models.experimental.parakeet.tt.tt_decoder import TtRNNTDecoder  # noqa: E402


def _repo_root_from_this_file() -> str:
    # .../models/experimental/parakeet/tests/<this_file>
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


# Ensure repo root is on sys.path so `models.*` imports work when running directly
REPO_ROOT = _repo_root_from_this_file()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def find_nemo_rnnt_decoder(asr_model) -> RNNTDecoder:
    for name, module in asr_model.named_modules():
        if isinstance(module, RNNTDecoder):
            print(f"✅ Found RNNTDecoder at `{name}`")
            return module
    raise RuntimeError("Could not find RNNTDecoder inside the loaded NeMo model")


def find_torch_lstm_module(dec_rnn: torch.nn.Module) -> torch.nn.LSTM:
    # NeMo norm=None uses LSTMDropout which holds .lstm (torch.nn.LSTM)
    if hasattr(dec_rnn, "lstm") and isinstance(dec_rnn.lstm, torch.nn.LSTM):
        return dec_rnn.lstm

    for m in dec_rnn.modules():
        if isinstance(m, torch.nn.LSTM):
            return m

    raise RuntimeError("Could not locate torch.nn.LSTM inside nemo_decoder.prediction['dec_rnn']")


def load_pretrained_nemo_decoder(model_name_or_path: str, map_location: str = "cpu") -> RNNTDecoder:
    print(f"\nLoading Parakeet model: {model_name_or_path!r}")

    model = nemo_asr.models.EncDecRNNTModel.from_pretrained(
        model_name=model_name_or_path,
        map_location=map_location,
    )

    model.eval()

    # ✅ Direct access (correct way)
    decoder = model.decoder

    if not isinstance(decoder, RNNTDecoder):
        raise TypeError("Loaded decoder is not RNNTDecoder")

    print("✅ RNNTDecoder loaded successfully")

    print(f"  blank_idx       : {decoder.blank_idx}")
    print(f"  pred_hidden    : {decoder.pred_hidden}")
    print(f"  pred_rnn_layers: {decoder.pred_rnn_layers}")
    print(f"  vocab_size     : {decoder.vocab_size}")

    return decoder


def extract_decoder_weights_from_nemo(nemo_decoder: RNNTDecoder) -> Dict[str, Any]:
    embed = nemo_decoder.prediction["embed"]
    dec_rnn = nemo_decoder.prediction["dec_rnn"]
    lstm = find_torch_lstm_module(dec_rnn)

    # Projection LSTM is not supported by current TT LSTM cell implementation
    proj_size = getattr(lstm, "proj_size", 0)
    if proj_size not in (0, None):
        raise NotImplementedError(
            f"NeMo decoder LSTM uses proj_size={proj_size}. "
            "Current TTNN TtLSTMCell in this repo does not implement LSTM projection."
        )

    print("\nExtracting weights from NeMo decoder:")
    print(f"  - embedding.weight: {tuple(embed.weight.shape)}")
    print(
        f"  - lstm.input_size={lstm.input_size}, hidden_size={lstm.hidden_size}, num_layers={lstm.num_layers}, proj_size={proj_size}"
    )

    embedding_weight = embed.weight.detach().cpu().clone()

    lstm_weights: List[Dict[str, torch.Tensor]] = []
    for layer_idx in range(lstm.num_layers):
        lstm_weights.append(
            {
                "weight_ih": getattr(lstm, f"weight_ih_l{layer_idx}").detach().cpu().clone(),
                "weight_hh": getattr(lstm, f"weight_hh_l{layer_idx}").detach().cpu().clone(),
                "bias_ih": getattr(lstm, f"bias_ih_l{layer_idx}").detach().cpu().clone(),
                "bias_hh": getattr(lstm, f"bias_hh_l{layer_idx}").detach().cpu().clone(),
            }
        )

    return {
        "vocab_size": nemo_decoder.blank_idx,  # blank index is vocab_size in NeMo
        "pred_hidden": nemo_decoder.pred_hidden,
        "pred_rnn_layers": nemo_decoder.pred_rnn_layers,
        "embedding_weight": embedding_weight,
        "lstm_weights": lstm_weights,
    }


def load_weights_to_tt_decoder(tt_decoder: TtRNNTDecoder, weights: Dict[str, Any]) -> None:
    print("\nLoading weights into TT decoder...")

    # Embedding table
    tt_decoder.embedding_weight = ttnn.from_torch(
        weights["embedding_weight"],
        dtype=ttnn.bfloat16,  # NeMo weights are fp32/fp16; embedding table ok in bf16
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=tt_decoder.device,
        memory_config=getattr(tt_decoder, "memory_config", ttnn.DRAM_MEMORY_CONFIG),
    )

    # LSTM weights (per-layer) -> TtLSTMCell._load_weights expects a list[dict]
    tt_decoder.lstm_dropout.lstm._load_weights(weights["lstm_weights"])
    print("  ✅ weights loaded")


def ensure_bth(x: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    """
    Normalize to [B, T, H].
    NeMo can sometimes return [T, B, H] (depending on local modifications); TT path should be [B, T, H].
    """
    if x.dim() != 3:
        raise ValueError(f"{name}: expected 3D tensor, got shape={tuple(x.shape)}")

    if x.shape[0] == batch_size:
        return x  # already [B, T, H]
    if x.shape[1] == batch_size:
        return x.transpose(0, 1).contiguous()  # [T, B, H] -> [B, T, H]

    raise ValueError(f"{name}: cannot infer batch dim. shape={tuple(x.shape)} batch_size={batch_size}")


def ensure_lbh(x: torch.Tensor, num_layers: int, batch_size: int, name: str) -> torch.Tensor:
    """
    Normalize to [L, B, H].
    """
    if x.dim() != 3:
        raise ValueError(f"{name}: expected 3D [L,B,H], got shape={tuple(x.shape)}")

    # Common: [L,B,H]
    if x.shape[0] == num_layers and x.shape[1] == batch_size:
        return x

    # Sometimes swapped: [B,L,H]
    if x.shape[0] == batch_size and x.shape[1] == num_layers:
        return x.transpose(0, 1).contiguous()

    raise ValueError(
        f"{name}: cannot normalize to [L,B,H]. shape={tuple(x.shape)} expected L={num_layers}, B={batch_size}"
    )


def compare_tensors(a: torch.Tensor, b: torch.Tensor, label: str, pcc_threshold: float) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape mismatch: impl={tuple(a.shape)} ref={tuple(b.shape)}")

    pcc = compute_pcc(a, b)
    mx = compute_max_abs_error(a, b)
    mn = compute_mean_abs_error(a, b)
    print(f"\n{label}:")
    print(f"  - PCC: {pcc:.6f}")
    print(f"  - Max Abs Error: {mx:.6f}")
    print(f"  - Mean Abs Error: {mn:.6f}")
    if pcc < pcc_threshold:
        print(f"  ❌ FAIL (PCC < {pcc_threshold})")
    else:
        print(f"  ✅ PASS (PCC >= {pcc_threshold})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", type=str, default="nvidia/parakeet-tdt-0.6b-v2", help="NeMo pretrained name or path to .nemo"
    )
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--U", type=int, default=1, help="label length (token sequence length)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pcc", type=float, default=0.99)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # 1) Load NeMo decoder + extract weights
    nemo_decoder = load_pretrained_nemo_decoder(args.model, map_location="cpu")
    weights = extract_decoder_weights_from_nemo(nemo_decoder)

    vocab_size = int(weights["vocab_size"])
    pred_hidden = int(weights["pred_hidden"])
    pred_rnn_layers = int(weights["pred_rnn_layers"])

    # 2) Open TT device + init TT decoder
    device = ttnn.open_device(device_id=args.device_id)
    try:
        prednet_config = {
            "pred_hidden": pred_hidden,
            "pred_rnn_layers": pred_rnn_layers,
            "dropout": 0.0,
            "forget_gate_bias": 1.0,
        }

        tt_decoder = TtRNNTDecoder(
            device=device,
            prednet=prednet_config,
            vocab_size=vocab_size,
            normalization_mode=None,
            random_state_sampling=False,
            blank_as_pad=True,
            dtype=ttnn.float32,
            embedding_weight_torch=weights["embedding_weight"],
            lstm_weights_torch=weights["lstm_weights"],
        )
        B = int(args.batch_size)
        U = int(args.U)

        # Test inputs (token ids): torch long for NeMo, uint32 for TTNN
        y_torch = torch.randint(low=0, high=vocab_size, size=(B, U), dtype=torch.long)
        y_tt = ttnn.from_torch(y_torch.to(torch.int64), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)

        # Helper to init NeMo state robustly
        # NeMo initialize_state uses y.size(0) => pass [B,1,H]
        nemo_state_init_in = torch.zeros((B, 1, nemo_decoder.pred_hidden), dtype=torch.float32)
        nemo_state = nemo_decoder.initialize_state(nemo_state_init_in)

        # Helper to init TT state (repo implementation takes batch_size)
        tt_state_init_in = ttnn.from_torch(
            nemo_state_init_in,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        tt_state = tt_decoder.initialize_state(tt_state_init_in)

        print("\n" + "=" * 80)
        print("CASE 1: tokens (y provided), add_sos=False")
        print("=" * 80)
        with torch.no_grad():
            nemo_out, (nemo_h, nemo_c) = nemo_decoder.predict(y=y_torch, state=nemo_state, add_sos=False, batch_size=B)

        tt_out, (tt_h, tt_c) = tt_decoder.predict(y=y_tt, state=tt_state, add_sos=False, batch_size=B)

        # Normalize shapes to match comparison intent
        nemo_out_bth = ensure_bth(nemo_out.to(torch.float32), B, "nemo_out")
        tt_out_bth = ensure_bth(ttnn.to_torch(tt_out).to(torch.float32), B, "tt_out")

        nemo_h_lbh = ensure_lbh(nemo_h.to(torch.float32), pred_rnn_layers, B, "nemo_h")
        nemo_c_lbh = ensure_lbh(nemo_c.to(torch.float32), pred_rnn_layers, B, "nemo_c")

        tt_h_lbh = ensure_lbh(ttnn.to_torch(tt_h).to(torch.float32), pred_rnn_layers, B, "tt_h")
        tt_c_lbh = ensure_lbh(ttnn.to_torch(tt_c).to(torch.float32), pred_rnn_layers, B, "tt_c")

        compare_tensors(tt_out_bth, nemo_out_bth, "Output (tokens, add_sos=False)", args.pcc)
        compare_tensors(tt_h_lbh, nemo_h_lbh, "Hidden state (tokens)", args.pcc)
        compare_tensors(tt_c_lbh, nemo_c_lbh, "Cell state (tokens)", args.pcc)

        print("\n" + "=" * 80)
        print("CASE 2: blank (y=None), add_sos=False")
        print("=" * 80)
        with torch.no_grad():
            nemo_out_blank, (nemo_h_blank, nemo_c_blank) = nemo_decoder.predict(
                y=None, state=None, add_sos=False, batch_size=B
            )

        tt_out_blank, (tt_h_blank, tt_c_blank) = tt_decoder.predict(y=None, state=None, add_sos=False, batch_size=B)

        nemo_out_blank_bth = ensure_bth(nemo_out_blank.to(torch.float32), B, "nemo_out_blank")
        tt_out_blank_bth = ensure_bth(ttnn.to_torch(tt_out_blank).to(torch.float32), B, "tt_out_blank")

        compare_tensors(tt_out_blank_bth, nemo_out_blank_bth, "Output (blank, add_sos=False)", args.pcc)

        print("\n" + "=" * 80)
        print("CASE 3: stateful (use blank state), add_sos=False")
        print("=" * 80)
        with torch.no_grad():
            nemo_out_stateful, (nemo_h_stateful, nemo_c_stateful) = nemo_decoder.predict(
                y=y_torch, state=(nemo_h_blank, nemo_c_blank), add_sos=False, batch_size=B
            )

        tt_out_stateful, (tt_h_stateful, tt_c_stateful) = tt_decoder.predict(
            y=y_tt, state=(tt_h_blank, tt_c_blank), add_sos=False, batch_size=B
        )

        nemo_out_stateful_bth = ensure_bth(nemo_out_stateful.to(torch.float32), B, "nemo_out_stateful")
        tt_out_stateful_bth = ensure_bth(ttnn.to_torch(tt_out_stateful).to(torch.float32), B, "tt_out_stateful")

        compare_tensors(tt_out_stateful_bth, nemo_out_stateful_bth, "Output (stateful, add_sos=False)", args.pcc)

        print(
            "\n✅ Done. If any comparison failed, check shape normalization and whether TT decoder implements true sequence LSTM (timestep loop)."
        )

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
