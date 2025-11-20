import torch
import ttnn


from models.experimental.bevdepth.reference.bev_depth_head import BEVDepthHead
from models.experimental.bevdepth.tt.bev_depth_head import TtBEVDepthHead
from models.experimental.bevdepth.tt.head_preprocessing import (
    load_weights,
    torch_load_weights,
)

from tests.ttnn.utils_for_testing import check_with_pcc

device = ttnn.open_device(device_id=0, l1_small_size=32768)

ref_head = BEVDepthHead()
torch_load_weights(ref_head, "../resources/bev_depth_lss_r50_256x704_128x128_24e_2key.pth")
ref_head.eval()

# Instantiate the model
ttnn_head = TtBEVDepthHead(device)
load_weights(ttnn_head, "../resources/bev_depth_lss_r50_256x704_128x128_24e_2key.pth")

# Generate random input with shape (2, 64, 128, 128)
torch.manual_seed(0)
torch_input = torch.randn(2, 64, 128, 128)

# Run PyTorch model
with torch.no_grad():
    ref_output = ref_head(torch_input)

# Convert input to TTNN format (NCHW -> NHWC)
torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
ttnn_input = ttnn.from_torch(
    torch_input_nhwc, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
)

# Run TTNN model
ttnn_output, out_h, out_w = ttnn_head(ttnn_input)

# Convert TTNN output back to PyTorch format
ttnn_output_torch = ttnn.to_torch(ttnn_output)
ttnn_output_torch = ttnn_output_torch.reshape(2, out_h, out_w, 2)
ttnn_output_torch = ttnn_output_torch.permute(0, 3, 1, 2)  # NHWC -> NCHW
print("Ref out:", ref_output.size())
print("TTNN out:", ttnn_output_torch.size())

# Compare outputs using PCC
passing, pcc_message = check_with_pcc(ref_output, ttnn_output_torch, pcc=0.95)
print(f"PCC Test: {pcc_message}")
print(f"Passing: {passing}")

ttnn.close_device(device)
