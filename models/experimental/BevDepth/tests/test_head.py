import torch
import ttnn


from models.experimental.BevDepth.bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import (
    BEVDepthLightningModel,
)
from models.experimental.BevDepth.tt.bev_depth_head import TtBEVDepthHead

from tests.ttnn.utils_for_testing import check_with_pcc

device = ttnn.open_device(device_id=0, l1_small_size=32768)

ref_head = BEVDepthLightningModel()
ref_head.load_weights("../resources/bev_depth_lss_r50_256x704_128x128_24e_2key.pth")
ref_head.eval()

# Instantiate the model
ttnn_head = TtBEVDepthHead(device)
ttnn_head.load_weights("../resources/bev_depth_lss_r50_256x704_128x128_24e_2key.pth")

# Generate random input with shape (2, 64, 128, 128)
torch.manual_seed(0)
torch_input = torch.randn(2, 256, 128, 128)

# Run PyTorch model
with torch.no_grad():
    ref_output = ref_head(torch_input)

# Convert input to TTNN format (NCHW -> NHWC)
torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
ttnn_input = ttnn.from_torch(
    torch_input_nhwc, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
)

# Run TTNN model
ttnn_output = ttnn_head(ttnn_input)
print("Ref output:", [{k: v.shape for k, v in d.items()} for d in ref_output])
# print("TTNN output:", [{k: len(v) for k, v in d.items()} for d in ttnn_output])
ttnn_output_torch = []
for d in ttnn_output:
    head_out = {}
    for k, (t, s) in d.items():
        out = ttnn.to_torch(t)
        out = out.reshape(s)
        out = out.permute(0, 3, 1, 2)
        head_out[k] = out
    ttnn_output_torch.append(head_out)

print("TTNN output:", [{k: v.shape for k, v in d.items()} for d in ttnn_output_torch])

for d1, d2 in zip(ref_output, ttnn_output_torch):
    for k in d1.keys():
        passing, pcc_message = check_with_pcc(d1[k], d2[k], pcc=0.95)
        print(f"PCC Test: {pcc_message}")
        print(f"Passing: {passing}")

ttnn.close_device(device)
