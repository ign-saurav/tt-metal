import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc, tt_to_torch_tensor
from models.experimental.detr3d.reference.torch_pointnet2_ops import BallQuery, GroupingOperation, QueryAndGroup
from models.experimental.detr3d.ttnn.pointnet_samodule_votes import (
    TtnnBallQuery,
    TtnnGroupingOperation,
    TtnnQueryAndGroup,
    TtnnFurthestPointSampling,
    TtnnGatherOperation,
)
from models.experimental.detr3d.reference import torch_pointnet2_ops as pointnet2_utils


@pytest.mark.parametrize("pcc", ((0.99,),))
def test_ball_query_pcc(device, pcc):
    # Test parameters
    radius = 0.4
    nsample = 32

    # Input shapes
    batch_size = 1
    m = 1024  # query points
    n = 2048  # input points

    # Create random input tensors
    torch.manual_seed(0)
    # new_xyz = torch.rand(batch_size, m, 3, dtype=torch.bfloat16)
    # xyz = torch.rand(batch_size, n, 3, dtype=torch.bfloat16)
    new_xyz = torch.load("new_xyz.pt")
    xyz = torch.load(" xyz.pt")

    # PyTorch implementation
    torch_ball_query = BallQuery(radius=radius, nsample=nsample)
    torch_output = torch_ball_query(xyz, new_xyz)

    # Convert inputs to TTNN tensors
    new_xyz_ttnn = ttnn.from_torch(new_xyz, device=device, dtype=ttnn.bfloat16)
    xyz_ttnn = ttnn.from_torch(xyz, device=device, dtype=ttnn.bfloat16)

    # TTNN implementation
    ttnn_ball_query = TtnnBallQuery(device=device, radius=radius, nsample=nsample)
    ttnn_output = ttnn_ball_query(xyz_ttnn, new_xyz_ttnn)

    # Convert TTNN output back to torch for comparison
    ttnn_output_converted = tt_to_torch_tensor(ttnn_output)

    # Compare results using PCC
    does_pass, pcc_message = comp_pcc(torch_output, ttnn_output_converted, pcc)

    print(f"PCC result: {pcc_message}")

    assert does_pass, f"PCC {pcc_message} is below threshold {pcc}"


@pytest.mark.parametrize(
    "batch_size, channels, num_points, num_queries, num_samples",
    [
        (1, 3, 2048, 1024, 32),  # Example from user
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        # ttnn.float32,
    ],
)
def test_group_points(device, batch_size, channels, num_points, num_queries, num_samples, dtype):
    """Test group_points operation comparing PyTorch and TTNN implementations."""
    torch.manual_seed(42)

    # Create input tensors
    torch_points = torch.randn(batch_size, channels, num_points, dtype=torch.float32)
    torch_idx = torch.randint(0, num_points, (batch_size, num_queries, num_samples))

    # PyTorch implementation
    pytorch_grouping = GroupingOperation()
    pytorch_output = pytorch_grouping(torch_points, torch_idx)

    # TTNN implementation
    tt_points = ttnn.from_torch(torch_points, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_idx = ttnn.from_torch(torch_idx, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_grouping = TtnnGroupingOperation()
    ttnn_output = ttnn_grouping(tt_points, tt_idx)

    # Convert back to torch for comparison
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # Compare results

    does_pass, pcc_message = comp_pcc(pytorch_output, ttnn_output_torch, 0.99)

    print(f"PCC result: {pcc_message}")

    assert does_pass, f"PCC {pcc_message} is below threshold {0.99}"


@pytest.mark.parametrize("pcc", ((0.99,),))
def test_query_and_group_pcc(device, pcc):
    """Test QueryAndGroup with specific parameters comparing PyTorch and TTNN implementations."""

    # Test parameters from user
    radius = 0.2
    nsample = 64
    use_xyz = True
    ret_grouped_xyz = True
    normalize_xyz = True
    sample_uniformly = False
    ret_unique_cnt = False

    # Input shapes from user
    batch_size = 1
    n = 2000  # input points
    m = 2048  # query points
    channels = 256  # feature channels

    # Create random input tensors
    torch.manual_seed(42)
    xyz = torch.randn(batch_size, n, 3, dtype=torch.bfloat16)
    new_xyz = torch.randn(batch_size, m, 3, dtype=torch.bfloat16)

    inds = pointnet2_utils.furthest_point_sample(xyz, m)
    new_xyz = pointnet2_utils.gather_operation(xyz.transpose(1, 2), inds).transpose(1, 2)
    features = None

    # PyTorch implementation
    # torch.Size([1, 20000, 3])
    # torch.Size([1, 2048, 3])
    torch_query_group = QueryAndGroup(
        radius=radius,
        nsample=nsample,
        use_xyz=use_xyz,
        ret_grouped_xyz=ret_grouped_xyz,
        normalize_xyz=normalize_xyz,
        sample_uniformly=sample_uniformly,
        ret_unique_cnt=ret_unique_cnt,
    )
    torch_output = torch_query_group(xyz, new_xyz, features)

    # Convert inputs to TTNN tensors
    xyz_ttnn = ttnn.from_torch(xyz, device=device, dtype=ttnn.bfloat16)
    new_xyz_ttnn = ttnn.from_torch(new_xyz, device=device, dtype=ttnn.bfloat16)
    features_ttnn = ttnn.from_torch(features, device=device, dtype=ttnn.bfloat16)

    # TTNN implementation
    ttnn_query_group = TtnnQueryAndGroup(
        device=device,
        radius=radius,
        nsample=nsample,
        use_xyz=use_xyz,
        ret_grouped_xyz=ret_grouped_xyz,
        normalize_xyz=normalize_xyz,
        sample_uniformly=sample_uniformly,
        ret_unique_cnt=ret_unique_cnt,
    )
    ttnn_output = ttnn_query_group(xyz_ttnn, new_xyz_ttnn, features_ttnn)

    # Convert TTNN outputs back to torch for comparison
    # Both implementations return tuples since ret_grouped_xyz=True
    ttnn_output_converted = tuple(tt_to_torch_tensor(o) for o in ttnn_output)

    # Compare results using PCC
    # Compare main features (first element of tuple)
    does_pass_features, pcc_message_features = comp_pcc(torch_output[0], ttnn_output_converted[0], pcc)
    print(f"Features PCC result: {pcc_message_features}")

    # Compare grouped_xyz (second element of tuple)
    does_pass_xyz, pcc_message_xyz = comp_pcc(torch_output[1], ttnn_output_converted[1], pcc)
    print(f"Grouped XYZ PCC result: {pcc_message_xyz}")

    # Assert both comparisons pass
    assert does_pass_features, f"Features PCC {pcc_message_features} is below threshold {pcc}"
    assert does_pass_xyz, f"Grouped XYZ PCC {pcc_message_xyz} is below threshold {pcc}"


@pytest.mark.parametrize("pcc", ((0.99,),))
def test_gather_operation_pcc(device, pcc):
    torch.manual_seed(0)

    # Shapes
    B, C, N, M = 1, 8, 256, 64

    # Inputs
    points = torch.randn(B, C, N)
    idx = torch.randint(0, N, (B, M))

    # PyTorch reference
    torch_out = pointnet2_utils.gather_operation(points, idx)

    # TTNN implementation
    points_ttnn = ttnn.from_torch(points, dtype=ttnn.bfloat16, device=device)
    idx_ttnn = ttnn.from_torch(idx, dtype=ttnn.uint32, device=device)

    ttnn_gather = TtnnGatherOperation()
    ttnn_out = ttnn_gather(points_ttnn, idx_ttnn)

    ttnn_out_torch = tt_to_torch_tensor(ttnn_out)

    # PCC check
    does_pass, pcc_message = comp_pcc(torch_out, ttnn_out_torch, pcc)
    print(f"GatherOperation PCC result: {pcc_message}")

    assert does_pass, f"GatherOperation PCC {pcc_message} is below threshold {pcc}"


@pytest.mark.parametrize("pcc", ((0.99,),))
def test_furthest_point_sampling_pcc(device, pcc):
    torch.manual_seed(0)

    # Shapes
    B, N, npoint = 1, 512, 128

    # Input
    xyz = torch.randn(B, N, 3, dtype=torch.float32)

    # PyTorch reference
    ref_idx = pointnet2_utils.furthest_point_sample(xyz, npoint)
    ref_xyz = torch.gather(xyz, 1, ref_idx.unsqueeze(-1).expand(-1, -1, 3))
    ref_dist = torch.cdist(ref_xyz, ref_xyz)

    # TTNN implementation
    xyz_ttnn = ttnn.from_torch(xyz, dtype=ttnn.bfloat16, device=device)

    ttnn_fps = TtnnFurthestPointSampling()
    ttnn_idx = ttnn_fps(xyz_ttnn, npoint, device)

    if isinstance(ttnn_idx, tuple):
        ttnn_idx = ttnn_idx[0]

    ttnn_idx = tt_to_torch_tensor(ttnn_idx).long()

    tt_xyz = torch.gather(xyz, 1, ttnn_idx.unsqueeze(-1).expand(-1, -1, 3))
    tt_dist = torch.cdist(tt_xyz, tt_xyz)

    # PCC check
    does_pass, pcc_message = comp_pcc(ref_dist, tt_dist, pcc)
    print(f"FurthestPointSampling PCC result: {pcc_message}")

    assert does_pass, f"FurthestPointSampling PCC {pcc_message} is below threshold {pcc}"


@pytest.mark.parametrize("pcc", ((0.995,),))
def test_compute_mask_pcc(device, pcc):
    # Create dummy class with only required functions
    class MaskOnlyEncoder:
        def __init__(self, device):
            self.device = device

        def compute_mask(self, xyz, radius, dist=None):
            with torch.no_grad():
                if dist is None or dist.shape[1] != xyz.shape[1]:
                    dist = torch.cdist(xyz, xyz, p=2)
                mask = dist >= radius

            mask_torch = torch.zeros_like(mask, dtype=torch.float)
            mask_torch = mask_torch.masked_fill(mask, float("-inf"))

            mask_ttnn = ttnn.from_torch(
                mask_torch,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
            )
            return mask_ttnn, dist

        def compute_mask_ttnn(self, xyz, radius, dist=None):
            tt_xyz = ttnn.from_torch(
                xyz,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            xyz_sq = ttnn.pow(tt_xyz, 2)
            norms = ttnn.sum(xyz_sq, dim=-1, keepdim=True)

            xyz_t = ttnn.permute(tt_xyz, (0, 2, 1))
            dot = ttnn.matmul(tt_xyz, xyz_t)

            norms_t = ttnn.permute(norms, (0, 2, 1))
            dist_sq = norms + norms_t - 2.0 * dot
            dist_sq = dist_sq + 1e-8
            dist_ttnn = ttnn.sqrt(dist_sq)

            radius_t = ttnn.full_like(dist_ttnn, radius)
            mask_ttnn = ttnn.ge(dist_ttnn, radius_t) * float("-inf")

            ttnn.deallocate(tt_xyz)
            return mask_ttnn, dist_ttnn

    # Test inputs
    B, N = 1, 64
    radius = 0.6

    torch.manual_seed(0)
    xyz = torch.randn(B, N, 3)
    xyz = xyz + torch.linspace(0, 1, N).view(1, N, 1)

    encoder = MaskOnlyEncoder(device)

    # Torch reference
    mask_ref, dist_ref = encoder.compute_mask(xyz, radius)

    # TTNN implementation
    mask_tt, dist_tt = encoder.compute_mask_ttnn(xyz, radius)

    mask_ref = ttnn.to_torch(mask_ref)
    mask_tt = ttnn.to_torch(mask_tt)
    dist_tt = ttnn.to_torch(dist_tt)

    # Mask semantic check
    ref_mask = torch.isinf(mask_ref)
    tt_mask = torch.isinf(mask_tt)
    mismatch = (ref_mask ^ tt_mask).float().mean().item()

    print("Mask mismatch ratio:", mismatch)
    assert mismatch < 1e-2

    # Distance PCC (only valid region)
    valid = dist_ref < radius
    assert valid.sum() > 20

    does_pass, pcc_msg = comp_pcc(
        dist_ref[valid],
        dist_tt[valid],
        pcc,
    )

    print(f"Distance PCC result: {pcc_msg}")
    assert does_pass, f"PCC {pcc_msg} is below threshold {pcc}"
