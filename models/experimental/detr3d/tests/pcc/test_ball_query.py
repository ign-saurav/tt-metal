import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc, tt_to_torch_tensor
from models.experimental.detr3d.reference.torch_pointnet2_ops import BallQuery, GroupingOperation, QueryAndGroup
from models.experimental.detr3d.ttnn.pointnet_samodule_votes import (
    TtnnBallQuery,
    TtnnGroupingOperation,
    TtnnQueryAndGroup,
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
