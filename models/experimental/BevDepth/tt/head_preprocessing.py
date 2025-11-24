import torch
import ttnn


def fold_batch_norm2d_into_conv2d(conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_eps=1e-05):
    """Fold BatchNorm2d parameters into Conv2d weights and bias"""
    # Compute folded weights and bias
    std = torch.sqrt(bn_running_var + bn_eps)
    weight = conv_weight * (bn_weight / std).reshape(-1, 1, 1, 1)
    bias = bn_bias - bn_running_mean * bn_weight / std

    return weight, bias


def load_task_head_weights(ttnn_model, task_head_tensors, key_prefix):
    conv_w = task_head_tensors[key_prefix + "0.conv.weight"]
    bn_w = task_head_tensors[key_prefix + "0.bn.weight"]
    bn_b = task_head_tensors[key_prefix + "0.bn.bias"]
    bn_rm = task_head_tensors[key_prefix + "0.bn.running_mean"]
    bn_rv = task_head_tensors[key_prefix + "0.bn.running_var"]

    conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(conv_w, bn_w, bn_b, bn_rm, bn_rv)
    ttnn_model.conv1_weight = ttnn.from_torch(conv1_weight)
    ttnn_model.conv1_bias = ttnn.from_torch(conv1_bias.reshape(1, 1, 1, -1))

    conv2_w = task_head_tensors[key_prefix + "1.weight"]
    conv2_b = task_head_tensors[key_prefix + "1.bias"]
    ttnn_model.conv2_weight = ttnn.from_torch(conv2_w)
    ttnn_model.conv2_bias = ttnn.from_torch(conv2_b.reshape(1, 1, 1, -1))


def torch_load_weights(torch_model, weight_path):
    ckpt = torch.load(weight_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    key = "model.head.task_heads.0.reg."
    task_heads_0 = {k: v for k, v in state.items() if key in k}
    with torch.no_grad():
        torch_model.net[0].conv.weight = torch.nn.Parameter(task_heads_0[key + "0.conv.weight"])
        torch_model.net[0].bn.weight = torch.nn.Parameter(task_heads_0[key + "0.bn.weight"])
        torch_model.net[0].bn.bias = torch.nn.Parameter(task_heads_0[key + "0.bn.bias"])
        torch_model.net[0].bn.running_mean = torch.nn.Parameter(task_heads_0[key + "0.bn.running_mean"])
        torch_model.net[0].bn.running_var = torch.nn.Parameter(task_heads_0[key + "0.bn.running_var"])

        torch_model.net[1].weight = torch.nn.Parameter(task_heads_0[key + "1.weight"])
        torch_model.net[1].bias = torch.nn.Parameter(task_heads_0[key + "1.bias"])
