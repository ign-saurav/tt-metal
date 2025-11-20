import torch
import ttnn


def fold_batch_norm2d_into_conv2d(conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_eps=1e-05):
    """Fold BatchNorm2d parameters into Conv2d weights and bias"""
    # Compute folded weights and bias
    std = torch.sqrt(bn_running_var + bn_eps)
    weight = conv_weight * (bn_weight / std).reshape(-1, 1, 1, 1)
    bias = bn_bias - bn_running_mean * bn_weight / std

    return weight, bias


def load_weights(ttnn_model, weight_path):
    ckpt = torch.load(weight_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    key = "model.head.task_heads.0.reg."
    task_heads_0 = {k: v for k, v in state.items() if key in k}
    conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(
        task_heads_0[key + "0.conv.weight"],
        task_heads_0[key + "0.bn.weight"],
        task_heads_0[key + "0.bn.bias"],
        task_heads_0[key + "0.bn.running_mean"],
        task_heads_0[key + "0.bn.running_var"],
    )
    ttnn_model.conv1_weight = ttnn.from_torch(conv1_weight)
    ttnn_model.conv1_bias = ttnn.from_torch(conv1_bias.reshape(1, 1, 1, -1))
    ttnn_model.conv2_weight = ttnn.from_torch(task_heads_0[key + "1.weight"])
    ttnn_model.conv2_bias = ttnn.from_torch(task_heads_0[key + "1.bias"].reshape(1, 1, 1, -1))
    print(task_heads_0.keys())

    # for name, tensor in state.items():
    #     # print(name, tensor.shape)
    #     if "model.head.task_heads.0.reg." in name:
    #         # ttnn_model.conv1_weight = ttnn.from_torch(tensor)
    #         print(name, tensor.shape)


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
