import torch
import ttnn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d


def conv_bn_to_params(conv, bn, mesh_mapper):
    if bn is None:
        weight = conv.weight.detach().clone().contiguous()
        bias = conv.bias.detach().clone().contiguous() if conv.bias is not None else torch.zeros(conv.out_channels)
    else:
        weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)

    return {
        "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
    }


def create_custom_mesh_preprocessor(mesh_mapper):
    """Return a custom preprocessor closure with mesh_mapper captured."""

    def custom_preprocessor(model, name, *, ttnn_module_args=None, convert_to_ttnn=True):
        parameters = {}

        if isinstance(model, torch.nn.Conv2d) or isinstance(model, torch.nn.BatchNorm2d):
            # Skip here — handled in parent scope
            return {}

        elif isinstance(model, torch.nn.Module):
            children = list(model.named_children())
            i = 0
            while i < len(children):
                child_name, child = children[i]

                # Detect Conv + BN pair
                if isinstance(child, torch.nn.Conv2d):
                    next_bn = None
                    if i + 1 < len(children):
                        next_name, next_child = children[i + 1]
                        if isinstance(next_child, torch.nn.BatchNorm2d):
                            next_bn = next_child
                            i += 1  # skip BN

                    params = conv_bn_to_params(child, next_bn, mesh_mapper)
                    parameters[child_name] = params

                else:
                    # Recurse
                    subparams = custom_preprocessor(
                        child,
                        f"{name}.{child_name}" if name else child_name,
                        ttnn_module_args=ttnn_module_args,
                        convert_to_ttnn=convert_to_ttnn,
                    )
                    if subparams:
                        parameters[child_name] = subparams

                i += 1

        return parameters

    return custom_preprocessor
