"""Inherited from `https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/models/dense_heads/centerpoint_head.py`"""  # noqa
# Copyright (c) OpenMMLab. All rights reserved.
# `https://huggingface.co/spaces/dineshreddy/WALT/blob/main/mmdet/models/builder.py`
# https://github.com/open-mmlab/mmcv/blob/v1.7.1/mmcv/utils/registry.py
from typing import Optional

# import numba
import numpy as np

# from mmcv.ops import nms, nms_rotated
from torch import Tensor

# Copyright (c) OpenMMLab. All rights reserved.
import copy


# from mmcv.runner import BaseModule, force_fp32
from torch import nn

# from mmdet3d.core.post_processing import nms_bev
# from mmdet3d.models import builder
# from mmdet3d.models.utils import clip_sigmoid
# from mmdet.core import multi_apply
# from mmdet.core import build_bbox_coder, multi_apply
# from ..builder import HEADS, build_loss


# from mmcv.utils import Registry, build_from_cfg

# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import warnings
from functools import partial
from typing import Any, Dict, Optional

# from .misc import deprecated_api_warning, is_seq_of


def build_from_cfg(cfg: Dict, registry: "Registry", default_args: Optional[Dict] = None) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='Resnet'), MODELS)
        >>> # Returns an instantiated object
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='resnet50'), MODELS)
        >>> # Return a result of the calling function

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError('`cfg` or `default_args` must contain the key "type", ' f"but got {cfg}\n{default_args}")
    if not isinstance(registry, Registry):
        raise TypeError("registry must be an mmcv.Registry object, " f"but got {type(registry)}")
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError("default_args must be a dict or None, " f"but got {type(default_args)}")

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f"{obj_cls.__name__}: {e}")


class Registry:
    """A registry to map strings to classes or functions.

    Registered object could be built from registry. Meanwhile, registered
    functions could be called from registry.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = MODELS.build(dict(type='resnet50'))

    Please refer to
    https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html for
    advanced usage.

    Args:
        name (str): Registry name.
        build_func(func, optional): Build function to construct instance from
            Registry, func:`build_from_cfg` is used if neither ``parent`` or
            ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Default: None.
        parent (Registry, optional): Parent registry. The class registered in
            children registry could be built from parent. Default: None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Default: None.
    """

    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope() if scope is None else scope

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + f"(name={self._name}, " f"items={self._module_dict})"
        return format_str

    @staticmethod
    def infer_scope():
        """Infer the scope of registry.

        The name of the package where registry is defined will be returned.

        Example:
            >>> # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.

        Returns:
            str: The inferred scope name.
        """
        # We access the caller using inspect.currentframe() instead of
        # inspect.stack() for performance reasons. See details in PR #1844
        frame = inspect.currentframe()
        # get the frame where `infer_scope()` is called
        infer_scope_caller = frame.f_back.f_back
        filename = inspect.getmodule(infer_scope_caller).__name__
        split_filename = filename.split(".")
        return split_filename[0]

    @staticmethod
    def split_scope_key(key):
        """Split scope and key.

        The first scope will be split from key.

        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'

        Return:
            tuple[str | None, str]: The former element is the first scope of
            the key, which can be ``None``. The latter is the remaining key.
        """
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            # get from self._children
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # goto root
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        """Add children for a registry.

        The ``registry`` will be added as children based on its scope.
        The parent registry could build objects from children registry.

        Example:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(type='mmdet.ResNet'))
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert registry.scope not in self.children, f"scope {registry.scope} exists in {self.name} registry"
        self.children[registry.scope] = registry

    # @deprecated_api_warning(name_dict=dict(module_class='module'))
    def _register_module(self, module, module_name=None, force=False):
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError("module must be a class or a function, " f"but got {type(module)}")

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered " f"in {self.name}")
            self._module_dict[name] = module

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn(
            "The old API of register_module(module, force=False) "
            "is deprecated and will be removed, please use the new API "
            "register_module(name=None, force=False, module=None) instead.",
            DeprecationWarning,
        )
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class or function to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")
        # NOTE: This is a walkaround to be compatible with the old api,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):  # or is_seq_of(name, str)):
            raise TypeError(
                "name must be either of None, an instance of str or a sequence" f"  of str, but got {type(name)}"
            )

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register


BACKBONES = Registry("backbone")
NECKS = Registry("neck")
HEADS = Registry("head")


def build(cfg, registry, default_args=None):
    """Build a module.
    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def circle_nms(dets: Tensor, thresh: float, post_max_size: int = 83) -> Tensor:
    """Circular NMS.

    An object is only counted as positive if no other center with a higher
    confidence exists within a radius r using a bird-eye view distance metric.

    Args:
        dets (Tensor): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int): Max number of prediction to be kept.
            Defaults to 83.

    Returns:
        Tensor: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1

    if post_max_size < len(keep):
        return keep[:post_max_size]

    return keep


# https://github.com/open-mmlab/mmcv/blob/v1.7.1/mmcv/runner/base_module.py
from abc import ABCMeta
from collections import defaultdict


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

    - ``init_cfg``: the config to control the initialization.
    - ``init_weights``: The function of parameter initialization and recording
      initialization information.
    - ``_params_init_info``: Used to track the parameter initialization
      information. This attribute only exists during executing the
      ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, init_cfg: Optional[dict] = None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super().__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

        # Backward compatibility in derived classes
        # if pretrained is not None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #         key, please consider using init_cfg')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @property
    def is_init(self) -> bool:
        return self._is_init

    def init_weights(self) -> None:
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, "_params_init_info"):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info: defaultdict = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param]["init_info"] = (
                    f"The value is the same before and "
                    f"after calling `init_weights` "
                    f"of {self.__class__.__name__} "
                )
                self._params_init_info[param]["tmp_mean_value"] = param.data.mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        # logger_names = list(logger_initialized.keys())
        # logger_name = logger_names[0] if logger_names else 'mmcv'

        from ..cnn import initialize
        from ..cnn.utils.weight_init import update_init_info

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                # print_log(
                # f'initialize {module_name} with init_cfg {self.init_cfg}',
                # logger=logger_name)
                # )
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if self.init_cfg["type"] == "Pretrained":
                        return

            for m in self.children():
                if hasattr(m, "init_weights"):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m, init_info=f"Initialized by " f"user-defined `init_weights`" f" in {m.__class__.__name__} "
                    )

            self._is_init = True
        else:
            warnings.warn(f"init_weights of {self.__class__.__name__} has " f"been called more than once.")

        if is_top_level_module:
            # self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info

    # @master_only
    # def _dump_init_info(self, logger_name: str) -> None:
    #     """Dump the initialization information to a file named
    #     `initialization.log.json` in workdir.

    #     Args:
    #         logger_name (str): The name of logger.
    #     """

    #     # logger = get_logger(logger_name)

    #     with_file_handler = False
    #     # dump the information to the logger file if there is a `FileHandler`
    #     for handler in logger.handlers:
    #         if isinstance(handler, FileHandler):
    #             handler.stream.write(
    #                 'Name of parameter - Initialization information\n')
    #             for name, param in self.named_parameters():
    #                 handler.stream.write(
    #                     f'\n{name} - {param.shape}: '
    #                     f"\n{self._params_init_info[param]['init_info']} \n")
    #             handler.stream.flush()
    #             with_file_handler = True
    # if not with_file_handler:
    #     for name, param in self.named_parameters():
    #         print_log(
    #             f'\n{name} - {param.shape}: '
    #             f"\n{self._params_init_info[param]['init_info']} \n ",
    #             logger=logger_name)

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f"\ninit_cfg={self.init_cfg}"
        return s


# https://github.com/open-mmlab/mmcv/blob/v1.7.1/mmcv/cnn/bricks/conv_module.py

# class ConvModule(nn.Module):
#     """A conv block that bundles conv/norm/activation layers.

#     This block simplifies the usage of convolution layers, which are commonly
#     used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
#     It is based upon three build methods: `build_conv_layer()`,
#     `build_norm_layer()` and `build_activation_layer()`.

#     Besides, we add some additional features in this module.
#     1. Automatically set `bias` of the conv layer.
#     2. Spectral norm is supported.
#     3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
#     supports zero and circular padding, and we add "reflect" padding mode.

#     Args:
#         in_channels (int): Number of channels in the input feature map.
#             Same as that in ``nn._ConvNd``.
#         out_channels (int): Number of channels produced by the convolution.
#             Same as that in ``nn._ConvNd``.
#         kernel_size (int | tuple[int]): Size of the convolving kernel.
#             Same as that in ``nn._ConvNd``.
#         stride (int | tuple[int]): Stride of the convolution.
#             Same as that in ``nn._ConvNd``.
#         padding (int | tuple[int]): Zero-padding added to both sides of
#             the input. Same as that in ``nn._ConvNd``.
#         dilation (int | tuple[int]): Spacing between kernel elements.
#             Same as that in ``nn._ConvNd``.
#         groups (int): Number of blocked connections from input channels to
#             output channels. Same as that in ``nn._ConvNd``.
#         bias (bool | str): If specified as `auto`, it will be decided by the
#             norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
#             False. Default: "auto".
#         conv_cfg (dict): Config dict for convolution layer. Default: None,
#             which means using conv2d.
#         norm_cfg (dict): Config dict for normalization layer. Default: None.
#         act_cfg (dict): Config dict for activation layer.
#             Default: dict(type='ReLU').
#         inplace (bool): Whether to use inplace mode for activation.
#             Default: True.
#         with_spectral_norm (bool): Whether use spectral norm in conv module.
#             Default: False.
#         padding_mode (str): If the `padding_mode` has not been supported by
#             current `Conv2d` in PyTorch, we will use our own padding layer
#             instead. Currently, we support ['zeros', 'circular'] with official
#             implementation and ['reflect'] with our own implementation.
#             Default: 'zeros'.
#         order (tuple[str]): The order of conv/norm/activation layers. It is a
#             sequence of "conv", "norm" and "act". Common examples are
#             ("conv", "norm", "act") and ("act", "conv", "norm").
#             Default: ('conv', 'norm', 'act').
#     """

#     _abbr_ = 'conv_block'

#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: Union[int, Tuple[int, int]],
#                  stride: Union[int, Tuple[int, int]] = 1,
#                  padding: Union[int, Tuple[int, int]] = 0,
#                  dilation: Union[int, Tuple[int, int]] = 1,
#                  groups: int = 1,
#                  bias: Union[bool, str] = 'auto',
#                  conv_cfg: Optional[Dict] = None,
#                  norm_cfg: Optional[Dict] = None,
#                  act_cfg: Optional[Dict] = dict(type='ReLU'),
#                  inplace: bool = True,
#                  with_spectral_norm: bool = False,
#                  padding_mode: str = 'zeros',
#                  order: tuple = ('conv', 'norm', 'act')):
#         super().__init__()
#         assert conv_cfg is None or isinstance(conv_cfg, dict)
#         assert norm_cfg is None or isinstance(norm_cfg, dict)
#         assert act_cfg is None or isinstance(act_cfg, dict)
#         official_padding_mode = ['zeros', 'circular']
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.act_cfg = act_cfg
#         self.inplace = inplace
#         self.with_spectral_norm = with_spectral_norm
#         self.with_explicit_padding = padding_mode not in official_padding_mode
#         self.order = order
#         assert isinstance(self.order, tuple) and len(self.order) == 3
#         assert set(order) == {'conv', 'norm', 'act'}

#         self.with_norm = norm_cfg is not None
#         self.with_activation = act_cfg is not None
#         # if the conv layer is before a norm layer, bias is unnecessary.
#         if bias == 'auto':
#             bias = not self.with_norm
#         self.with_bias = bias

#         if self.with_explicit_padding:
#             pad_cfg = dict(type=padding_mode)
#             self.padding_layer = build_padding_layer(pad_cfg, padding)

#         # reset padding to 0 for conv module
#         conv_padding = 0 if self.with_explicit_padding else padding
#         # build convolution layer
#         self.conv = build_conv_layer(
#             conv_cfg,
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=conv_padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias)
#         # export the attributes of self.conv to a higher level for convenience
#         self.in_channels = self.conv.in_channels
#         self.out_channels = self.conv.out_channels
#         self.kernel_size = self.conv.kernel_size
#         self.stride = self.conv.stride
#         self.padding = padding
#         self.dilation = self.conv.dilation
#         self.transposed = self.conv.transposed
#         self.output_padding = self.conv.output_padding
#         self.groups = self.conv.groups

#         if self.with_spectral_norm:
#             self.conv = nn.utils.spectral_norm(self.conv)

#         # build normalization layers
#         if self.with_norm:
#             # norm layer is after conv layer
#             if order.index('norm') > order.index('conv'):
#                 norm_channels = out_channels
#             else:
#                 norm_channels = in_channels
#             self.norm_name, norm = build_norm_layer(
#                 norm_cfg, norm_channels)  # type: ignore
#             self.add_module(self.norm_name, norm)
#             if self.with_bias:
#                 if isinstance(norm, (_BatchNorm, _InstanceNorm)):
#                     warnings.warn(
#                         'Unnecessary conv bias before batch/instance norm')
#         else:
#             self.norm_name = None  # type: ignore

#         # build activation layer
#         if self.with_activation:
#             act_cfg_ = act_cfg.copy()  # type: ignore
#             # nn.Tanh has no 'inplace' argument
#             if act_cfg_['type'] not in [
#                     'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
#             ]:
#                 act_cfg_.setdefault('inplace', inplace)
#             self.activate = build_activation_layer(act_cfg_)

#         # Use msra init by default
#         self.init_weights()

#     @property
#     def norm(self):
#         if self.norm_name:
#             return getattr(self, self.norm_name)
#         else:
#             return None

#     def init_weights(self):
#         # 1. It is mainly for customized conv layers with their own
#         #    initialization manners by calling their own ``init_weights()``,
#         #    and we do not want ConvModule to override the initialization.
#         # 2. For customized conv layers without their own initialization
#         #    manners (that is, they don't have their own ``init_weights()``)
#         #    and PyTorch's conv layers, they will be initialized by
#         #    this method with default ``kaiming_init``.
#         # Note: For PyTorch's conv layers, they will be overwritten by our
#         #    initialization implementation using default ``kaiming_init``.
#         if not hasattr(self.conv, 'init_weights'):
#             if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
#                 nonlinearity = 'leaky_relu'
#                 a = self.act_cfg.get('negative_slope', 0.01)
#             else:
#                 nonlinearity = 'relu'
#                 a = 0
#             kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
#         if self.with_norm:
#             constant_init(self.norm, 1, bias=0)


#     def forward(self,
#                 x: torch.Tensor,
#                 activate: bool = True,
#                 norm: bool = True) -> torch.Tensor:
#         for layer in self.order:
#             if layer == 'conv':
#                 if self.with_explicit_padding:
#                     x = self.padding_layer(x)
#                 x = self.conv(x)
#             elif layer == 'norm' and norm and self.with_norm:
#                 x = self.norm(x)
#             elif layer == 'act' and activate and self.with_activation:
#                 x = self.activate(x)
#         return x
class CenterHead(BaseModule):
    """CenterHead for CenterPoint.

    Args:
        in_channels (list[int] | int, optional): Channels of the input
            feature map. Default: [128].
        tasks (list[dict], optional): Task information including class number
            and class names. Default: None.
        train_cfg (dict, optional): Train-time configs. Default: None.
        test_cfg (dict, optional): Test-time configs. Default: None.
        bbox_coder (dict, optional): Bbox coder configs. Default: None.
        common_heads (dict, optional): Conv information for common heads.
            Default: dict().
        loss_cls (dict, optional): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict, optional): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict, optional): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int, optional): Output channels for share_conv
            layer. Default: 64.
        num_heatmap_convs (int, optional): Number of conv layers for heatmap
            conv layer. Default: 2.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(
        self,
        in_channels=[128],
        tasks=None,
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
        common_heads=dict(),
        #  loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        #  loss_bbox=dict(
        #  type='L1Loss', reduction='none', loss_weight=0.25),
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
        share_conv_channel=64,
        num_heatmap_convs=2,
        conv_cfg=dict(type="Conv2d"),
        norm_cfg=dict(type="BN2d"),
        bias="auto",
        norm_bbox=True,
        init_cfg=None,
    ):
        assert init_cfg is None, "To prevent abnormal initialization " "behavior, init_cfg is not allowed to be set"
        super(CenterHead, self).__init__(init_cfg=init_cfg)

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox

        # self.loss_cls = build_loss(loss_cls)
        # self.loss_bbox = build_loss(loss_bbox)
        # self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_anchor_per_locs = [n for n in num_classes]
        self.fp16_enabled = False

        # a shared convolution
        # self.shared_conv = ConvModule(
        #     in_channels,
        #     share_conv_channel,
        #     kernel_size=3,
        #     padding=1,
        #     conv_cfg=conv_cfg,
        #     norm_cfg=norm_cfg,
        #     bias=bias)

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
            self.task_heads.append(build_head(separate_head))

        self.with_velocity = "vel" in common_heads.keys()

    # https://huggingface.co/spaces/dineshreddy/WALT/blob/main/mmdet/core/utils/misc.py
    def multi_apply(func, *args, **kwargs):
        """Apply function to a list of arguments.
        Note:
            This function applies the ``func`` to multiple inputs and
            map the multiple outputs of the ``func`` into different
            list. Each list contains the same type of outputs corresponding
            to different inputs.
        Args:
            func (Function): A function that will be applied to a list of
                arguments
        Returns:
            tuple(list): A tuple containing multiple list, each list contains \
                a kind of returned results by the function
        """
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)
        return tuple(map(list, zip(*map_results)))

    # def forward_single(self, x):
    #     """Forward function for CenterPoint.

    #     Args:
    #         x (torch.Tensor): Input feature map with the shape of
    #             [B, 512, 128, 128].

    #     Returns:
    #         list[dict]: Output results for tasks.
    #     """
    #     ret_dicts = []

    #     x = self.shared_conv(x)

    #     for task in self.task_heads:
    #         ret_dicts.append(task(x))

    #     return ret_dicts

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return self.multi_apply(self.forward_single, feats)

    # def _gather_feat(self, feat, ind, mask=None):
    #     """Gather feature map.

    #     Given feature map and index, return indexed feature map.

    #     Args:
    #         feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
    #         ind (torch.Tensor): Index of the ground truth boxes with the
    #             shape of [B, max_obj].
    #         mask (torch.Tensor, optional): Mask of the feature map with the
    #             shape of [B, max_obj]. Default: None.

    #     Returns:
    #         torch.Tensor: Feature map after gathering with the shape
    #             of [B, max_obj, 10].
    #     """
    #     dim = feat.size(2)
    #     ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    #     feat = feat.gather(1, ind)
    #     if mask is not None:
    #         mask = mask.unsqueeze(2).expand_as(feat)
    #         feat = feat[mask]
    #         feat = feat.view(-1, dim)
    #     return feat

    # def get_targets(self, gt_bboxes_3d, gt_labels_3d):
    #     """Generate targets.

    #     How each output is transformed:

    #         Each nested list is transposed so that all same-index elements in
    #         each sub-list (1, ..., N) become the new sub-lists.
    #             [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
    #             ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

    #         The new transposed nested list is converted into a list of N
    #         tensors generated by concatenating tensors in the new sub-lists.
    #             [ tensor0, tensor1, tensor2, ... ]

    #     Args:
    #         gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
    #             truth gt boxes.
    #         gt_labels_3d (list[torch.Tensor]): Labels of boxes.

    #     Returns:
    #         Returns:
    #             tuple[list[torch.Tensor]]: Tuple of target including
    #                 the following results in order.

    #                 - list[torch.Tensor]: Heatmap scores.
    #                 - list[torch.Tensor]: Ground truth boxes.
    #                 - list[torch.Tensor]: Indexes indicating the
    #                     position of the valid boxes.
    #                 - list[torch.Tensor]: Masks indicating which
    #                     boxes are valid.
    #     """
    #     heatmaps, anno_boxes, inds, masks = multi_apply(
    #         self.get_targets_single, gt_bboxes_3d, gt_labels_3d)
    #     # Transpose heatmaps
    #     heatmaps = list(map(list, zip(*heatmaps)))
    #     heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
    #     # Transpose anno_boxes
    #     anno_boxes = list(map(list, zip(*anno_boxes)))
    #     anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
    #     # Transpose inds
    #     inds = list(map(list, zip(*inds)))
    #     inds = [torch.stack(inds_) for inds_ in inds]
    #     # Transpose inds
    #     masks = list(map(list, zip(*masks)))
    #     masks = [torch.stack(masks_) for masks_ in masks]
    #     return heatmaps, anno_boxes, inds, masks

    # def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
    #     """Generate training targets for a single sample.

    #     Args:
    #         gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
    #         gt_labels_3d (torch.Tensor): Labels of boxes.

    #     Returns:
    #         tuple[list[torch.Tensor]]: Tuple of target including
    #             the following results in order.

    #             - list[torch.Tensor]: Heatmap scores.
    #             - list[torch.Tensor]: Ground truth boxes.
    #             - list[torch.Tensor]: Indexes indicating the position
    #                 of the valid boxes.
    #             - list[torch.Tensor]: Masks indicating which boxes
    #                 are valid.
    #     """
    #     device = gt_labels_3d.device
    #     gt_bboxes_3d = torch.cat(
    #         (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
    #         dim=1).to(device)
    #     max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
    #     grid_size = torch.tensor(self.train_cfg['grid_size'])
    #     pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
    #     voxel_size = torch.tensor(self.train_cfg['voxel_size'])

    #     feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

    #     # reorganize the gt_dict by tasks
    #     task_masks = []
    #     flag = 0
    #     for class_name in self.class_names:
    #         task_masks.append([
    #             torch.where(gt_labels_3d == class_name.index(i) + flag)
    #             for i in class_name
    #         ])
    #         flag += len(class_name)

    #     task_boxes = []
    #     task_classes = []
    #     flag2 = 0
    #     for idx, mask in enumerate(task_masks):
    #         task_box = []
    #         task_class = []
    #         for m in mask:
    #             task_box.append(gt_bboxes_3d[m])
    #             # 0 is background for each task, so we need to add 1 here.
    #             task_class.append(gt_labels_3d[m] + 1 - flag2)
    #         task_boxes.append(torch.cat(task_box, axis=0).to(device))
    #         task_classes.append(torch.cat(task_class).long().to(device))
    #         flag2 += len(mask)
    #     draw_gaussian = draw_heatmap_gaussian
    #     heatmaps, anno_boxes, inds, masks = [], [], [], []

    #     for idx, task_head in enumerate(self.task_heads):
    #         heatmap = gt_bboxes_3d.new_zeros(
    #             (len(self.class_names[idx]), feature_map_size[1],
    #              feature_map_size[0]))

    #         if self.with_velocity:
    #             anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
    #                                               dtype=torch.float32)
    #         else:
    #             anno_box = gt_bboxes_3d.new_zeros((max_objs, 8),
    #                                               dtype=torch.float32)

    #         ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
    #         mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

    #         num_objs = min(task_boxes[idx].shape[0], max_objs)

    #         for k in range(num_objs):
    #             cls_id = task_classes[idx][k] - 1

    #             width = task_boxes[idx][k][3]
    #             length = task_boxes[idx][k][4]
    #             width = width / voxel_size[0] / self.train_cfg[
    #                 'out_size_factor']
    #             length = length / voxel_size[1] / self.train_cfg[
    #                 'out_size_factor']

    #             if width > 0 and length > 0:
    #                 radius = gaussian_radius(
    #                     (length, width),
    #                     min_overlap=self.train_cfg['gaussian_overlap'])
    #                 radius = max(self.train_cfg['min_radius'], int(radius))

    #                 # be really careful for the coordinate system of
    #                 # your box annotation.
    #                 x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
    #                     1], task_boxes[idx][k][2]

    #                 coor_x = (
    #                     x - pc_range[0]
    #                 ) / voxel_size[0] / self.train_cfg['out_size_factor']
    #                 coor_y = (
    #                     y - pc_range[1]
    #                 ) / voxel_size[1] / self.train_cfg['out_size_factor']

    #                 center = torch.tensor([coor_x, coor_y],
    #                                       dtype=torch.float32,
    #                                       device=device)
    #                 center_int = center.to(torch.int32)

    #                 # throw out not in range objects to avoid out of array
    #                 # area when creating the heatmap
    #                 if not (0 <= center_int[0] < feature_map_size[0]
    #                         and 0 <= center_int[1] < feature_map_size[1]):
    #                     continue

    #                 draw_gaussian(heatmap[cls_id], center_int, radius)

    #                 new_idx = k
    #                 x, y = center_int[0], center_int[1]

    #                 assert (y * feature_map_size[0] + x <
    #                         feature_map_size[0] * feature_map_size[1])

    #                 ind[new_idx] = y * feature_map_size[0] + x
    #                 mask[new_idx] = 1
    #                 # TODO: support other outdoor dataset
    #                 rot = task_boxes[idx][k][6]
    #                 box_dim = task_boxes[idx][k][3:6]
    #                 if self.norm_bbox:
    #                     box_dim = box_dim.log()
    #                 if self.with_velocity:
    #                     vx, vy = task_boxes[idx][k][7:]
    #                     anno_box[new_idx] = torch.cat([
    #                         center - torch.tensor([x, y], device=device),
    #                         z.unsqueeze(0), box_dim,
    #                         torch.sin(rot).unsqueeze(0),
    #                         torch.cos(rot).unsqueeze(0),
    #                         vx.unsqueeze(0),
    #                         vy.unsqueeze(0)
    #                     ])
    #                 else:
    #                     anno_box[new_idx] = torch.cat([
    #                         center - torch.tensor([x, y], device=device),
    #                         z.unsqueeze(0), box_dim,
    #                         torch.sin(rot).unsqueeze(0),
    #                         torch.cos(rot).unsqueeze(0)
    #                     ])

    #         heatmaps.append(heatmap)
    #         anno_boxes.append(anno_box)
    #         masks.append(mask)
    #         inds.append(ind)
    #     return heatmaps, anno_boxes, inds, masks

    # @force_fp32(apply_to=('preds_dicts'))
    # def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
    #     """Loss function for CenterHead.

    #     Args:
    #         gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
    #             truth gt boxes.
    #         gt_labels_3d (list[torch.Tensor]): Labels of boxes.
    #         preds_dicts (dict): Output of forward function.

    #     Returns:
    #         dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
    #     """
    #     heatmaps, anno_boxes, inds, masks = self.get_targets(
    #         gt_bboxes_3d, gt_labels_3d)
    #     loss_dict = dict()
    #     for task_id, preds_dict in enumerate(preds_dicts):
    #         # heatmap focal loss
    #         preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
    #         num_pos = heatmaps[task_id].eq(1).float().sum().item()
    #         loss_heatmap = self.loss_cls(
    #             preds_dict[0]['heatmap'],
    #             heatmaps[task_id],
    #             avg_factor=max(num_pos, 1))
    #         target_box = anno_boxes[task_id]
    #         # reconstruct the anno_box from multiple reg heads
    #         if self.with_velocity:
    #             preds_dict[0]['anno_box'] = torch.cat(
    #                 (preds_dict[0]['reg'], preds_dict[0]['height'],
    #                  preds_dict[0]['dim'], preds_dict[0]['rot'],
    #                  preds_dict[0]['vel']),
    #                 dim=1)
    #         else:
    #             preds_dict[0]['anno_box'] = torch.cat(
    #                 (preds_dict[0]['reg'], preds_dict[0]['height'],
    #                  preds_dict[0]['dim'], preds_dict[0]['rot']),
    #                 dim=1)

    #         # Regression loss for dimension, offset, height, rotation
    #         ind = inds[task_id]
    #         num = masks[task_id].float().sum()
    #         pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
    #         pred = pred.view(pred.size(0), -1, pred.size(3))
    #         pred = self._gather_feat(pred, ind)
    #         mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
    #         isnotnan = (~torch.isnan(target_box)).float()
    #         mask *= isnotnan

    #         code_weights = self.train_cfg.get('code_weights', None)
    #         bbox_weights = mask * mask.new_tensor(code_weights)
    #         loss_bbox = self.loss_bbox(
    #             pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
    #         loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
    #         loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
    #     return loss_dict

    # def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
    #     """Generate bboxes from bbox head predictions.

    #     Args:
    #         preds_dicts (tuple[list[dict]]): Prediction results.
    #         img_metas (list[dict]): Point cloud and image's meta info.

    #     Returns:
    #         list[dict]: Decoded bbox, scores and labels after nms.
    #     """
    #     rets = []
    #     for task_id, preds_dict in enumerate(preds_dicts):
    #         num_class_with_bg = self.num_classes[task_id]
    #         batch_size = preds_dict[0]['heatmap'].shape[0]
    #         batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

    #         batch_reg = preds_dict[0]['reg']
    #         batch_hei = preds_dict[0]['height']

    #         if self.norm_bbox:
    #             batch_dim = torch.exp(preds_dict[0]['dim'])
    #         else:
    #             batch_dim = preds_dict[0]['dim']

    #         batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
    #         batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

    #         if 'vel' in preds_dict[0]:
    #             batch_vel = preds_dict[0]['vel']
    #         else:
    #             batch_vel = None
    #         temp = self.bbox_coder.decode(
    #             batch_heatmap,
    #             batch_rots,
    #             batch_rotc,
    #             batch_hei,
    #             batch_dim,
    #             batch_vel,
    #             reg=batch_reg,
    #             task_id=task_id)
    #         assert self.test_cfg['nms_type'] in ['circle', 'rotate']
    #         batch_reg_preds = [box['bboxes'] for box in temp]
    #         batch_cls_preds = [box['scores'] for box in temp]
    #         batch_cls_labels = [box['labels'] for box in temp]
    #         if self.test_cfg['nms_type'] == 'circle':
    #             ret_task = []
    #             for i in range(batch_size):
    #                 boxes3d = temp[i]['bboxes']
    #                 scores = temp[i]['scores']
    #                 labels = temp[i]['labels']
    #                 centers = boxes3d[:, [0, 1]]
    #                 boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
    #                 keep = torch.tensor(
    #                     circle_nms(
    #                         boxes.detach().cpu().numpy(),
    #                         self.test_cfg['min_radius'][task_id],
    #                         post_max_size=self.test_cfg['post_max_size']),
    #                     dtype=torch.long,
    #                     device=boxes.device)

    #                 boxes3d = boxes3d[keep]
    #                 scores = scores[keep]
    #                 labels = labels[keep]
    #                 ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
    #                 ret_task.append(ret)
    #             rets.append(ret_task)
    #         else:
    #             rets.append(
    #                 self.get_task_detections(num_class_with_bg,
    #                                          batch_cls_preds, batch_reg_preds,
    #                                          batch_cls_labels, img_metas))

    #     # Merge branches results
    #     num_samples = len(rets[0])

    #     ret_list = []
    #     for i in range(num_samples):
    #         for k in rets[0][i].keys():
    #             if k == 'bboxes':
    #                 bboxes = torch.cat([ret[i][k] for ret in rets])
    #                 bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
    #                 bboxes = img_metas[i]['box_type_3d'](
    #                     bboxes, self.bbox_coder.code_size)
    #             elif k == 'scores':
    #                 scores = torch.cat([ret[i][k] for ret in rets])
    #             elif k == 'labels':
    #                 flag = 0
    #                 for j, num_class in enumerate(self.num_classes):
    #                     rets[j][i][k] += flag
    #                     flag += num_class
    #                 labels = torch.cat([ret[i][k].int() for ret in rets])
    #         ret_list.append([bboxes, scores, labels])
    #     return ret_list

    # def get_task_detections(self, num_class_with_bg, batch_cls_preds,
    #                         batch_reg_preds, batch_cls_labels, img_metas):
    #     """Rotate nms for each task.

    #     Args:
    #         num_class_with_bg (int): Number of classes for the current task.
    #         batch_cls_preds (list[torch.Tensor]): Prediction score with the
    #             shape of [N].
    #         batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
    #             shape of [N, 9].
    #         batch_cls_labels (list[torch.Tensor]): Prediction label with the
    #             shape of [N].
    #         img_metas (list[dict]): Meta information of each sample.

    #     Returns:
    #         list[dict[str: torch.Tensor]]: contains the following keys:

    #             -bboxes (torch.Tensor): Prediction bboxes after nms with the
    #                 shape of [N, 9].
    #             -scores (torch.Tensor): Prediction scores after nms with the
    #                 shape of [N].
    #             -labels (torch.Tensor): Prediction labels after nms with the
    #                 shape of [N].
    #     """
    #     predictions_dicts = []
    #     post_center_range = self.test_cfg['post_center_limit_range']
    #     if len(post_center_range) > 0:
    #         post_center_range = torch.tensor(
    #             post_center_range,
    #             dtype=batch_reg_preds[0].dtype,
    #             device=batch_reg_preds[0].device)

    #     for i, (box_preds, cls_preds, cls_labels) in enumerate(
    #             zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

    #         # Apply NMS in bird eye view

    #         # get the highest score per prediction, then apply nms
    #         # to remove overlapped box.
    #         if num_class_with_bg == 1:
    #             top_scores = cls_preds.squeeze(-1)
    #             top_labels = torch.zeros(
    #                 cls_preds.shape[0],
    #                 device=cls_preds.device,
    #                 dtype=torch.long)

    #         else:
    #             top_labels = cls_labels.long()
    #             top_scores = cls_preds.squeeze(-1)

    #         if self.test_cfg['score_threshold'] > 0.0:
    #             thresh = torch.tensor(
    #                 [self.test_cfg['score_threshold']],
    #                 device=cls_preds.device).type_as(cls_preds)
    #             top_scores_keep = top_scores >= thresh
    #             top_scores = top_scores.masked_select(top_scores_keep)

    #         if top_scores.shape[0] != 0:
    #             if self.test_cfg['score_threshold'] > 0.0:
    #                 box_preds = box_preds[top_scores_keep]
    #                 top_labels = top_labels[top_scores_keep]

    #             boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
    #                 box_preds[:, :], self.bbox_coder.code_size).bev)
    #             # the nms in 3d detection just remove overlap boxes.

    #             selected = nms_bev(
    #                 boxes_for_nms,
    #                 top_scores,
    #                 thresh=self.test_cfg['nms_thr'],
    #                 pre_max_size=self.test_cfg['pre_max_size'],
    #                 post_max_size=self.test_cfg['post_max_size'])
    #         else:
    #             selected = []

    #         # if selected is not None:
    #         selected_boxes = box_preds[selected]
    #         selected_labels = top_labels[selected]
    #         selected_scores = top_scores[selected]

    #         # finally generate predictions.
    #         if selected_boxes.shape[0] != 0:
    #             box_preds = selected_boxes
    #             scores = selected_scores
    #             label_preds = selected_labels
    #             final_box_preds = box_preds
    #             final_scores = scores
    #             final_labels = label_preds
    #             if post_center_range is not None:
    #                 mask = (final_box_preds[:, :3] >=
    #                         post_center_range[:3]).all(1)
    #                 mask &= (final_box_preds[:, :3] <=
    #                          post_center_range[3:]).all(1)
    #                 predictions_dict = dict(
    #                     bboxes=final_box_preds[mask],
    #                     scores=final_scores[mask],
    #                     labels=final_labels[mask])
    #             else:
    #                 predictions_dict = dict(
    #                     bboxes=final_box_preds,
    #                     scores=final_scores,
    #                     labels=final_labels)
    #         else:
    #             dtype = batch_reg_preds[0].dtype
    #             device = batch_reg_preds[0].device
    #             predictions_dict = dict(
    #                 bboxes=torch.zeros([0, self.bbox_coder.code_size],
    #                                    dtype=dtype,
    #                                    device=device),
    #                 scores=torch.zeros([0], dtype=dtype, device=device),
    #                 labels=torch.zeros([0],
    #                                    dtype=top_labels.dtype,
    #                                    device=device))

    #         predictions_dicts.append(predictions_dict)

    #     return predictions_dicts
