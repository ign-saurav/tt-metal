# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

##########################################################################
# Adapted from MMCV, MMSegmentation, MMDetection, and MMDetection3D
# Original sources:
# - MMDetection: https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1/mmdet3d
# - MMDetection3D: https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1/mmdet3d
# - MMSegmentation: https://github.com/open-mmlab/mmsegmentation/tree/v0.14.1/mmseg
# - MMEngine: https://github.com/open-mmlab/mmengine/blob/main/mmengine
# - MMCV: https://github.com/open-mmlab/mmcv/tree/v1.4.0/mmcv
# - MapTR: https://github.com/hustvl/MapTR/tree/main/projects/mmdet3d_plugin
# Original work Copyright (c) OpenMMLab.
# Licensed under the Apache License, Version 2.0.
##########################################################################

from __future__ import division

# Standard library imports
import bisect
import collections
import copy
import functools
import importlib
import inspect
import json
import logging
import math
import os
import os.path as osp
import pickle
import re
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from collections import abc as abc_collections
from collections.abc import Mapping, Sequence
from functools import partial
from inspect import getfullargspec
from logging import FileHandler
from types import ModuleType

# Third-party imports
import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from mmengine import Config
from packaging.version import parse
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import Sampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


def load_file(filename):
    """Load data from file (supports pickle, json, yaml).

    Args:
        filename (str): Path to file.

    Returns:
        Data loaded from file.
    """
    if filename.endswith(".pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)
    elif filename.endswith(".json"):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    elif filename.endswith((".yml", ".yaml")):
        with open(filename, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        with open(filename, "rb") as f:
            return pickle.load(f)


def master_only(func):
    """Decorator to ensure function only runs on master process."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc_collections.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    Args:
        seq (list): The list to be checked.
        expected_type (type): Expected type of list items.

    Returns:
        bool: Whether the list is valid.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    Args:
        seq (tuple): The tuple to be checked.
        expected_type (type): Expected type of tuple items.

    Returns:
        bool: Whether the tuple is valid.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

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
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

    # Filter out internal config keys
    filtered_args = {k: v for k, v in args.items() if not k.startswith("_")}

    try:
        return obj_cls(**filtered_args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f"{obj_cls.__name__}: {e}")


class Registry:
    """A registry to map strings to classes.

    Registered object could be built from registry.
    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))

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
            # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.


        Returns:
            scope (str): The inferred scope name.
        """
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
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
            scope (str, None): The first scope.
            key (str): The remaining key.
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
        self.children[registry.scope] = registry

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, " f"but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered " f"in {self.name}")
            self._module_dict[name] = module_class

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn(
            "The old API of register_module(module, force=False) "
            "is deprecated and will be removed, please use the new API "
            "register_module(name=None, force=False, module=None) instead."
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
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")
        # NOTE: This is a walkaround to be compatible with the old api,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                "name must be either of None, an instance of str or a sequence" f"  of str, but got {type(name)}"
            )

        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register


def deprecated_api_warning(name_dict, cls_name=None):
    """A decorator to check if some arguments are deprecate and try to replace
    deprecate src_arg_name to dst_arg_name.

    Args:
        name_dict(dict):
            key (str): Deprecate argument names.
            val (str): Expected argument names.

    Returns:
        func: New function.
    """

    def api_warning_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get name of the function
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f"{cls_name}.{func_name}"
            if args:
                arg_names = args_info.args[: len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            "instead"
                        )
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        assert dst_arg_name not in kwargs, (
                            f"The expected behavior is to replace "
                            f"the deprecated key `{src_arg_name}` to "
                            f"new key `{dst_arg_name}`, but got them "
                            f"in the arguments at the same time, which "
                            f"is confusing. `{src_arg_name} will be "
                            f"deprecated in the future, please "
                            f"use `{dst_arg_name}` instead."
                        )

                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            "instead"
                        )
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            # apply converted arguments to the decorated method
            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper


def assert_tensor_type(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f"{args[0].__class__.__name__} has no attribute " f"{func.__name__} for type {args[0].datatype}"
            )
        return func(*args, **kwargs)

    return wrapper


class DataContainer:
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.data)})"

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()


TORCH_VERSION = torch.__version__

logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode="w"):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            "logger should be either a logging.Logger object, str, " f'"silent" or None, but got {type(logger)}'
        )


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert "parrots" not in version_str
    version = parse(version_str)
    assert version.release, f"failed to parse version {version_str}"
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {"a": -3, "b": -2, "rc": -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f"unknown prerelease version {version.pre[0]}, " "version checking may go wrong")
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])
    return tuple(release)


def load_ext(name, funcs):
    """Load extension module."""
    if torch.__version__ != "parrots":
        try:
            # Try to import the actual extension module
            ext = importlib.import_module("mmcv._ext")
            for fun in funcs:
                if not hasattr(ext, fun):
                    warnings.warn(f"{fun} miss in module mmcv._ext")
            return ext
        except (ImportError, ModuleNotFoundError):
            mock_ext = ModuleType("_ext")
            for fun in funcs:

                def make_stub_func(func_name):
                    def stub_func(*args, **kwargs):
                        raise NotImplementedError(
                            f"{func_name} is not available. CUDA extensions are required for this operation. "
                            f"For CPU-only inference, use PyTorch fallback implementations."
                        )

                    return stub_func

                # Fix closure by binding func_name properly
                setattr(mock_ext, fun, make_stub_func(fun))
            return mock_ext
    else:
        # Parrots version - not used in CPU-only
        raise NotImplementedError("Parrots not supported in CPU-only version")


# Create ext_loader object with load_ext method
class ExtLoader:
    """Extension loader object that mimics mmcv.utils.ext_loader."""

    def load_ext(self, name, funcs):
        return load_ext(name, funcs)


ext_loader = ExtLoader()


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """CPU version of multi-scale deformable attention.

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """
    import torch.nn.functional as F

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


try:
    from addict import Dict
except ImportError:
    # Minimal Dict implementation if addict is not available
    class Dict(dict):
        """A dictionary that allows attribute-style access."""

        def __init__(self, *args, **kwargs):
            super(Dict, self).__init__(*args, **kwargs)
            for arg in args:
                if isinstance(arg, dict):
                    for k, v in arg.items():
                        self[k] = self._convert(v)
            if kwargs:
                for k, v in kwargs.items():
                    self[k] = self._convert(v)

        def _convert(self, value):
            if isinstance(value, dict) and not isinstance(value, Dict):
                return Dict(value)
            elif isinstance(value, list):
                return [self._convert(item) for item in value]
            return value

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            del self[name]


BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
DEPRECATION_KEY = "_deprecation_"
RESERVED_KEYS = ["filename", "text", "pretty_text"]


class ConfigDict(Dict):
    """A dictionary for config that raises KeyError on missing keys."""

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no " f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


# Full implementation at: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/utils/config.py
class Config(ConfigDict):
    """Config class for loading configuration files.


    This implementation supports:
    - Loading from .py files
    - Base config inheritance (via _base_)
    - Config merging
    """

    @classmethod
    def fromfile(cls, filename, use_predefined_variables=True, import_custom_modules=True):
        """Build a Config instance from a Python file.

        Args:
            filename (str): Path to the config file.
            use_predefined_variables (bool): Whether to use predefined variables.
            import_custom_modules (bool): Whether to import custom modules.

        Returns:
            Config: Config instance.

        """
        if isinstance(filename, str):
            if not osp.isabs(filename):
                filename = osp.join(os.getcwd(), filename)
            if not osp.isfile(filename):
                raise FileNotFoundError(f"Config file {filename} does not exist")
        else:
            raise TypeError("filename must be a str")

        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in [".py", ".json", ".yaml", ".yml"]:
            raise IOError("Only py/yml/yaml/json type are supported now!")

        if fileExtname == ".py":
            cfg_dict, cfg_text = cls._file2dict(filename)
        elif fileExtname in [".yml", ".yaml"]:
            try:
                import yaml
            except ImportError:
                raise ImportError("Please install pyyaml: pip install pyyaml")
            with open(filename, "r", encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f)
            cfg_text = filename + "\n"
            with open(filename, "r", encoding="utf-8") as f:
                cfg_text += f.read()
        elif fileExtname == ".json":
            with open(filename, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
            cfg_text = filename + "\n"
            with open(filename, "r", encoding="utf-8") as f:
                cfg_text += f.read()

        if cfg_dict.get("_base_", None):
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop("_base_")
            base_filename = base_filename if isinstance(base_filename, list) else [base_filename]

            cfg_dict_list = []
            cfg_text_list = []
            for f in base_filename:
                _cfg_file = f if osp.isabs(f) else osp.join(cfg_dir, f)
                _cfg_dict, _cfg_text = cls._file2dict(_cfg_file)
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict):
                    base_cfg_dict = cls._merge_a_into_b(c, base_cfg_dict)
                else:
                    base_cfg_dict = c
            cfg_dict = cls._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_text = "\n".join(cfg_text_list) + "\n" + cfg_text

        if import_custom_modules and cfg_dict.get("custom_imports", None):
            cls._import_modules_from_strings(cfg_dict["custom_imports"])

        return cls(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def _file2dict(filename):
        """Load a Python file as a dictionary.

        Args:
            filename (str): Path to the Python file.

        Returns:
            tuple: (cfg_dict, cfg_text)
        """
        import shutil
        import sys
        from importlib import import_module

        filename = osp.abspath(osp.expanduser(filename))
        if not osp.isfile(filename):
            raise FileNotFoundError(f"Config file {filename} does not exist")

        if filename.endswith(".py"):
            with tempfile.TemporaryDirectory() as temp_config_dir:
                temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=".py", delete=False)
                temp_config_name = osp.basename(temp_config_file.name)
                shutil.copyfile(filename, temp_config_file.name)
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}
                if temp_module_name in sys.modules:
                    del sys.modules[temp_module_name]
                temp_config_file.close()
                os.unlink(temp_config_file.name)
        else:
            raise IOError("Only .py files are supported")

        cfg_text = filename + "\n"
        with open(filename, "r", encoding="utf-8") as f:
            cfg_text += f.read()

        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b):
        """Merge dict a into dict b.

        Args:
            a (dict): Source dictionary.
            b (dict): Target dictionary.

        Returns:
            dict: Merged dictionary.
        """
        for k, v in a.items():
            if k in b:
                if isinstance(b[k], dict) and isinstance(v, dict):
                    b[k] = Config._merge_a_into_b(v, b[k])
                else:
                    b[k] = v
            else:
                b[k] = v
        return b

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        """Initialize Config.

        Args:
            cfg_dict (dict): Configuration dictionary.
            cfg_text (str): Configuration text.
            filename (str): Configuration filename.
        """
        super(Config, self).__init__()
        if cfg_dict is None:
            cfg_dict = dict()
        if isinstance(cfg_dict, dict):
            super(Config, self).__init__(cfg_dict)
        else:
            raise TypeError("cfg_dict must be a dict")
        self._cfg_text = cfg_text
        self._filename = filename

    def merge_from_dict(self, options):
        """Merge options into config.

        Args:
            options (dict): Options to merge.
        """
        options = self._parse_cfg_text(options)
        if "cfg_options" in options:
            options = options["cfg_options"]
        self._merge_a_into_b(options, self)

    @staticmethod
    def _parse_cfg_text(cfg_text):
        """Parse config text.

        Args:
            cfg_text (str): Config text.

        Returns:
            dict: Parsed config dictionary.
        """
        if isinstance(cfg_text, str):
            if cfg_text.startswith("{"):
                return json.loads(cfg_text)
            else:
                return eval(cfg_text)
        else:
            return cfg_text

    @staticmethod
    def _import_modules_from_strings(imports):
        """Import modules from strings.

        Args:
            imports (list): List of module strings to import.
        """
        from importlib import import_module

        if isinstance(imports, str):
            imports = [imports]
        for imp in imports:
            if not isinstance(imp, str):
                raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
            import_module(imp)


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

        - ``init_cfg``: the config to control the initialization.
        - ``init_weights``: The function of parameter
            initialization and recording initialization
            information.
        - ``_params_init_info``: Used to track the parameter
            initialization information. This attribute only
            exists during executing the ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.

    """

    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super(BaseModule, self).__init__()
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, "_params_init_info"):
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True
            for name, param in self.named_parameters():
                self._params_init_info[param]["init_info"] = (
                    f"The value is the same before and "
                    f"after calling `init_weights` "
                    f"of {self.__class__.__name__} "
                )
                self._params_init_info[param]["tmp_mean_value"] = param.data.mean()

            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else "mmcv"

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(f"initialize {module_name} with init_cfg {self.init_cfg}", logger=logger_name)
                # TODO: Call initialize(self, self.init_cfg) when mmcv.cnn is added
                if isinstance(self.init_cfg, dict):
                    if self.init_cfg.get("type") == "Pretrained":
                        return

            for m in self.children():
                if hasattr(m, "init_weights"):
                    m.init_weights()
                    # TODO: Call update_init_info when mmcv.cnn.utils.weight_init is added

            self._is_init = True
        else:
            warnings.warn(f"init_weights of {self.__class__.__name__} has " f"been called more than once.")

        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                if hasattr(sub_module, "_params_init_info"):
                    del sub_module._params_init_info

    @master_only
    def _dump_init_info(self, logger_name):
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.

        Args:
            logger_name (str): The name of logger.
        """

        logger = get_logger(logger_name)

        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write("Name of parameter - Initialization information\n")
                for name, param in self.named_parameters():
                    handler.stream.write(
                        f"\n{name} - {param.shape}: " f"\n{self._params_init_info[param]['init_info']} \n"
                    )
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                print_log(
                    f"\n{name} - {param.shape}: " f"\n{self._params_init_info[param]['init_info']} \n ",
                    logger=logger_name,
                )

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f"\ninit_cfg={self.init_cfg}"
        return s


class Sequential(BaseModule, nn.Sequential):
    """Sequential module in openmmlab.

    Args:
        init_cfg (dict, optional): Initialization config dict.

    """

    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList in openmmlab.

    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.

    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


class TransformerLayerSequence(BaseModule):
    """Base class for TransformerEncoder and TransformerDecoder in vision transformer.

    Args:
        transformerlayer (list[dict] | dict): Configs of `TransformerLayer` in
            encoder/decoder. If it is a dict, it would be repeated `num_layers` times
            to a list.
        num_layers (int): The number of `TransformerLayer`. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, transformerlayer=None, num_layers=None, init_cfg=None):
        super(TransformerLayerSequence, self).__init__(init_cfg)
        self.num_layers = num_layers
        self.layers = ModuleList()
        if transformerlayer is None:
            transformerlayer = []
        if isinstance(transformerlayer, dict):
            transformerlayer = [copy.deepcopy(transformerlayer) for _ in range(num_layers)]
        else:
            assert num_layers is None or len(transformerlayer) == num_layers, (
                f"The length of "
                f"transformerlayer {len(transformerlayer)} is "
                f"not consistent with num_layers {num_layers}."
            )
        for i, layer_cfg in enumerate(transformerlayer):
            self.layers.append(build_transformer_layer(layer_cfg))


def get_dist_info():
    """Get distributed information."""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def cast_tensor_type(inputs, src_type, dst_type):
    """Cast type of tensor."""
    if isinstance(inputs, torch.Tensor):
        return inputs.type(dst_type) if inputs.type() == src_type else inputs
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, Mapping):
        return type(inputs)({k: cast_tensor_type(v, src_type, dst_type) for k, v in inputs.items()})
    elif isinstance(inputs, (list, tuple)):
        return type(inputs)(cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def auto_fp16(apply_to=None, out_fp32=False):
    """Decorator to enable fp16 training automatically.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If inputs arguments are fp32 tensors, they will
    be converted to fp16 automatically. Arguments other than fp32 tensors are
    ignored. If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend, otherwise, original mmcv implementation will be adopted.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp32 (bool): Whether to convert the output back to fp32.

    """

    def auto_fp16_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError("@auto_fp16 can only be used to decorate the " "method of nn.Module")
            if not (hasattr(args[0], "fp16_enabled") and args[0].fp16_enabled):
                return old_func(*args, **kwargs)

            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            # NOTE: default args are not taken into consideration
            if args:
                arg_names = args_info.args[: len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value

            # apply converted arguments to the decorated method
            output = old_func(*new_args, **new_kwargs)

            # cast the results back to fp32 if necessary
            if out_fp32 and isinstance(output, torch.Tensor):
                output = output.float()
            elif out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper


def force_fp32(apply_to=None, out_fp16=False):
    """Decorator to convert input arguments to fp32 in force.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If there are some inputs that must be processed
    in fp32 mode, then this decorator can handle it. If inputs arguments are
    fp16 tensors, they will be converted to fp32 automatically. Arguments other
    than fp16 tensors are ignored. If you are using PyTorch >= 1.6,
    torch.cuda.amp is used as the backend, otherwise, original mmcv
    implementation will be adopted.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp16 (bool): Whether to convert the output back to fp16.

    """

    def force_fp32_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError("@force_fp32 can only be used to decorate the " "method of nn.Module")
            if not (hasattr(args[0], "fp16_enabled") and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            if args:
                arg_names = args_info.args[: len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value

            # apply converted arguments to the decorated method
            output = old_func(*new_args, **new_kwargs)

            # cast the results back to fp16 if necessary
            if out_fp16 and isinstance(output, torch.Tensor):
                output = output.half()
            elif out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper


def constant_init(module, val, bias=0):
    """Initialize module parameters with constant values."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"):
    """Initialize module parameters with Kaiming initialization."""
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    """Initialize module parameters with Xavier initialization."""
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """Initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def build_model_from_cfg(cfg, registry, default_args=None):
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a config
            dict or a list of config dicts. If cfg is a list, a
            the built modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.

    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


CONV_LAYERS = Registry("conv layer")
NORM_LAYERS = Registry("norm layer")
ACTIVATION_LAYERS = Registry("activation layer")
PADDING_LAYERS = Registry("padding layer")
PLUGIN_LAYERS = Registry("plugin layer")

DROPOUT_LAYERS = Registry("drop out layers")
POSITIONAL_ENCODING = Registry("position encoding")
ATTENTION = Registry("attention")
FEEDFORWARD_NETWORK = Registry("feed-forward Network")
TRANSFORMER_LAYER = Registry("transformerLayer")
TRANSFORMER_LAYER_SEQUENCE = Registry("transformer-layers sequence")
TRANSFORMER = Registry("Transformer")


def build_positional_encoding(cfg, default_args=None):
    """Build positional encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


@POSITIONAL_ENCODING.register_module()
class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for BEV."""

    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50, init_cfg=None):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = (
            torch.cat((x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)), dim=-1)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


def build_attention(cfg, default_args=None):
    """Build attention module."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_feedforward_network(cfg, default_args=None):
    """Build feedforward network."""
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)


@FEEDFORWARD_NETWORK.register_module()
class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Defaults to 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        feedforward_channels,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(FFN, self).__init__(init_cfg)
        assert num_fcs >= 2, f"num_fcs should be at least 2. got {num_fcs}."
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        if act_cfg.get("type") == "GELU":
            self.activate = nn.GELU()
        elif act_cfg.get("type") == "ReLU":
            self.activate = nn.ReLU(inplace=act_cfg.get("inplace", False))
        else:
            self.activate = nn.ReLU(inplace=False)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            layers.append(self.activate)
            layers.append(nn.Dropout(ffn_drop))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


@TRANSFORMER_LAYER.register_module()
class BaseTransformerLayer(BaseModule):
    """
    Base `TransformerLayer` for vision transformer.
    """  # noqa: E501

    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        deprecated_args = dict(
            feedforward_channels="feedforward_channels", ffn_dropout="ffn_drop", ffn_num_fcs="num_fcs"
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments "
                    f"to a dict named `ffn_cfgs`. "
                )
                ffn_cfgs[new_name] = kwargs[ori_name]

        super(BaseTransformerLayer, self).__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & set(["self_attn", "norm", "ffn", "cross_attn"]) == set(operation_order), (
            f"The operation_order of"
            f" {self.__class__.__name__} should "
            f"contains all four operation type "
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"
        )

        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count("ffn")
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if "embed_dims" not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]["embed_dims"] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]["embed_dims"] == self.embed_dims

            self.ffns.append(build_feedforward_network(ffn_cfgs[ffn_index]))

        self.norms = ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `BaseTransformerLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER.register_module()
class DetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
        self,
        attn_cfgs,
        feedforward_channels,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        ffn_num_fcs=2,
        **kwargs,
    ):
        super(DetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )
        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])


@ATTENTION.register_module()
class MultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        dropout_layer=dict(type="Dropout", drop_prob=0.0),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super().__init__(init_cfg)
        if "dropout" in kwargs:
            import warnings

            warnings.warn(
                "The arguments `dropout` in MultiheadAttention "
                "has been deprecated, now you can separately "
                "set `attn_drop`(float), proj_drop(float), "
                "and `dropout_layer`(dict) ",
                DeprecationWarning,
            )
            attn_drop = kwargs["dropout"]
            dropout_layer["drop_prob"] = kwargs.pop("dropout")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `MultiheadAttention`."""
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos

        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


@ATTENTION.register_module()
class MultiScaleDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        self._is_init = True

    @deprecated_api_warning({"residual": "identity"}, cls_name="MultiScaleDeformableAttention")
    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        # For CPU-only inference, use PyTorch fallback
        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


def build_transformer_layer(cfg, default_args=None):
    """Build transformer layer."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Build transformer layer sequence."""
    if isinstance(cfg, dict):
        cfg = cfg.copy()
        if "transformerlayers" in cfg and "transformerlayer" not in cfg:
            cfg["transformerlayer"] = cfg.pop("transformerlayers")
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)


def build_norm_layer(cfg, num_features, postfix=""):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
        num_features (int): Number of input channels.
        postfix (int, str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if "type" not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    # Register common norm layers if not already registered
    if layer_type == "BN" and "BN" not in NORM_LAYERS:
        NORM_LAYERS.register_module(name="BN", module=nn.BatchNorm2d)
    elif layer_type == "SyncBN" and "SyncBN" not in NORM_LAYERS:
        # For CPU-only, SyncBN falls back to BatchNorm2d
        NORM_LAYERS.register_module(name="SyncBN", module=nn.BatchNorm2d)
    elif layer_type == "LN" and "LN" not in NORM_LAYERS:
        NORM_LAYERS.register_module(name="LN", module=nn.LayerNorm)

    if layer_type not in NORM_LAYERS:
        raise KeyError(f"Unrecognized norm type {layer_type}")

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr_map = {"BN": "bn", "SyncBN": "bn", "LN": "ln"}
    abbr = abbr_map.get(layer_type, layer_type.lower())

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    # Filter out internal config keys (starting with _)
    filtered_cfg = {k: v for k, v in cfg_.items() if not k.startswith("_")}
    layer = norm_layer(num_features, **filtered_cfg)

    if not requires_grad:
        for param in layer.parameters():
            param.requires_grad = False
    else:
        for param in layer.parameters():
            param.requires_grad = True

    return name, layer


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a conv layer.
            Default: None.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type="Conv2d")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict or None")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    # Register common conv layers if not already registered
    if layer_type == "Conv2d" and "Conv2d" not in CONV_LAYERS:
        CONV_LAYERS.register_module(name="Conv2d", module=nn.Conv2d)

    if layer_type not in CONV_LAYERS:
        raise KeyError(f"Unrecognized conv type {layer_type}")

    conv_layer = CONV_LAYERS.get(layer_type)
    # Filter out internal config keys (starting with _)
    filtered_cfg = {k: v for k, v in cfg_.items() if not k.startswith("_")}
    layer = conv_layer(*args, **kwargs, **filtered_cfg)

    return layer


def build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    if cfg is None:
        cfg_ = dict(type="ReLU")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict or None")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    # Register common activation layers if not already registered
    if layer_type == "ReLU" and "ReLU" not in ACTIVATION_LAYERS:
        ACTIVATION_LAYERS.register_module(name="ReLU", module=nn.ReLU)
    elif layer_type == "LeakyReLU" and "LeakyReLU" not in ACTIVATION_LAYERS:
        ACTIVATION_LAYERS.register_module(name="LeakyReLU", module=nn.LeakyReLU)
    elif layer_type == "Tanh" and "Tanh" not in ACTIVATION_LAYERS:
        ACTIVATION_LAYERS.register_module(name="Tanh", module=nn.Tanh)
    elif layer_type == "Sigmoid" and "Sigmoid" not in ACTIVATION_LAYERS:
        ACTIVATION_LAYERS.register_module(name="Sigmoid", module=nn.Sigmoid)
    elif layer_type == "PReLU" and "PReLU" not in ACTIVATION_LAYERS:
        ACTIVATION_LAYERS.register_module(name="PReLU", module=nn.PReLU)
    elif layer_type == "GELU" and "GELU" not in ACTIVATION_LAYERS:
        ACTIVATION_LAYERS.register_module(name="GELU", module=nn.GELU)
    elif layer_type == "SiLU" and "SiLU" not in ACTIVATION_LAYERS:
        ACTIVATION_LAYERS.register_module(name="SiLU", module=nn.SiLU)

    if layer_type not in ACTIVATION_LAYERS:
        raise KeyError(f"Unrecognized activation type {layer_type}")

    activation_layer = ACTIVATION_LAYERS.get(layer_type)
    # Filter out internal config keys (starting with _)
    filtered_cfg = {k: v for k, v in cfg_.items() if not k.startswith("_")}
    layer = activation_layer(**filtered_cfg)

    return layer


def build_padding_layer(cfg, padding):
    """Build padding layer.

    Args:
        cfg (dict): The padding layer config, which should contain:
            - type (str): Layer type.
        padding (int | tuple[int]): Padding value.

    Returns:
        nn.Module: Created padding layer.
    """
    if cfg is None:
        return None

    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if "type" not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    # Register common padding layers if not already registered
    if layer_type == "reflect" and "reflect" not in PADDING_LAYERS:
        PADDING_LAYERS.register_module(name="reflect", module=nn.ReflectionPad2d)
    elif layer_type == "replicate" and "replicate" not in PADDING_LAYERS:
        PADDING_LAYERS.register_module(name="replicate", module=nn.ReplicationPad2d)
    elif layer_type == "zero" and "zero" not in PADDING_LAYERS:
        # Zero padding is handled by conv layer itself
        return None

    if layer_type not in PADDING_LAYERS:
        raise KeyError(f"Unrecognized padding type {layer_type}")

    padding_layer = PADDING_LAYERS.get(layer_type)
    layer = padding_layer(padding)

    return layer


try:
    from torch.nn.modules.instancenorm import _InstanceNorm
except ImportError:
    _InstanceNorm = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)


@PLUGIN_LAYERS.register_module()
class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        with_spectral_norm=False,
        padding_mode="zeros",
        order=("conv", "norm", "act"),
    ):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ["zeros", "circular"]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(["conv", "norm", "act"])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                # Check if norm is BatchNorm or InstanceNorm
                is_batch_norm = isinstance(norm, _BatchNorm)
                if isinstance(_InstanceNorm, tuple):
                    is_instance_norm = isinstance(norm, _InstanceNorm)
                else:
                    is_instance_norm = isinstance(norm, _InstanceNorm)
                if is_batch_norm or is_instance_norm:
                    warnings.warn("Unnecessary conv bias before batch/instance norm")
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_["type"] not in ["Tanh", "PReLU", "Sigmoid", "HSigmoid", "Swish"]:
                act_cfg_.setdefault("inplace", inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and self.act_cfg["type"] == "LeakyReLU":
                nonlinearity = "leaky_relu"
                a = self.act_cfg.get("negative_slope", 0.01)
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class BasicBlock(BaseModule):
    """Basic block for ResNet."""

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, "Not implemented yet."
        assert plugins is None, "Not implemented yet."

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg, inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

        if self.with_cp and x.requires_grad:
            import torch.utils.checkpoint as checkpoint

            out = checkpoint.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ["pytorch", "caffe"]
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ["after_conv1", "after_conv2", "after_conv3"]
            assert all(p["position"] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [plugin["cfg"] for plugin in plugins if plugin["position"] == "after_conv1"]
            self.after_conv2_plugins = [plugin["cfg"] for plugin in plugins if plugin["position"] == "after_conv2"]
            self.after_conv3_plugins = [plugin["cfg"] for plugin in plugins if plugin["position"] == "after_conv3"]

        if self.style == "pytorch":
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop("fallback_on_stride", False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        else:
            assert self.conv_cfg is None, "conv_cfg must be None for DCN"
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(conv_cfg, planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(plugin, in_channels=in_channels, postfix=plugin.pop("postfix", ""))
            assert not hasattr(self, name), f"duplicate plugin {name}"
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            import torch.utils.checkpoint as checkpoint

            out = checkpoint.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(
        self,
        block,
        inplanes,
        planes,
        num_blocks,
        stride=1,
        avg_down=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        downsample_first=True,
        **kwargs,
    ):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False)
                )
            downsample.extend(
                [
                    build_conv_layer(
                        conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=conv_stride, bias=False
                    ),
                    build_norm_layer(norm_cfg, planes * block.expansion)[1],
                ]
            )
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs,
                )
            )
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(inplanes=inplanes, planes=planes, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs)
                )

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(inplanes=inplanes, planes=inplanes, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs)
                )
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs,
                )
            )
        super(ResLayer, self).__init__(*layers)


def build_plugin_layer(cfg, postfix="", in_channels=None, **kwargs):
    """Build plugin layer."""
    if cfg is None:
        return None, None
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()
    plugin_type = cfg_.pop("type")
    if plugin_type in PLUGIN_LAYERS:
        plugin_layer = PLUGIN_LAYERS.get(plugin_type)
        if in_channels is not None:
            cfg_.setdefault("in_channels", in_channels)
        layer = plugin_layer(**cfg_, **kwargs)
        name = f"{plugin_type}{postfix}"
        return name, layer
    else:
        # For CPU-only inference, return None if plugin is not available
        return None, None


class ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg=None,
    ):
        super(ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")

        block_init_cfg = None
        assert not (init_cfg and pretrained), "init_cfg and pretrained cannot be setting at the same time"
        if isinstance(pretrained, str):
            warnings.warn("DeprecationWarning: pretrained is deprecated, " 'please use "init_cfg" instead')
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type="Kaiming", layer="Conv2d"),
                    dict(type="Constant", val=1, layer=["_BatchNorm", "GroupNorm"]),
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(type="Constant", val=0, override=dict(name="norm2"))
                    elif block is Bottleneck:
                        block_init_cfg = dict(type="Constant", val=0, override=dict(name="norm3"))
        else:
            raise TypeError("pretrained must be a str or None")

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage."""
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop("stages", None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg, in_channels, stem_channels // 2, kernel_size=3, stride=2, padding=1, bias=False
                ),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg, stem_channels // 2, stem_channels, kernel_size=3, stride=1, padding=1, bias=False
                ),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg, in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


@DROPOUT_LAYERS.register_module()
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1

    """

    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


@DROPOUT_LAYERS.register_module()
class Dropout(nn.Dropout):
    """A wrapper for ``torch.nn.Dropout``, We rename the ``p`` of
    ``torch.nn.Dropout`` to ``drop_prob`` so as to be consistent with
    ``DropPath``

    Args:
        drop_prob (float): Probability of the elements to be
            zeroed. Default: 0.5.
        inplace (bool):  Do the operation inplace or not. Default: False.

    """

    def __init__(self, drop_prob=0.5, inplace=False):
        super().__init__(p=drop_prob, inplace=inplace)


def build_dropout(cfg, default_args=None):
    """Builder for drop out layers."""
    return build_from_cfg(cfg, DROPOUT_LAYERS, default_args)


MMCV_MODELS = Registry("models")
_MODELS_DET = Registry("models", parent=MMCV_MODELS)

BACKBONES = _MODELS_DET
NECKS = _MODELS_DET
HEADS = _MODELS_DET
DETECTORS = _MODELS_DET
VOXEL_ENCODERS = _MODELS_DET
MIDDLE_ENCODERS = _MODELS_DET
FUSION_LAYERS = _MODELS_DET


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    raise NotImplementedError("Loss functions are not supported in inference-only mode.")


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn("train_cfg and test_cfg is deprecated, " "please specify them in model", UserWarning)
    assert cfg.get("train_cfg") is None or train_cfg is None, "train_cfg specified in both outer field and model field "
    assert cfg.get("test_cfg") is None or test_cfg is None, "test_cfg specified in both outer field and model field "
    return DETECTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def fp16_clamp(x, min=None, max=None):
    if x.dtype == torch.float16:
        return x.float().clamp(min, max).half()
    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps_tensor = union.new_tensor([eps])
    union = torch.max(union, eps_tensor)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps_tensor)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False
        if init_cfg is not None:
            self.init_cfg = init_cfg

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, "roi_head") and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return (hasattr(self, "roi_head") and self.roi_head.with_bbox) or (
            hasattr(self, "bbox_head") and self.bbox_head is not None
        )

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return (hasattr(self, "roi_head") and self.roi_head.with_mask) or (
            hasattr(self, "mask_head") and self.mask_head is not None
        )

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, "imgs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got {type(var)}")

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f"num of augmentations ({len(imgs)}) " f"!= num of image meta ({len(img_metas)})")

        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]["batch_input_shape"] = tuple(img.size()[-2:])

        if num_augs == 1:
            if "proposals" in kwargs:
                kwargs["proposals"] = kwargs["proposals"][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, "aug test does not support " "inference with batch size " f"{imgs[0].size(0)}"
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=("img",))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            raise NotImplementedError(
                "Training is disabled in this embedded MapTR reference. Use return_loss=False for inference."
            )
        return self.forward_test(**kwargs)


class BaseDenseHead(nn.Module):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        """Initialize weights of the head."""


class BBoxTestMixin:
    """Mixin class for test time bbox augmentation."""

    def simple_test_bboxes(self, *args, **kwargs):
        """Test det bboxes without test-time augmentation."""


@HEADS.register_module()
class AnchorFreeHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-free head (FCOS, Fovea, RepPoints, etc.)."""

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=4,
        strides=(4, 8, 16, 32, 64),
        dcn_on_last_conv=False,
        conv_bias="auto",
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="IoULoss", loss_weight=1.0),
        conv_cfg=None,
        norm_cfg=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="conv_cls", std=0.01, bias_prob=0.01),
        ),
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == "auto" or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = None
        self.loss_bbox = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

    def forward(self, feats):
        """Forward features from the upstream network."""
        return multi_apply(self.forward_single, feats)[:2]

    def forward_single(self, x):
        """Forward features of a single scale level."""


@HEADS.register_module()
class DETRHead(AnchorFreeHead):
    """Implements the DETR transformer head."""

    _version = 2

    def __init__(
        self,
        num_classes,
        in_channels,
        num_query=100,
        num_reg_fcs=2,
        transformer=None,
        sync_cls_avg_factor=False,
        positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True),
        loss_cls=dict(type="CrossEntropyLoss", bg_cls_weight=0.1, use_sigmoid=False, loss_weight=1.0, class_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="ClassificationCost", weight=1.0),
                reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            )
        ),
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
        **kwargs,
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get("class_weight", None)
        if class_weight is not None and (self.__class__ is DETRHead):
            assert isinstance(class_weight, float), (
                "Expected " "class_weight to have type float. Found " f"{type(class_weight)}."
            )
            bg_cls_weight = loss_cls.get("bg_cls_weight", class_weight)
            assert isinstance(bg_cls_weight, float), (
                "Expected " "bg_cls_weight to have type float. Found " f"{type(bg_cls_weight)}."
            )
            class_weight = torch.ones(num_classes + 1) * class_weight
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({"class_weight": class_weight})
            if "bg_cls_weight" in loss_cls:
                loss_cls.pop("bg_cls_weight")
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            pass
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        if isinstance(loss_cls, dict):
            use_sigmoid = loss_cls.get("use_sigmoid", True)
        elif loss_cls is not None:
            use_sigmoid = getattr(loss_cls, "use_sigmoid", True)
        else:
            use_sigmoid = True
        if use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

    def forward(self, feats, img_metas):
        """Forward function."""
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, img_metas_list)

    def forward_single(self, x, img_metas):
        """Forward function for a single feature level."""


DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")


def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    if isinstance(bboxes, torch.Tensor):
        boxes_3d = bboxes.cpu()
    elif hasattr(bboxes, "tensor") and isinstance(bboxes.tensor, torch.Tensor):
        boxes_3d = bboxes.tensor.cpu()
    elif hasattr(bboxes, "to") and callable(getattr(bboxes, "to", None)):
        boxes_3d = bboxes.to("cpu")
    elif hasattr(bboxes, "cpu") and callable(getattr(bboxes, "cpu", None)):
        boxes_3d = bboxes.cpu()
    else:
        boxes_3d = torch.as_tensor(bboxes, device="cpu")
    result_dict = dict(boxes_3d=boxes_3d, scores_3d=scores.cpu(), labels_3d=labels.cpu())

    if attrs is not None:
        result_dict["attrs_3d"] = attrs.cpu()

    return result_dict


DC = DataContainer


class Base3DDetector(BaseDetector):
    """Base class for detectors."""

    def forward_test(self, points, img_metas, img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(points, "points"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError("num of augmentations ({}) != num of image meta ({})".format(len(points), len(img_metas)))

        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
        else:
            return self.aug_test(points, img_metas, img, **kwargs)

    @auto_fp16(apply_to=("img", "points"))
    def forward(self, return_loss=True, **kwargs):
        """Forward entrypoint; training disabled in embedded MapTR reference."""
        if return_loss:
            raise NotImplementedError(
                "3D detector training is disabled in this embedded MapTR reference. Use return_loss=False for inference."
            )
        return self.forward_test(**kwargs)

    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (list[dict]): Input points and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data["points"][0], DC):
                points = data["points"][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data["points"][0], torch.Tensor):
                points = data["points"][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} " f"for visualization!")
            if isinstance(data["img_metas"][0], DC):
                pts_filename = data["img_metas"][0]._data[0][batch_id]["pts_filename"]
                box_mode_3d = data["img_metas"][0]._data[0][batch_id]["box_mode_3d"]
            elif mmcv.is_list_of(data["img_metas"][0], dict):
                pts_filename = data["img_metas"][0][batch_id]["pts_filename"]
                box_mode_3d = data["img_metas"][0][batch_id]["box_mode_3d"]
            else:
                ValueError(f"Unsupported data type {type(data['img_metas'][0])} " f"for visualization!")
            file_name = osp.split(pts_filename)[-1].split(".")[0]

            assert out_dir is not None, "Expect out_dir, got none."

            pred_bboxes = result[batch_id]["boxes_3d"]

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR, Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d, Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(f"Unsupported box_mode_3d {box_mode_3d} for convertion!")
            pred_bboxes = pred_bboxes.tensor.cpu().numpy()


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments."""
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def merge_aug_bboxes_3d(aug_bboxes, img_metas, test_cfg):
    """Merge augmented detection bboxes and scores."""
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        recovered_bboxes.append(bboxes)
    return recovered_bboxes[0] if len(recovered_bboxes) == 1 else recovered_bboxes


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


@DETECTORS.register_module()
class MVXTwoStageDetector(Base3DDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        super(MVXTwoStageDetector, self).__init__(init_cfg=init_cfg)

        if pts_voxel_layer:
            self.pts_voxel_layer = None
        if pts_voxel_encoder:
            self.pts_voxel_encoder = None
        if pts_middle_encoder:
            self.pts_middle_encoder = None
        if pts_backbone:
            self.pts_backbone = build_backbone(pts_backbone) if isinstance(pts_backbone, dict) else pts_backbone
        if pts_fusion_layer:
            self.pts_fusion_layer = None
        if pts_neck is not None:
            self.pts_neck = None
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = build_head(pts_bbox_head) if isinstance(pts_bbox_head, dict) else pts_bbox_head

        if img_backbone:
            self.img_backbone = build_backbone(img_backbone) if isinstance(img_backbone, dict) else img_backbone
        if img_neck is not None:
            self.img_neck = build_neck(img_neck) if isinstance(img_neck, dict) else img_neck
        if img_rpn_head is not None:
            self.img_rpn_head = None
        if img_roi_head is not None:
            self.img_roi_head = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get("img", None)
            pts_pretrained = pretrained.get("pts", None)
        else:
            raise ValueError(f"pretrained should be a dict, got {type(pretrained)}")

        if self.with_img_backbone:
            if img_pretrained is not None:
                warnings.warn(
                    "DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg"
                )
                if hasattr(self.img_backbone, "init_cfg"):
                    self.img_backbone.init_cfg = dict(type="Pretrained", checkpoint=img_pretrained)
        if self.with_img_roi_head:
            if img_pretrained is not None:
                warnings.warn(
                    "DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg"
                )
                if hasattr(self.img_roi_head, "init_cfg"):
                    self.img_roi_head.init_cfg = dict(type="Pretrained", checkpoint=img_pretrained)

        if self.with_pts_backbone:
            if pts_pretrained is not None:
                warnings.warn(
                    "DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg"
                )
                if hasattr(self.pts_backbone, "init_cfg"):
                    self.pts_backbone.init_cfg = dict(type="Pretrained", checkpoint=pts_pretrained)

    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self, "img_shared_head") and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self, "pts_bbox_head") and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self, "img_bbox_head") and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, "img_backbone") and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, "pts_backbone") and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self, "pts_fusion_layer") and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, "img_neck") and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, "pts_neck") and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, "img_rpn_head") and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, "img_roi_head") and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self, "voxel_encoder") and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self, "middle_encoder") and self.middle_encoder is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        return None

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict["pts_bbox"] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = []
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict["img_bbox"] = img_bbox
        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs, img_metas)
        return img_feats, pts_feats

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels) for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas, self.pts_bbox_head.test_cfg)
        return merged_bboxes


class Box3DMode:
    LIDAR = 0
    DEPTH = 1
    CAMERA = 2
    CAM = 2  # Alias for CAMERA
    LIDAR_BOX = 0
    DEPTH_BOX = 1
    CAMERA_BOX = 2

    @staticmethod
    def convert(boxes, src, dst, rt_mat=None):
        if src == dst:
            return boxes
        if isinstance(boxes, torch.Tensor):
            return Box3DMode._convert_tensor(boxes, src, dst, rt_mat)
        elif isinstance(boxes, np.ndarray):
            return Box3DMode._convert_numpy(boxes, src, dst, rt_mat)
        else:
            raise NotImplementedError

    @staticmethod
    def _convert_tensor(boxes, src, dst, rt_mat=None):
        if src == Box3DMode.LIDAR and dst == Box3DMode.DEPTH:
            return boxes.clone()
        elif src == Box3DMode.DEPTH and dst == Box3DMode.LIDAR:
            return boxes.clone()
        else:
            return boxes.clone()

    @staticmethod
    def _convert_numpy(boxes, src, dst, rt_mat=None):
        if src == Box3DMode.LIDAR and dst == Box3DMode.DEPTH:
            return boxes.copy()
        elif src == Box3DMode.DEPTH and dst == Box3DMode.LIDAR:
            return boxes.copy()
        else:
            return boxes.copy()


class Coord3DMode:
    LIDAR = 0
    DEPTH = 1
    CAMERA = 2

    @staticmethod
    def convert_point(points, src, dst, rt_mat=None):
        if src == dst:
            return points
        if isinstance(points, torch.Tensor):
            return Coord3DMode._convert_tensor(points, src, dst, rt_mat)
        elif isinstance(points, np.ndarray):
            return Coord3DMode._convert_numpy(points, src, dst, rt_mat)
        else:
            raise NotImplementedError

    @staticmethod
    def _convert_tensor(points, src, dst, rt_mat=None):
        if src == Coord3DMode.LIDAR and dst == Coord3DMode.DEPTH:
            return points.clone()
        elif src == Coord3DMode.DEPTH and dst == Coord3DMode.LIDAR:
            return points.clone()
        else:
            return points.clone()

    @staticmethod
    def _convert_numpy(points, src, dst, rt_mat=None):
        if src == Coord3DMode.LIDAR and dst == Coord3DMode.DEPTH:
            return points.copy()
        elif src == Coord3DMode.DEPTH and dst == Coord3DMode.LIDAR:
            return points.copy()
        else:
            return points.copy()


def get_box_type(box_type):
    """Get the type and mode of box structure.

    Args:
        box_type (str): The type of box structure.
            The valid value are "LiDAR", "Camera", or "Depth".

    Returns:
        tuple: Box type and box mode.
    """
    box_type_lower = box_type.lower()
    if box_type_lower == "lidar":
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.LIDAR
    elif box_type_lower == "camera":
        # CameraInstance3DBoxes is not implemented, use LiDARInstance3DBoxes as fallback
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.CAM
    elif box_type_lower == "depth":
        # DepthInstance3DBoxes is not implemented, use LiDARInstance3DBoxes as fallback
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.DEPTH
    else:
        raise ValueError('Only "box_type" of "camera", "lidar", "depth"' f" are supported, got {box_type}")

    return box_type_3d, box_mode_3d


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range.
            Defaults to 0.5.
        period (float, optional): Period of the value. Defaults to np.pi.

    Returns:
        torch.Tensor: Value in the range of
            [-offset * period, (1-offset) * period]
    """
    return val - torch.floor(val / period + offset) * period


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

    Returns:
        torch.Tensor: Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes


class BasePoints(object):
    """Base class for Points.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z). Default to 3.
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Default to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Default to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, points_dim)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == points_dim, tensor.size()

        self.tensor = tensor
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.rotation_axis = 0

    @property
    def coord(self):
        """torch.Tensor: Coordinates of each point with size (N, 3)."""
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor):
        """Set the coordinates of each point."""
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        self.tensor[:, :3] = tensor

    @property
    def height(self):
        """torch.Tensor: A vector with height of each point."""
        if self.attribute_dims is not None and "height" in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims["height"]]
        else:
            return None

    @height.setter
    def height(self, tensor):
        """Set the height of each point."""
        try:
            tensor = tensor.reshape(self.shape[0])
        except (RuntimeError, ValueError):
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and "height" in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims["height"]] = tensor
        else:
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor.unsqueeze(1)], dim=1)
            self.attribute_dims.update(dict(height=attr_dim))
            self.points_dim += 1

    @property
    def color(self):
        """torch.Tensor: A vector with color of each point."""
        if self.attribute_dims is not None and "color" in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims["color"]]
        else:
            return None

    @color.setter
    def color(self, tensor):
        """Set the color of each point."""
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if tensor.max() >= 256 or tensor.min() < 0:
            warnings.warn("point got color value beyond [0, 255]")
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and "color" in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims["color"]] = tensor
        else:
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(dict(color=[attr_dim, attr_dim + 1, attr_dim + 2]))
            self.points_dim += 3

    @property
    def shape(self):
        """torch.Shape: Shape of points."""
        return self.tensor.shape

    def shuffle(self):
        """Shuffle the points.

        Returns:
            torch.Tensor: The shuffled index.
        """
        idx = torch.randperm(self.__len__(), device=self.tensor.device)
        self.tensor = self.tensor[idx]
        return idx

    def rotate(self, rotation, axis=None):
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (float, np.ndarray, torch.Tensor): Rotation matrix
                or angle.
            axis (int): Axis to rotate at. Defaults to None.
        """
        if not isinstance(rotation, torch.Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1, f"invalid rotation shape {rotation.shape}"

        if axis is None:
            axis = self.rotation_axis

        if rotation.numel() == 1:
            rot_sin = torch.sin(rotation)
            rot_cos = torch.cos(rotation)
            if axis == 1:
                rot_mat_T = rotation.new_tensor([[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]])
            elif axis == 2 or axis == -1:
                rot_mat_T = rotation.new_tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
            elif axis == 0:
                rot_mat_T = rotation.new_tensor([[0, rot_cos, -rot_sin], [0, rot_sin, rot_cos], [1, 0, 0]])
            else:
                raise ValueError("axis should in range")
            rot_mat_T = rot_mat_T.T
        elif rotation.numel() == 9:
            rot_mat_T = rotation
        else:
            raise NotImplementedError
        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T

        return rot_mat_T

    @abstractmethod
    def flip(self, bev_direction="horizontal"):
        """Flip the points in BEV along given BEV direction."""

    def translate(self, trans_vector):
        """Translate points with the given translation vector.

        Args:
            trans_vector (np.ndarray, torch.Tensor): Translation
                vector of size 3 or nx3.
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert trans_vector.shape[0] == self.tensor.shape[0] and trans_vector.shape[1] == 3
        else:
            raise NotImplementedError(f"Unsupported translation vector of shape {trans_vector.shape}")
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Returns:
            torch.Tensor: A binary vector indicating whether each point is
                inside the reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > point_range[0])
            & (self.tensor[:, 1] > point_range[1])
            & (self.tensor[:, 2] > point_range[2])
            & (self.tensor[:, 0] < point_range[3])
            & (self.tensor[:, 1] < point_range[4])
            & (self.tensor[:, 2] < point_range[5])
        )
        return in_range_flags

    @abstractmethod
    def in_range_bev(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each point is inside
                the reference range.
        """

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst: The target Box mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.

        Returns:
            BasePoints: The converted box of the same type
                in the `dst` mode.
        """

    def scale(self, scale_factor):
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor

    def __getitem__(self, item):
        """Indexing on Points."""
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1), points_dim=self.points_dim, attribute_dims=self.attribute_dims
            )
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] if item[1].stop is None else item[1].stop
                step = 1 if item[1].step is None else item[1].step
                item = list(item)
                item[1] = list(range(start, stop, step))
                item = tuple(item)
            elif isinstance(item[1], int):
                item = list(item)
                item[1] = [item[1]]
                item = tuple(item)
            p = self.tensor[item[0], item[1]]

            keep_dims = list(set(item[1]).intersection(set(range(3, self.tensor.shape[1]))))
            if self.attribute_dims is not None:
                attribute_dims = self.attribute_dims.copy()
                for key in self.attribute_dims.keys():
                    cur_attribute_dims = attribute_dims[key]
                    if isinstance(cur_attribute_dims, int):
                        cur_attribute_dims = [cur_attribute_dims]
                    intersect_attr = list(set(cur_attribute_dims).intersection(set(keep_dims)))
                    if len(intersect_attr) == 1:
                        attribute_dims[key] = intersect_attr[0]
                    elif len(intersect_attr) > 1:
                        attribute_dims[key] = intersect_attr
                    else:
                        attribute_dims.pop(key)
            else:
                attribute_dims = None
        elif isinstance(item, (slice, np.ndarray, torch.Tensor)):
            p = self.tensor[item]
            attribute_dims = self.attribute_dims
        else:
            raise NotImplementedError(f"Invalid slice {item}!")

        assert p.dim() == 2, f"Indexing on Points with {item} failed to return a matrix!"
        return original_type(p, points_dim=p.shape[1], attribute_dims=attribute_dims)

    def __len__(self):
        """int: Number of points in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    @classmethod
    def cat(cls, points_list):
        """Concatenate a list of Points into a single Points.

        Args:
            points_list (list[BasePoints]): List of points.

        Returns:
            BasePoints: The concatenated Points.
        """
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(points, cls) for points in points_list)

        cat_points = cls(
            torch.cat([p.tensor for p in points_list], dim=0),
            points_dim=points_list[0].tensor.shape[1],
            attribute_dims=points_list[0].attribute_dims,
        )
        return cat_points

    def to(self, device):
        """Convert current points to a specific device.

        Args:
            device (str | torch.device): The name of the device.

        Returns:
            BasePoints: A new boxes object on the
                specific device.
        """
        original_type = type(self)
        return original_type(self.tensor.to(device), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    def clone(self):
        """Clone the Points.

        Returns:
            BasePoints: Box object with the same properties
                as self.
        """
        original_type = type(self)
        return original_type(self.tensor.clone(), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    @property
    def device(self):
        """str: The device of the points are on."""
        return self.tensor.device

    def __iter__(self):
        """Yield a point as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A point of shape (4,).
        """
        yield from self.tensor

    def new_point(self, data):
        """Create a new point object with data.

        Args:
            data (torch.Tensor | numpy.array | list): Data to be copied.

        Returns:
            BasePoints: A new point object with ``data``,
                the object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) if not isinstance(data, torch.Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(new_tensor, points_dim=self.points_dim, attribute_dims=self.attribute_dims)


class LiDARPoints(BasePoints):
    """Points of instances in LIDAR coordinates.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z). Default to 3.
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Default to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Default to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        super(LiDARPoints, self).__init__(tensor, points_dim=points_dim, attribute_dims=attribute_dims)
        self.rotation_axis = 2

    def flip(self, bev_direction="horizontal"):
        """Flip the boxes in BEV along given BEV direction."""
        if bev_direction == "horizontal":
            self.tensor[:, 1] = -self.tensor[:, 1]
        elif bev_direction == "vertical":
            self.tensor[:, 0] = -self.tensor[:, 0]

    def in_range_bev(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each point is inside \
                the reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > point_range[0])
            & (self.tensor[:, 1] > point_range[1])
            & (self.tensor[:, 0] < point_range[2])
            & (self.tensor[:, 1] < point_range[3])
        )
        return in_range_flags

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Coord3DMode`): The target Point mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type \
                in the `dst` mode.
        """
        return Coord3DMode.convert_point(point=self, src=Coord3DMode.LIDAR, dst=dst, rt_mat=rt_mat)


class CameraPoints(BasePoints):
    """Points of instances in CAM coordinates.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z). Default to 3.
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Default to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Default to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        super(CameraPoints, self).__init__(tensor, points_dim=points_dim, attribute_dims=attribute_dims)
        self.rotation_axis = 1

    def flip(self, bev_direction="horizontal"):
        """Flip the boxes in BEV along given BEV direction."""
        if bev_direction == "horizontal":
            self.tensor[:, 0] = -self.tensor[:, 0]
        elif bev_direction == "vertical":
            self.tensor[:, 2] = -self.tensor[:, 2]

    def in_range_bev(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each point is inside \
                the reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > point_range[0])
            & (self.tensor[:, 2] > point_range[1])
            & (self.tensor[:, 0] < point_range[2])
            & (self.tensor[:, 2] < point_range[3])
        )
        return in_range_flags

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Coord3DMode`): The target Point mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type \
                in the `dst` mode.
        """
        return Coord3DMode.convert_point(point=self, src=Coord3DMode.CAM, dst=dst, rt_mat=rt_mat)


class DepthPoints(BasePoints):
    """Points of instances in DEPTH coordinates.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z). Default to 3.
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Default to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Default to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        super(DepthPoints, self).__init__(tensor, points_dim=points_dim, attribute_dims=attribute_dims)
        self.rotation_axis = 2

    def flip(self, bev_direction="horizontal"):
        """Flip the boxes in BEV along given BEV direction."""
        if bev_direction == "horizontal":
            self.tensor[:, 0] = -self.tensor[:, 0]
        elif bev_direction == "vertical":
            self.tensor[:, 1] = -self.tensor[:, 1]

    def in_range_bev(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each point is inside \
                the reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > point_range[0])
            & (self.tensor[:, 1] > point_range[1])
            & (self.tensor[:, 0] < point_range[2])
            & (self.tensor[:, 1] < point_range[3])
        )
        return in_range_flags

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Coord3DMode`): The target Point mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type \
                in the `dst` mode.
        """
        return Coord3DMode.convert_point(point=self, src=Coord3DMode.DEPTH, dst=dst, rt_mat=rt_mat)


def get_points_type(points_type):
    """Get the class of points according to coordinate type.

    Args:
        points_type (str): The type of points coordinate.
            The valid value are "CAMERA", "LIDAR", or "DEPTH".

    Returns:
        class: Points type.
    """
    if points_type == "CAMERA":
        points_cls = CameraPoints
    elif points_type == "LIDAR":
        points_cls = LiDARPoints
    elif points_type == "DEPTH":
        points_cls = DepthPoints
    else:
        raise ValueError('Only "points_type" of "CAMERA", "LIDAR", or "DEPTH"' f" are supported, got {points_type}")

    return points_cls


class BaseInstance3DBoxes(object):
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in
        the box is (0.5, 0.5, 0).

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x box_dim matrix.
        box_dim (int): Number of the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw).
            Default to 7.
        with_yaw (bool): Whether the box is with yaw rotation.
            If False, the value of yaw will be set to 0 as minmax boxes.
            Default to True.
        origin (tuple[float]): The relative position of origin in the box.
            Default to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, box_dim)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()

        if tensor.shape[-1] == 6:
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def volume(self):
        """torch.Tensor: A vector with volume of each box."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):
        """torch.Tensor: Corners of each box with size (N, 8, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """torch.Tensor: A vector with yaw of each box."""
        return self.tensor[:, 6]

    @property
    def height(self):
        """torch.Tensor: A vector with height of each box."""
        return self.tensor[:, 5]

    @property
    def top_height(self):
        """torch.Tensor: A vector with the top height of each box."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self):
        """torch.Tensor: A vector with bottom's height of each box."""
        return self.tensor[:, 2]

    @property
    def center(self):
        """Calculate the center of all the boxes.

        Returns:
            torch.Tensor: A tensor with center of each box.
        """
        return self.bottom_center

    @property
    def bottom_center(self):
        """torch.Tensor: A tensor with center of each box."""
        return self.tensor[:, :3]

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box."""

    @property
    def corners(self):
        """torch.Tensor: a tensor with 8 corners of each box."""

    @abstractmethod
    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or
        rotation matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor, numpy.ndarray, BasePoints, optional):
                Points to rotate. Defaults to None.
        """

    @abstractmethod
    def flip(self, bev_direction="horizontal"):
        """Flip the boxes in BEV along given BEV direction."""

    def translate(self, trans_vector):
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (torch.Tensor): Translation vector of size 1x3.
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Returns:
            torch.Tensor: A binary vector indicating whether each box is
                inside the reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > box_range[0])
            & (self.tensor[:, 1] > box_range[1])
            & (self.tensor[:, 2] > box_range[2])
            & (self.tensor[:, 0] < box_range[3])
            & (self.tensor[:, 1] < box_range[4])
            & (self.tensor[:, 2] < box_range[5])
        )
        return in_range_flags

    @abstractmethod
    def in_range_bev(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each box is inside
                the reference range.
        """

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst: The target Box mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.

        Returns:
            BaseInstance3DBoxes: The converted box of the same type
                in the `dst` mode.
        """

    def scale(self, scale_factor):
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor

    def limit_yaw(self, offset=0.5, period=np.pi):
        """Limit the yaw to a given period and offset.

        Args:
            offset (float): The offset of the yaw.
            period (float): The expected period.
        """
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)

    def nonempty(self, threshold: float = 0.0):
        """Find boxes that are non-empty.

        Args:
            threshold (float): The threshold of minimal sizes.

        Returns:
            torch.Tensor: A binary vector which represents whether each
                box is empty (False) or non-empty (True).
        """
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = (size_x > threshold) & (size_y > threshold) & (size_z > threshold)
        return keep

    def __getitem__(self, item):
        """Indexing on Boxes."""
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), box_dim=self.box_dim, with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __len__(self):
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    @classmethod
    def cat(cls, boxes_list):
        """Concatenate a list of Boxes into a single Boxes.

        Args:
            boxes_list (list[BaseInstance3DBoxes]): List of boxes.

        Returns:
            BaseInstance3DBoxes: The concatenated Boxes.
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(box, cls) for box in boxes_list)

        cat_boxes = cls(
            torch.cat([b.tensor for b in boxes_list], dim=0),
            box_dim=boxes_list[0].tensor.shape[1],
            with_yaw=boxes_list[0].with_yaw,
        )
        return cat_boxes

    def to(self, device):
        """Convert current boxes to a specific device.

        Args:
            device (str | torch.device): The name of the device.

        Returns:
            BaseInstance3DBoxes: A new boxes object on the
                specific device.
        """
        original_type = type(self)
        return original_type(self.tensor.to(device), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def clone(self):
        """Clone the Boxes.

        Returns:
            BaseInstance3DBoxes: Box object with the same properties
                as self.
        """
        original_type = type(self)
        return original_type(self.tensor.clone(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    @property
    def device(self):
        """str: The device of the boxes are on."""
        return self.tensor.device

    def __iter__(self):
        """Yield a box as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A box of shape (4,).
        """
        yield from self.tensor

    def new_box(self, data):
        """Create a new box object with data.

        Args:
            data (torch.Tensor | numpy.array | list): Data to be copied.

        Returns:
            BaseInstance3DBoxes: A new bbox object with ``data``,
                the object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) if not isinstance(data, torch.Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)


class LiDARInstance3DBoxes:
    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, box_dim)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()
        self.tensor = tensor.clone()
        self.box_dim = box_dim
        self.with_yaw = with_yaw

    @property
    def gravity_center(self):
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def bottom_center(self):
        return self.tensor[:, :3]

    @property
    def dims(self):
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        return self.tensor[:, 6]

    def convert_to(self, dst, rt_mat=None):
        if dst == Box3DMode.LIDAR:
            return self
        else:
            return self

    def __getitem__(self, item):
        return type(self)(self.tensor[item], box_dim=self.box_dim, with_yaw=self.with_yaw)


class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, abc_collections.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("transform must be callable or a dict")

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results.get("img_filename", [])
        if not filename or len(filename) == 0:
            # No images available - return empty results
            results["filename"] = []
            results["img"] = []
            results["img_shape"] = None
            results["ori_shape"] = None
            results["pad_shape"] = None
            results["scale_factor"] = 1.0
            results["img_norm_cfg"] = dict(
                mean=np.zeros(3, dtype=np.float32), std=np.ones(3, dtype=np.float32), to_rgb=False
            )
            return results

        # Filter out non-existent files and load only existing ones
        valid_filenames = []
        valid_images = []
        for name in filename:
            if os.path.exists(name):
                try:
                    img_data = mmcv.imread(name, self.color_type)
                    if img_data is not None:
                        valid_filenames.append(name)
                        valid_images.append(img_data)
                except Exception as e:
                    print(f"Warning: Failed to load image {name}: {e}")
                    continue

        if len(valid_images) == 0:
            # No valid images found - return empty results
            results["filename"] = []
            results["img"] = []
            results["img_shape"] = None
            results["ori_shape"] = None
            results["pad_shape"] = None
            results["scale_factor"] = 1.0
            results["img_norm_cfg"] = dict(
                mean=np.zeros(3, dtype=np.float32), std=np.ones(3, dtype=np.float32), to_rgb=False
            )
            return results

        # img is of shape (h, w, c, num_views)
        img = np.stack(valid_images, axis=-1)

        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = valid_filenames
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32), std=np.ones(num_channels, dtype=np.float32), to_rgb=False
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class DefaultFormatBundle3D(object):
    # DataContainer is defined earlier in this file
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names=None, with_label=True):
        self.class_names = class_names
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if "img" in results:
            if isinstance(results["img"], list):
                imgs = [img.transpose(2, 0, 1) for img in results["img"]]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results["img"] = DataContainer(to_tensor(imgs), stack=True)
            else:
                img = results["img"].transpose(2, 0, 1)
                results["img"] = DataContainer(to_tensor(img), stack=True)

        for key in ["proposals", "gt_bboxes", "gt_bboxes_ignore", "gt_labels"]:
            if key not in results:
                continue
            if isinstance(results[key], list) and len(results[key]) > 0:
                if isinstance(results[key][0], torch.Tensor):
                    results[key] = DataContainer([to_tensor(bbox) for bbox in results[key]])
                else:
                    results[key] = DataContainer(to_tensor(results[key]))
            else:
                results[key] = DataContainer(to_tensor(results[key]))

        if "gt_bboxes_3d" in results:
            if isinstance(results["gt_bboxes_3d"], list):
                results["gt_bboxes_3d"] = DataContainer([bbox for bbox in results["gt_bboxes_3d"]], cpu_only=True)
            else:
                results["gt_bboxes_3d"] = DataContainer(results["gt_bboxes_3d"], cpu_only=True)

        if "gt_labels_3d" in results and self.with_label:
            if isinstance(results["gt_labels_3d"], list):
                results["gt_labels_3d"] = DataContainer([to_tensor(label) for label in results["gt_labels_3d"]])
            else:
                results["gt_labels_3d"] = DataContainer(to_tensor(results["gt_labels_3d"]))

        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(class_names={self.class_names}, with_label={self.with_label})"


@PIPELINES.register_module()
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        img_scale (tuple | list[tuple]): Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            scaling.
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip direction. Options
            are 'horizontal' and 'vertical'. Default: 'horizontal'.
        transforms (list[dict]): Transforms to apply in each augmentation.
    """

    def __init__(self, img_scale=None, pts_scale_ratio=1, flip=False, flip_direction="horizontal", transforms=None):
        self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio if isinstance(pts_scale_ratio, list) else [pts_scale_ratio]
        assert isinstance(self.pts_scale_ratio, list)
        assert isinstance(self.img_scale, list)
        self.flip = flip
        self.flip_direction = flip_direction if isinstance(flip_direction, list) else [flip_direction]
        self.transforms = Compose(transforms) if transforms else None

    def __call__(self, results):
        """Call function to apply test time augmentations on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Results after applying test time augmentations,
                'img_scale', 'flip', and 'flip_direction' keys are added
                into result dict.
        """
        aug_data = []
        if self.img_scale is None or self.img_scale[0] is None:
            img_scale = [None]
        else:
            img_scale = self.img_scale
        for img_scale_item, pts_scale_ratio_item in zip(img_scale, self.pts_scale_ratio):
            for flip_item in [False, True] if self.flip else [False]:
                for flip_direction_item in self.flip_direction:
                    _results = results.copy()
                    _results["img_scale"] = img_scale_item
                    _results["flip"] = flip_item
                    _results["flip_direction"] = flip_direction_item
                    _results["pts_scale_ratio"] = pts_scale_ratio_item
                    data = self.transforms(_results) if self.transforms else _results
                    aug_data.append(data)
        results["aug_data"] = aug_data
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"pts_scale_ratio={self.pts_scale_ratio}, "
        repr_str += f"flip={self.flip}, "
        repr_str += f"flip_direction={self.flip_direction}, "
        repr_str += f"transforms={self.transforms})"
        return repr_str


class Custom3DDataset:
    CLASSES = None

    def __init__(
        self,
        data_root=None,
        ann_file=None,
        pipeline=None,
        classes=None,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
    ):
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_annotations(self.ann_file) if self.ann_file else []
        self.pipeline = Compose(pipeline) if pipeline else None

    def get_classes(self, classes=None):
        if classes is not None:
            return classes
        return self.CLASSES

    def load_annotations(self, ann_file):
        return load_file(ann_file) if ann_file else []

    def get_data_info(self, index):
        raise NotImplementedError

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    def prepare_test_data(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        return self.prepare_test_data(index)

    def __len__(self):
        return len(self.data_infos)


@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    AttrMapping = {
        "cycle.with_rider": 0,
        "cycle.without_rider": 1,
        "pedestrian.moving": 2,
        "pedestrian.standing": 3,
        "pedestrian.sitting_lying_down": 4,
        "vehicle.moving": 5,
        "vehicle.parked": 6,
        "vehicle.stopped": 7,
    }
    AttrMapping_rev = [
        "cycle.with_rider",
        "cycle.without_rider",
        "pedestrian.moving",
        "pedestrian.standing",
        "pedestrian.sitting_lying_down",
        "vehicle.moving",
        "vehicle.parked",
        "vehicle.stopped",
    ]
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
    ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        try:
            from nuscenes.eval.detection.config import config_factory

            self.eval_detection_configs = config_factory(self.eval_version)
        except ImportError:
            self.eval_detection_configs = None
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def get_cat_ids(self, idx):
        """Get category distribution of single scene."""
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        if hasattr(self, "cat2id"):
            for name in gt_names:
                if name in self.CLASSES:
                    cat_ids.append(self.cat2id[name])
        else:
            for name in gt_names:
                if name in self.CLASSES:
                    cat_ids.append(self.CLASSES.index(name))
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file."""
        data = load_file(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index."""
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
        )

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                lidar2img_rts.append(cam_info["lidar2img"])

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                )
            )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index."""
        info = self.data_infos[index]
        if "gt_boxes" in info:
            gt_boxes = info["gt_boxes"]
            gt_names = info["gt_names"]
            gt_labels = []
            for cat in gt_names:
                if cat in self.CLASSES:
                    gt_labels.append(self.CLASSES.index(cat))
                else:
                    gt_labels.append(-1)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            annos = dict(
                gt_boxes=gt_boxes,
                gt_names=gt_names,
                gt_labels=gt_labels,
            )
        else:
            annos = dict()
        return annos

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend="disk"),
            ),
            dict(type="LoadPointsFromMultiSweeps", sweeps_num=10, file_client_args=dict(backend="disk")),
            dict(type="DefaultFormatBundle3D", class_names=self.CLASSES, with_label=False),
            dict(type="Collect3D", keys=["points"]),
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization."""
        assert out_dir is not None, "Expect out_dir, got none."
        pipeline = self._get_pipeline(pipeline) if pipeline else self._build_default_pipeline()
        for i, result in enumerate(results):
            if "pts_bbox" in result.keys():
                result = result["pts_bbox"]
            data_info = self.data_infos[i]
            pts_path = data_info["lidar_path"]
            file_name = osp.split(pts_path)[-1].split(".")[0]

    def _get_pipeline(self, pipeline):
        return Compose(pipeline) if pipeline else self.pipeline

    def _extract_data(self, index, pipeline, key):
        data_info = self.get_data_info(index)
        data = {key: None}
        return torch.tensor([])


class GroupSampler(Sampler):
    """Sampler that groups samples together."""

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, "flag")
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(math.ceil(size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(math.ceil(size / self.samples_per_gpu)) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu : (i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


Linear = nn.Linear


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


BBOX_CODERS = Registry("bbox_coder")


def build_bbox_coder(cfg, **default_args):
    """Builder of box coder."""
    return build_from_cfg(cfg, BBOX_CODERS, default_args)


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has same
            shape with input.

    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def collate(batch, samples_per_gpu=1):
    """Collate a batch of data.

    Args:
        batch (list): A list of dicts containing data.
        samples_per_gpu (int): Number of samples per GPU.

    Returns:
        dict: Collated batch data.
    """
    if not isinstance(batch, list):
        return batch

    if len(batch) == 0:
        return batch

    elem = batch[0]
    if elem is None:
        return None

    if isinstance(elem, dict):
        result = {}
        for key in elem:
            values = []
            for d in batch:
                if d is not None and key in d:
                    values.append(d[key])
                else:
                    values.append(None)

            if all(v is None for v in values):
                result[key] = None
            elif any(v is None for v in values):
                filtered_values = [v for v in values if v is not None]
                if len(filtered_values) > 0:
                    result[key] = collate(filtered_values, samples_per_gpu)
                else:
                    result[key] = None
            else:
                result[key] = collate(values, samples_per_gpu)
        return result
    elif isinstance(elem, DataContainer):
        if elem.stack:
            if isinstance(elem.data, torch.Tensor):
                return DataContainer(
                    default_collate([e.data for e in batch if e is not None and e.data is not None]),
                    stack=elem.stack,
                    padding_value=elem.padding_value,
                    cpu_only=elem.cpu_only,
                    pad_dims=elem.pad_dims,
                )
            elif isinstance(elem.data, (list, tuple)):
                return DataContainer(
                    type(elem.data)(
                        [
                            default_collate([e.data[i] for e in batch if e is not None and e.data is not None])
                            for i in range(len(elem.data))
                        ]
                    ),
                    stack=elem.stack,
                    padding_value=elem.padding_value,
                    cpu_only=elem.cpu_only,
                    pad_dims=elem.pad_dims,
                )
            else:
                return DataContainer(
                    [e.data for e in batch if e is not None],
                    stack=elem.stack,
                    padding_value=elem.padding_value,
                    cpu_only=elem.cpu_only,
                    pad_dims=elem.pad_dims,
                )
        else:
            return DataContainer(
                [e.data for e in batch if e is not None],
                stack=elem.stack,
                padding_value=elem.padding_value,
                cpu_only=elem.cpu_only,
                pad_dims=elem.pad_dims,
            )
    elif isinstance(elem, (list, tuple)):
        if len(elem) > 0 and isinstance(elem[0], DataContainer):
            collated_items = []
            for i in range(len(elem)):
                items_at_i = [b[i] for b in batch if b is not None and i < len(b) and b[i] is not None]
                if len(items_at_i) > 0:
                    collated_items.append(collate(items_at_i, samples_per_gpu))
                else:
                    collated_items.append(None)
            return type(elem)(collated_items)
        else:
            zipped = list(zip(*batch))
            return type(elem)([collate(list(samples), samples_per_gpu) for samples in zipped])
    elif elem is None:
        return None
    elif isinstance(elem, type):
        return batch
    elif isinstance(elem, (str, int, float, bool)):
        if len(set(batch)) == 1:
            return batch[0]
        return batch
    elif isinstance(elem, (torch.Tensor, np.ndarray)):
        filtered_batch = [b for b in batch if b is not None]
        if len(filtered_batch) == 0:
            return None
        return default_collate(filtered_batch)
    else:
        filtered_batch = [b for b in batch if b is not None]
        if len(filtered_batch) == 0:
            return None
        try:
            return default_collate(filtered_batch)
        except (TypeError, ValueError):
            return batch


@NECKS.register_module()
class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg=dict(mode="nearest"),
        init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
    ):
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
        elif add_extra_convs:  # True
            self.add_extra_convs = "on_input"

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        if self.fp16_enabled and len(inputs) > 0 and inputs[0].dtype == torch.half:
            # Convert all parameters to FP16 if they're still FP32
            for module_list in [self.lateral_convs, self.fpn_convs]:
                for m in module_list:
                    for param in m.parameters():
                        if param.dtype == torch.float32:
                            param.data = param.data.half()
                    for buffer in m.buffers():
                        if buffer.dtype == torch.float32:
                            buffer.data = buffer.data.half()

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if "scale_factor" in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == "on_input":
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == "on_lateral":
                    extra_source = laterals[-1]
                elif self.add_extra_convs == "on_output":
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


BACKBONES.register_module(name="ResNet", module=ResNet)


def replace_ImageToTensor(pipelines):
    """Replace ImageToTensor to DefaultFormatBundle in the pipeline.

    Args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list[dict]: The new pipeline list with all ImageToTensor replaced by
            DefaultFormatBundle.
    """
    pipelines = pipelines.copy()
    for i, pipeline in enumerate(pipelines):
        if pipeline["type"] == "ImageToTensor":
            pipelines[i] = {"type": "DefaultFormatBundle"}
    return pipelines


def bbox_overlaps(bboxes1, bboxes2, mode="iou", eps=1e-6, use_legacy_coordinate=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.

    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ["iou", "iof"]
    if not use_legacy_coordinate:
        extra_length = 0.0
    else:
        extra_length = 1.0
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(y_end - y_start + extra_length, 0)
        if mode == "iou":
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def build_model(cfg, train_cfg=None, test_cfg=None):
    """A function wrapper for building 3D detector or segmentor according to cfg.

    Should be deprecated in the future.
    """
    return DETECTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


@PIPELINES.register_module()
class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("transform must be callable or a dict")

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.separate_eval = separate_eval
        if not separate_eval:
            if len(set([type(ds) for ds in datasets])) != 1:
                raise NotImplementedError("All the datasets should have same types")

        if hasattr(datasets[0], "flag"):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

    def get_cat_ids(self, idx):
        """Get category ids of concatenated dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_cat_ids(sample_idx)

    def evaluate(self, results, logger=None, **kwargs):
        raise NotImplementedError("Evaluation is not supported in inference-only mode.")


def _concat_dataset(cfg, default_args=None):
    ann_files = cfg["ann_file"]
    img_prefixes = cfg.get("img_prefix", None)
    seg_prefixes = cfg.get("seg_prefix", None)
    proposal_files = cfg.get("proposal_file", None)
    separate_eval = cfg.get("separate_eval", True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if "separate_eval" in data_cfg:
            data_cfg.pop("separate_eval")
        data_cfg["ann_file"] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg["img_prefix"] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg["seg_prefix"] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg["proposal_file"] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def wrap_fp16_model(model):
    """Wrap the FP32 model to FP16.

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend, otherwise, original mmcv implementation will be adopted.

    For PyTorch >= 1.6, this function will
    1. Set fp16 flag inside the model to True.

    Otherwise:
    1. Convert FP32 model to FP16.
    2. Remain some necessary layers to be FP32, e.g., normalization layers.
    3. Set `fp16_enabled` flag inside the model to True.

    Args:
        model (nn.Module): Model in FP32.
    """
    if TORCH_VERSION == "parrots" or digit_version(TORCH_VERSION) < digit_version("1.6.0"):
        # convert model to fp16
        model.half()
    for m in model.modules():
        if hasattr(m, "fp16_enabled"):
            m.fp16_enabled = True


def is_module_wrapper(module):
    """Check if a module is a wrapper.

    The following modules are regarded as wrappers:
    1. DataParallel
    2. DistributedDataParallel
    3. ModelParallel (used in model parallelization)

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: True if the input module is a wrapper module.
    """
    module_wrappers = (DataParallel, DistributedDataParallel)
    return isinstance(module, module_wrappers)


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, all_missing_keys, unexpected_keys, err_msg
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None

    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append("unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(filename, map_location=None, logger=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`.
            Default: None.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    if not osp.isfile(filename):
        raise IOError(f"{filename} is not a checkpoint file")
    return torch.load(filename, map_location=map_location)


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None, revise_keys=[(r"^module\.", "")]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    original_keys_count = len(state_dict)
    print(f"[CHECKPOINT LOADING] Original checkpoint keys: {original_keys_count}")

    metadata = getattr(state_dict, "_metadata", OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict({re.sub(p, r, k): v for k, v in state_dict.items()})

    after_revise_keys_count = len(state_dict)
    print(f"[CHECKPOINT LOADING] Keys after revise_keys (remove 'module.' prefix): {after_revise_keys_count}")

    new_state_dict = OrderedDict()
    ffn_mapped_count = 0
    for k, v in state_dict.items():
        new_key = k
        if ".ffns.0.layers.0.0." in k:
            new_key = k.replace(".ffns.0.layers.0.0.", ".ffns.0.layers.0.")
            ffn_mapped_count += 1
        elif ".ffns.0.layers.1." in k:
            new_key = k.replace(".ffns.0.layers.1.", ".ffns.0.layers.3.")
            ffn_mapped_count += 1
        new_state_dict[new_key] = v

    after_ffn_mapping_count = len(new_state_dict)
    print(
        f"[CHECKPOINT LOADING] Keys after FFN mapping: {after_ffn_mapping_count} (FFN keys mapped: {ffn_mapped_count})"
    )

    model_state_dict = model.state_dict()
    model_keys_count = len(model_state_dict)
    print(f"[CHECKPOINT LOADING] Model state_dict keys: {model_keys_count}")

    checkpoint_keys_set = set(new_state_dict.keys())
    model_keys_set = set(model_state_dict.keys())

    missing_keys = model_keys_set - checkpoint_keys_set
    unexpected_keys = checkpoint_keys_set - model_keys_set

    missing_keys_filtered = {k for k in missing_keys if "num_batches_tracked" not in k}

    print(f"[CHECKPOINT LOADING] Missing keys (in model but not in checkpoint): {len(missing_keys_filtered)}")
    if missing_keys_filtered:
        for key in sorted(missing_keys_filtered)[:10]:
            print(f"  - {key}")
        if len(missing_keys_filtered) > 10:
            print(f"  ... and {len(missing_keys_filtered) - 10} more")

    print(f"[CHECKPOINT LOADING] Unexpected keys (in checkpoint but not in model): {len(unexpected_keys)}")
    if unexpected_keys:
        for key in sorted(unexpected_keys)[:10]:
            print(f"  - {key}")
        if len(unexpected_keys) > 10:
            print(f"  ... and {len(unexpected_keys) - 10} more")

    matched_keys = checkpoint_keys_set & model_keys_set
    print(f"[CHECKPOINT LOADING] Matched keys (will be loaded): {len(matched_keys)}")

    new_state_dict._metadata = metadata
    load_state_dict(model, new_state_dict, strict, logger)

    print(f"[CHECKPOINT LOADING] Checkpoint loading completed")
    return checkpoint


def dynamic_voxelize_pytorch(points, voxel_size, coors_range, max_points=-1):
    """Pure PyTorch implementation of dynamic voxelization."""
    voxel_size = torch.tensor(voxel_size, dtype=points.dtype, device=points.device)
    coors_range = torch.tensor(coors_range, dtype=points.dtype, device=points.device)

    coors = (points[:, :3] - coors_range[:3]) / voxel_size
    coors = coors.long()

    grid_size = ((coors_range[3:] - coors_range[:3]) / voxel_size).long()
    coors[:, 0] = torch.clamp(coors[:, 0], 0, grid_size[0] - 1)
    coors[:, 1] = torch.clamp(coors[:, 1], 0, grid_size[1] - 1)
    coors[:, 2] = torch.clamp(coors[:, 2], 0, grid_size[2] - 1)

    return coors


def hard_voxelize_pytorch(
    points,
    voxels,
    coors,
    num_points_per_voxel,
    voxel_size,
    coors_range,
    max_points,
    max_voxels,
    ndim=3,
    deterministic=True,
):
    """Pure PyTorch implementation of hard voxelization."""
    voxel_size = torch.tensor(voxel_size, dtype=points.dtype, device=points.device)
    coors_range = torch.tensor(coors_range, dtype=points.dtype, device=points.device)

    coors_temp = (points[:, :3] - coors_range[:3]) / voxel_size
    coors_temp = coors_temp.long()

    grid_size = ((coors_range[3:] - coors_range[:3]) / voxel_size).long()
    coors_temp[:, 0] = torch.clamp(coors_temp[:, 0], 0, grid_size[0] - 1)
    coors_temp[:, 1] = torch.clamp(coors_temp[:, 1], 0, grid_size[1] - 1)
    coors_temp[:, 2] = torch.clamp(coors_temp[:, 2], 0, grid_size[2] - 1)

    coors_flat = coors_temp[:, 0] * grid_size[1] * grid_size[2] + coors_temp[:, 1] * grid_size[2] + coors_temp[:, 2]

    unique_coors, inverse_indices, counts = torch.unique(coors_flat, return_inverse=True, return_counts=True)

    valid_mask = counts <= max_points
    valid_unique_coors = unique_coors[valid_mask]
    valid_counts = counts[valid_mask]

    if len(valid_unique_coors) > max_voxels:
        if deterministic:
            indices = torch.argsort(valid_counts, descending=True)[:max_voxels]
        else:
            indices = torch.randperm(len(valid_unique_coors), device=points.device)[:max_voxels]
        valid_unique_coors = valid_unique_coors[indices]
        valid_counts = valid_counts[indices]

    voxel_num = len(valid_unique_coors)

    coors_dict = {}
    for i, coor_flat in enumerate(valid_unique_coors):
        z = coor_flat % grid_size[2]
        y = (coor_flat // grid_size[2]) % grid_size[1]
        x = coor_flat // (grid_size[1] * grid_size[2])
        coors_dict[coor_flat.item()] = (x.item(), y.item(), z.item())

    coors[:voxel_num, 0] = torch.tensor(
        [coors_dict[coor.item()][0] for coor in valid_unique_coors], device=points.device, dtype=torch.int32
    )
    coors[:voxel_num, 1] = torch.tensor(
        [coors_dict[coor.item()][1] for coor in valid_unique_coors], device=points.device, dtype=torch.int32
    )
    coors[:voxel_num, 2] = torch.tensor(
        [coors_dict[coor.item()][2] for coor in valid_unique_coors], device=points.device, dtype=torch.int32
    )

    num_points_per_voxel[:voxel_num] = valid_counts.long()

    coor_to_voxel_idx = {coor.item(): idx for idx, coor in enumerate(valid_unique_coors)}

    point_to_voxel = torch.zeros(len(points), dtype=torch.long, device=points.device)
    for i, coor_flat in enumerate(coors_flat):
        if coor_flat.item() in coor_to_voxel_idx:
            point_to_voxel[i] = coor_to_voxel_idx[coor_flat.item()]

    for voxel_idx in range(voxel_num):
        point_indices = torch.where(point_to_voxel == voxel_idx)[0]
        num_points = min(len(point_indices), max_points)
        if num_points > 0:
            voxels[voxel_idx, :num_points] = points[point_indices[:num_points]]

    return voxel_num


class Voxelization(nn.Module):
    """Pure PyTorch implementation of Voxelization."""

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000, deterministic=True):
        super(Voxelization, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = (max_voxels, max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        self.pcd_shape = [*input_feat_shape, 1]

    def forward(self, input):
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        if max_voxels == -1 or self.max_num_points == -1:
            coors = dynamic_voxelize_pytorch(
                input,
                self.voxel_size,
                self.point_cloud_range,
                self.max_num_points,
            )
            return coors
        else:
            voxels = input.new_zeros(size=(max_voxels, self.max_num_points, input.size(1)))
            coors = input.new_zeros(size=(max_voxels, 3), dtype=torch.int32)
            num_points_per_voxel = input.new_zeros(size=(max_voxels,), dtype=torch.int32)
            voxel_num = hard_voxelize_pytorch(
                input,
                voxels,
                coors,
                num_points_per_voxel,
                self.voxel_size,
                self.point_cloud_range,
                self.max_num_points,
                max_voxels,
                3,
                self.deterministic,
            )
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            return voxels_out, coors_out, num_points_per_voxel_out

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "voxel_size=" + str(self.voxel_size)
        tmpstr += ", point_cloud_range=" + str(self.point_cloud_range)
        tmpstr += ", max_num_points=" + str(self.max_num_points)
        tmpstr += ", max_voxels=" + str(self.max_voxels)
        tmpstr += ", deterministic=" + str(self.deterministic)
        tmpstr += ")"
        return tmpstr


def dynamic_point_to_voxel_forward_pytorch(feats, coors, reduce_type="max"):
    """Pure PyTorch implementation of dynamic point to voxel forward."""
    if coors.size(-1) == 3:
        coors_max = coors.max(dim=0)[0]
        coors_flat = (
            coors[:, 0] * (coors_max[1] + 1) * (coors_max[2] + 1) + coors[:, 1] * (coors_max[2] + 1) + coors[:, 2]
        )
    else:
        coors_max = coors.max(dim=0)[0]
        coors_flat = (
            coors[:, 0] * (coors_max[1] + 1) * (coors_max[2] + 1) * (coors_max[3] + 1)
            + coors[:, 1] * (coors_max[2] + 1) * (coors_max[3] + 1)
            + coors[:, 2] * (coors_max[3] + 1)
            + coors[:, 3]
        )

    unique_coors, inverse_indices, counts = torch.unique(coors_flat, return_inverse=True, return_counts=True)

    point2voxel_map = inverse_indices
    voxel_points_count = counts

    unique_coors_list = []
    for coor_flat in unique_coors:
        if coors.size(-1) == 3:
            coors_max = coors.max(dim=0)[0]
            z = coor_flat % (coors_max[2] + 1)
            y = (coor_flat // (coors_max[2] + 1)) % (coors_max[1] + 1)
            x = coor_flat // ((coors_max[1] + 1) * (coors_max[2] + 1))
            unique_coors_list.append([x.item(), y.item(), z.item()])
        else:
            coors_max = coors.max(dim=0)[0]
            w = coor_flat % (coors_max[3] + 1)
            z = (coor_flat // (coors_max[3] + 1)) % (coors_max[2] + 1)
            y = (coor_flat // ((coors_max[2] + 1) * (coors_max[3] + 1))) % (coors_max[1] + 1)
            x = coor_flat // ((coors_max[1] + 1) * (coors_max[2] + 1) * (coors_max[3] + 1))
            unique_coors_list.append([x.item(), y.item(), z.item(), w.item()])

    voxel_coors = torch.tensor(unique_coors_list, dtype=coors.dtype, device=coors.device)

    if reduce_type == "max":
        voxel_feats = torch.zeros((len(unique_coors), feats.size(1)), dtype=feats.dtype, device=feats.device)
        for i in range(len(unique_coors)):
            mask = point2voxel_map == i
            if mask.any():
                voxel_feats[i] = feats[mask].max(dim=0)[0]
    elif reduce_type == "sum":
        voxel_feats = torch.zeros((len(unique_coors), feats.size(1)), dtype=feats.dtype, device=feats.device)
        for i in range(len(unique_coors)):
            mask = point2voxel_map == i
            if mask.any():
                voxel_feats[i] = feats[mask].sum(dim=0)
    elif reduce_type == "mean":
        voxel_feats = torch.zeros((len(unique_coors), feats.size(1)), dtype=feats.dtype, device=feats.device)
        for i in range(len(unique_coors)):
            mask = point2voxel_map == i
            if mask.any():
                voxel_feats[i] = feats[mask].mean(dim=0)
    else:
        raise ValueError(f"Unsupported reduce_type: {reduce_type}")

    return voxel_feats, voxel_coors, point2voxel_map, voxel_points_count


class DynamicScatter(nn.Module):
    """Pure PyTorch implementation of DynamicScatter."""

    def __init__(self, voxel_size, point_cloud_range, average_points: bool):
        super(DynamicScatter, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward_single(self, points, coors):
        reduce = "mean" if self.average_points else "max"
        voxel_feats, voxel_coors, _, _ = dynamic_point_to_voxel_forward_pytorch(
            points.contiguous(), coors.contiguous(), reduce
        )
        return voxel_feats, voxel_coors

    def forward(self, points, coors):
        if coors.size(-1) == 3:
            return self.forward_single(points, coors)
        else:
            batch_size = coors[-1, 0] + 1
            voxels, voxel_coors = [], []
            for i in range(batch_size):
                inds = torch.where(coors[:, 0] == i)
                voxel, voxel_coor = self.forward_single(points[inds], coors[inds][:, 1:])
                coor_pad = nn.functional.pad(voxel_coor, (1, 0), mode="constant", value=i)
                voxel_coors.append(coor_pad)
                voxels.append(voxel)
            features = torch.cat(voxels, dim=0)
            feature_coors = torch.cat(voxel_coors, dim=0)
            return features, feature_coors

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "voxel_size=" + str(self.voxel_size)
        tmpstr += ", point_cloud_range=" + str(self.point_cloud_range)
        tmpstr += ", average_points=" + str(self.average_points)
        tmpstr += ")"
        return tmpstr


# Create a builder module-like object with build_middle_encoder function
class BuilderModule:
    """Builder module for mmdet3d models."""

    def build_middle_encoder(self, cfg):
        """Build middle level encoder."""
        return MIDDLE_ENCODERS.build(cfg)

    def build_voxel_encoder(self, cfg):
        """Build voxel encoder."""
        return VOXEL_ENCODERS.build(cfg)

    def build_fusion_layer(self, cfg):
        """Build fusion layer."""
        return FUSION_LAYERS.build(cfg)


builder = BuilderModule()


# Pure PyTorch implementation without CUDA dependencies
class QuickCumsumPytorch(torch.autograd.Function):
    """Pure PyTorch implementation of QuickCumsum."""

    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        if len(interval_lengths) > 1:
            interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        if len(interval_starts) > 0:
            interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        geom_feats = geom_feats.int()

        # Initialize output tensor: (B, D, H, W, C)
        out = torch.zeros((B, D, H, W, x.shape[1]), dtype=x.dtype, device=x.device)

        # For each interval, sum the features and place in the output
        for i, start_idx in enumerate(interval_starts):
            length = interval_lengths[i]
            if length > 0 and start_idx < geom_feats.shape[0]:
                b, d, h, w = geom_feats[start_idx].tolist()
                if 0 <= b < B and 0 <= d < D and 0 <= h < H and 0 <= w < W:
                    # Sum features in this interval
                    out[b, d, h, w] += x[start_idx : start_idx + length].sum(0)

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        # Compute total size needed
        total_size = interval_starts[-1] + interval_lengths[-1] if len(interval_lengths) > 0 else 0
        x_grad = torch.zeros((total_size, out_grad.shape[-1]), dtype=out_grad.dtype, device=out_grad.device)

        # Distribute gradients back
        for i, start_idx in enumerate(interval_starts):
            length = interval_lengths[i]
            if length > 0 and start_idx < geom_feats.shape[0]:
                b, d, h, w = geom_feats[start_idx].tolist()
                if 0 <= b < B and 0 <= d < D and 0 <= h < H and 0 <= w < W:
                    grad_val = out_grad[b, d, h, w]
                    x_grad[start_idx : start_idx + length] = grad_val

        return x_grad, None, None, None, None, None, None


def bev_pool(feats, coords, B, D, H, W):
    """Pure PyTorch implementation of BEV pooling.


    Args:
        feats: (N, C) feature tensor
        coords: (N, 4) coordinate tensor [x, y, z, batch_idx]
        B: batch size
        D: depth dimension
        H: height dimension
        W: width dimension

    Returns:
        output: (B, C, D, H, W) pooled BEV features
    """
    assert feats.shape[0] == coords.shape[0]

    # Compute ranks for sorting
    ranks = coords[:, 0] * (W * D * B) + coords[:, 1] * (D * B) + coords[:, 2] * B + coords[:, 3]

    # Sort by ranks
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    # Apply pooling
    x = QuickCumsumPytorch.apply(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


# Import build_dataloader
from models.experimental.MapTR.reference.datasets import build_dataloader  # noqa: F401
