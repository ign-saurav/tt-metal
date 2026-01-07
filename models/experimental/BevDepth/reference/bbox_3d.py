# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

########################################################
# Adapted from: https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/bbox_3d/base_box3d.py
# Copyright (c) OpenMMLab. All rights reserved.
########################################################
import numpy as np
import torch
from torch import Tensor
import functools
from inspect import getfullargspec
import warnings
from abc import abstractmethod
from typing import Iterator, Optional, Sequence, Tuple, Union, Callable, Type


########################################################
# Adapted from https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/points/base_points.py
########################################################
class BasePoints:
    """Base class for Points.

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The points
            data with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...). Defaults to 3.
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...).
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        points_dim: int = 3,
        attribute_dims: Optional[dict] = None,
    ) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, points_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == points_dim, (
            "The points dimension must be 2 and the length of the last "
            f"dimension must be {points_dim}, but got points with shape "
            f"{tensor.shape}."
        )

        self.tensor = tensor.clone()
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.rotation_axis = 0

    @property
    def coord(self) -> Tensor:
        """Tensor: Coordinates of each point in shape (N, 3)."""
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the coordinates of each point.

        Args:
            tensor (Tensor or np.ndarray): Coordinates of each point with shape
                (N, 3).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        self.tensor[:, :3] = tensor

    @property
    def height(self) -> Union[Tensor, None]:
        """Tensor or None: Returns a vector with height of each point in shape
        (N, )."""
        if self.attribute_dims is not None and "height" in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims["height"]]
        else:
            return None

    @height.setter
    def height(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the height of each point.

        Args:
            tensor (Tensor or np.ndarray): Height of each point with shape
                (N, ).
        """
        try:
            tensor = tensor.reshape(self.shape[0])
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and "height" in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims["height"]] = tensor
        else:
            # add height attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor.unsqueeze(1)], dim=1)
            self.attribute_dims.update(dict(height=attr_dim))
            self.points_dim += 1

    @property
    def color(self) -> Union[Tensor, None]:
        """Tensor or None: Returns a vector with color of each point in shape
        (N, 3)."""
        if self.attribute_dims is not None and "color" in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims["color"]]
        else:
            return None

    @color.setter
    def color(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the color of each point.

        Args:
            tensor (Tensor or np.ndarray): Color of each point with shape
                (N, 3).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if tensor.max() >= 256 or tensor.min() < 0:
            warnings.warn("point got color value beyond [0, 255]")
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and "color" in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims["color"]] = tensor
        else:
            # add color attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(dict(color=[attr_dim, attr_dim + 1, attr_dim + 2]))
            self.points_dim += 3

    @property
    def shape(self) -> torch.Size:
        """torch.Size: Shape of points."""
        return self.tensor.shape

    def shuffle(self) -> Tensor:
        """Shuffle the points.

        Returns:
            Tensor: The shuffled index.
        """
        idx = torch.randperm(self.__len__(), device=self.tensor.device)
        self.tensor = self.tensor[idx]
        return idx

    def rotate(self, rotation: Union[Tensor, np.ndarray, float], axis: Optional[int] = None) -> Tensor:
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (Tensor or np.ndarray or float): Rotation matrix or angle.
            axis (int, optional): Axis to rotate at. Defaults to None.

        Returns:
            Tensor: Rotation matrix.
        """
        if not isinstance(rotation, Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1, f"invalid rotation shape {rotation.shape}"

        if axis is None:
            axis = self.rotation_axis

        if rotation.numel() == 1:
            rotated_points, rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, :3][None], rotation, axis=axis, return_mat=True
            )
            self.tensor[:, :3] = rotated_points.squeeze(0)
            rot_mat_T = rot_mat_T.squeeze(0)
        else:
            # rotation.numel() == 9
            self.tensor[:, :3] = self.tensor[:, :3] @ rotation
            rot_mat_T = rotation

        return rot_mat_T

    @abstractmethod
    def flip(self, bev_direction: str = "horizontal") -> None:
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
                Defaults to 'horizontal'.
        """

    def translate(self, trans_vector: Union[Tensor, np.ndarray]) -> None:
        """Translate points with the given translation vector.

        Args:
            trans_vector (Tensor or np.ndarray): Translation vector of size 3
                or nx3.
        """
        if not isinstance(trans_vector, Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert trans_vector.shape[0] == self.tensor.shape[0] and trans_vector.shape[1] == 3
        else:
            raise NotImplementedError(f"Unsupported translation vector of shape {trans_vector.shape}")
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, point_range: Union[Tensor, np.ndarray, Sequence[float]]) -> Tensor:
        """Check whether the points are in the given range.

        Args:
            point_range (Tensor or np.ndarray or Sequence[float]): The range of
                point (x_min, y_min, z_min, x_max, y_max, z_max).

        Note:
            In the original implementation of SECOND, checking whether a box in
            the range checks whether the points are in a convex polygon, we try
            to reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
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

    @property
    def bev(self) -> Tensor:
        """Tensor: BEV of the points in shape (N, 2)."""
        return self.tensor[:, [0, 1]]

    def in_range_bev(self, point_range: Union[Tensor, np.ndarray, Sequence[float]]) -> Tensor:
        """Check whether the points are in the given range.

        Args:
            point_range (Tensor or np.ndarray or Sequence[float]): The range of
                point in order of (x_min, y_min, x_max, y_max).

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = (
            (self.bev[:, 0] > point_range[0])
            & (self.bev[:, 1] > point_range[1])
            & (self.bev[:, 0] < point_range[2])
            & (self.bev[:, 1] < point_range[3])
        )
        return in_range_flags

    @abstractmethod
    def convert_to(self, dst: int, rt_mat: Optional[Union[Tensor, np.ndarray]] = None) -> "BasePoints":
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Point mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type in the
            ``dst`` mode.
        """

    def scale(self, scale_factor: float) -> None:
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor

    def __getitem__(self, item: Union[int, tuple, slice, np.ndarray, Tensor]) -> "BasePoints":
        """
        Args:
            item (int or tuple or slice or np.ndarray or Tensor): Index of
                points.

        Note:
            The following usage are allowed:

            1. `new_points = points[3]`: Return a `Points` that contains only
               one point.
            2. `new_points = points[2:10]`: Return a slice of points.
            3. `new_points = points[vector]`: Whether vector is a
               torch.BoolTensor with `length = len(points)`. Nonzero elements
               in the vector will be selected.
            4. `new_points = points[3:11, vector]`: Return a slice of points
               and attribute dims.
            5. `new_points = points[4:12, 2]`: Return a slice of points with
               single attribute.

            Note that the returned Points might share storage with this Points,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BasePoints`: A new object of :class:`BasePoints` after
            indexing.
        """
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
        elif isinstance(item, (slice, np.ndarray, Tensor)):
            p = self.tensor[item]
            attribute_dims = self.attribute_dims
        else:
            raise NotImplementedError(f"Invalid slice {item}!")

        assert p.dim() == 2, f"Indexing on Points with {item} failed to return a matrix!"
        return original_type(p, points_dim=p.shape[1], attribute_dims=attribute_dims)

    def __len__(self) -> int:
        """int: Number of points in the current object."""
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        """str: Return a string that describes the object."""
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    @classmethod
    def cat(cls, points_list: Sequence["BasePoints"]) -> "BasePoints":
        """Concatenate a list of Points into a single Points.

        Args:
            points_list (Sequence[:obj:`BasePoints`]): List of points.

        Returns:
            :obj:`BasePoints`: The concatenated points.
        """
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(points, cls) for points in points_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned points never share storage with input
        cat_points = cls(
            torch.cat([p.tensor for p in points_list], dim=0),
            points_dim=points_list[0].points_dim,
            attribute_dims=points_list[0].attribute_dims,
        )
        return cat_points

    def numpy(self) -> np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self, device: Union[str, torch.device], *args, **kwargs) -> "BasePoints":
        """Convert current points to a specific device.

        Args:
            device (str or :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BasePoints`: A new points object on the specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device, *args, **kwargs), points_dim=self.points_dim, attribute_dims=self.attribute_dims
        )

    def cpu(self) -> "BasePoints":
        """Convert current points to cpu device.

        Returns:
            :obj:`BasePoints`: A new points object on the cpu device.
        """
        original_type = type(self)
        return original_type(self.tensor.cpu(), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    def cuda(self, *args, **kwargs) -> "BasePoints":
        """Convert current points to cuda device.

        Returns:
            :obj:`BasePoints`: A new points object on the cuda device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cuda(*args, **kwargs), points_dim=self.points_dim, attribute_dims=self.attribute_dims
        )

    def clone(self) -> "BasePoints":
        """Clone the points.

        Returns:
            :obj:`BasePoints`: Point object with the same properties as self.
        """
        original_type = type(self)
        return original_type(self.tensor.clone(), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    def detach(self) -> "BasePoints":
        """Detach the points.

        Returns:
            :obj:`BasePoints`: Point object with the same properties as self.
        """
        original_type = type(self)
        return original_type(self.tensor.detach(), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    @property
    def device(self) -> torch.device:
        """torch.device: The device of the points are on."""
        return self.tensor.device

    def __iter__(self) -> Iterator[Tensor]:
        """Yield a point as a Tensor at a time.

        Returns:
            Iterator[Tensor]: A point of shape (points_dim, ).
        """
        yield from self.tensor

    def new_point(self, data: Union[Tensor, np.ndarray, Sequence[Sequence[float]]]) -> "BasePoints":
        """Create a new point object with data.

        The new point and its tensor has the similar properties as self and
        self.tensor, respectively.

        Args:
            data (Tensor or np.ndarray or Sequence[Sequence[float]]): Data to
                be copied.

        Returns:
            :obj:`BasePoints`: A new point object with ``data``, the object's
            other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) if not isinstance(data, Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(new_tensor, points_dim=self.points_dim, attribute_dims=self.attribute_dims)


class BaseInstance3DBoxes:
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in the
        box is (0.5, 0.5, 0).

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The boxes
            data with shape (N, box_dim).
        box_dim (int): Number of the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw). Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation. If False, the
            value of yaw will be set to 0 as minmax boxes. Defaults to True.
        origin (Tuple[float]): Relative position of the box origin.
            Defaults to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    YAW_AXIS: int = 0


########################################################
# Adapted from: https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/utils/array_converter.py
########################################################
TemplateArrayType = Union[np.ndarray, torch.Tensor, list, tuple, int, float]


def array_converter(
    to_torch: bool = True,
    apply_to: Tuple[str, ...] = tuple(),
    template_arg_name_: Optional[str] = None,
    recover: bool = True,
) -> Callable:
    """Wrapper function for data-type agnostic processing.

    First converts input arrays to PyTorch tensors or NumPy arrays for middle
    calculation, then convert output to original data-type if `recover=True`.

    Args:
        to_torch (bool): Whether to convert to PyTorch tensors for middle
            calculation. Defaults to True.
        apply_to (Tuple[str]): The arguments to which we apply data-type
            conversion. Defaults to an empty tuple.
        template_arg_name_ (str, optional): Argument serving as the template
            (return arrays should have the same dtype and device as the
            template). Defaults to None. If None, we will use the first
            argument in `apply_to` as the template argument.
        recover (bool): Whether or not to recover the wrapped function outputs
            to the `template_arg_name_` type. Defaults to True.

    Raises:
        ValueError: When template_arg_name_ is not among all args, or when
            apply_to contains an arg which is not among all args, a ValueError
            will be raised. When the template argument or an argument to
            convert is a list or tuple, and cannot be converted to a NumPy
            array, a ValueError will be raised.
        TypeError: When the type of the template argument or an argument to
            convert does not belong to the above range, or the contents of such
            an list-or-tuple-type argument do not share the same data type, a
            TypeError will be raised.

    Returns:
        Callable: Wrapped function.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Use torch addition for a + b,
        >>> # and convert return values to the type of a
        >>> @array_converter(apply_to=('a', 'b'))
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> a = np.array([1.1])
        >>> b = np.array([2.2])
        >>> simple_add(a, b)
        >>>
        >>> # Use numpy addition for a + b,
        >>> # and convert return values to the type of b
        >>> @array_converter(to_torch=False, apply_to=('a', 'b'),
        >>>                  template_arg_name_='b')
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> simple_add(a, b)
        >>>
        >>> # Use torch funcs for floor(a) if flag=True else ceil(a),
        >>> # and return the torch tensor
        >>> @array_converter(apply_to=('a',), recover=False)
        >>> def floor_or_ceil(a, flag=True):
        >>>     return torch.floor(a) if flag else torch.ceil(a)
        >>>
        >>> floor_or_ceil(a, flag=False)
    """

    def array_converter_wrapper(func):
        """Outer wrapper for the function."""

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            """Inner wrapper for the arguments."""
            if len(apply_to) == 0:
                return func(*args, **kwargs)

            func_name = func.__name__

            arg_spec = getfullargspec(func)

            arg_names = arg_spec.args
            arg_num = len(arg_names)
            default_arg_values = arg_spec.defaults
            if default_arg_values is None:
                default_arg_values = []
            no_default_arg_num = len(arg_names) - len(default_arg_values)

            kwonly_arg_names = arg_spec.kwonlyargs
            kwonly_default_arg_values = arg_spec.kwonlydefaults
            if kwonly_default_arg_values is None:
                kwonly_default_arg_values = {}

            all_arg_names = arg_names + kwonly_arg_names

            # in case there are args in the form of *args
            if len(args) > arg_num:
                named_args = args[:arg_num]
                nameless_args = args[arg_num:]
            else:
                named_args = args
                nameless_args = []

            # template argument data type is used for all array-like arguments
            if template_arg_name_ is None:
                template_arg_name = apply_to[0]
            else:
                template_arg_name = template_arg_name_

            if template_arg_name not in all_arg_names:
                raise ValueError(f"{template_arg_name} is not among the " f"argument list of function {func_name}")

            # inspect apply_to
            for arg_to_apply in apply_to:
                if arg_to_apply not in all_arg_names:
                    raise ValueError(f"{arg_to_apply} is not an argument of {func_name}")

            new_args = []
            new_kwargs = {}

            converter = ArrayConverter()
            target_type = torch.Tensor if to_torch else np.ndarray

            # non-keyword arguments
            for i, arg_value in enumerate(named_args):
                if arg_names[i] in apply_to:
                    new_args.append(converter.convert(input_array=arg_value, target_type=target_type))
                else:
                    new_args.append(arg_value)

                if arg_names[i] == template_arg_name:
                    template_arg_value = arg_value

            kwonly_default_arg_values.update(kwargs)
            kwargs = kwonly_default_arg_values

            # keyword arguments and non-keyword arguments using default value
            for i in range(len(named_args), len(all_arg_names)):
                arg_name = all_arg_names[i]
                if arg_name in kwargs:
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(input_array=kwargs[arg_name], target_type=target_type)
                    else:
                        new_kwargs[arg_name] = kwargs[arg_name]
                else:
                    default_value = default_arg_values[i - no_default_arg_num]
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(input_array=default_value, target_type=target_type)
                    else:
                        new_kwargs[arg_name] = default_value
                if arg_name == template_arg_name:
                    template_arg_value = kwargs[arg_name]

            # add nameless args provided by *args (if exists)
            new_args += nameless_args

            return_values = func(*new_args, **new_kwargs)
            converter.set_template(template_arg_value)

            def recursive_recover(input_data):
                if isinstance(input_data, (tuple, list)):
                    new_data = []
                    for item in input_data:
                        new_data.append(recursive_recover(item))
                    return tuple(new_data) if isinstance(input_data, tuple) else new_data
                elif isinstance(input_data, dict):
                    new_data = {}
                    for k, v in input_data.items():
                        new_data[k] = recursive_recover(v)
                    return new_data
                elif isinstance(input_data, (torch.Tensor, np.ndarray)):
                    return converter.recover(input_data)
                else:
                    return input_data

            if recover:
                return recursive_recover(return_values)
            else:
                return return_values

        return new_func

    return array_converter_wrapper


class ArrayConverter:
    """Utility class for data-type agnostic processing.

    Args:
        template_array (np.ndarray or torch.Tensor or list or tuple or int or
            float, optional): Template array. Defaults to None.
    """

    SUPPORTED_NON_ARRAY_TYPES = (
        int,
        float,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    )

    def __init__(self, template_array: Optional[TemplateArrayType] = None) -> None:
        if template_array is not None:
            self.set_template(template_array)

    def set_template(self, array: TemplateArrayType) -> None:
        """Set template array.

        Args:
            array (np.ndarray or torch.Tensor or list or tuple or int or
                float): Template array.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to a
                NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range, or the
                contents of a list or tuple do not share the same data type, a
                TypeError is raised.
        """
        self.array_type = type(array)
        self.is_num = False
        self.device = "cpu"

        if isinstance(array, np.ndarray):
            self.dtype = array.dtype
        elif isinstance(array, torch.Tensor):
            self.dtype = array.dtype
            self.device = array.device
        elif isinstance(array, (list, tuple)):
            try:
                array = np.array(array)
                if array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
                self.dtype = array.dtype
            except (ValueError, TypeError):
                print("The following list cannot be converted to a numpy " f"array of supported dtype:\n{array}")
                raise
        elif isinstance(array, (int, float)):
            self.array_type = np.ndarray
            self.is_num = True
            self.dtype = np.dtype(type(array))
        else:
            raise TypeError(f"Template type {self.array_type} is not supported.")

    def convert(
        self,
        input_array: TemplateArrayType,
        target_type: Optional[Type] = None,
        target_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Convert input array to target data type.

        Args:
            input_array (np.ndarray or torch.Tensor or list or tuple or int or
                float): Input array.
            target_type (Type, optional): Type to which input array is
                converted. It should be `np.ndarray` or `torch.Tensor`.
                Defaults to None.
            target_array (np.ndarray or torch.Tensor, optional): Template array
                to which input array is converted. Defaults to None.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to a
                NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range, or the
                contents of a list or tuple do not share the same data type, a
                TypeError is raised.

        Returns:
            np.ndarray or torch.Tensor: The converted array.
        """
        if isinstance(input_array, (list, tuple)):
            try:
                input_array = np.array(input_array)
                if input_array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
            except (ValueError, TypeError):
                print("The input cannot be converted to a single-type numpy " f"array:\n{input_array}")
                raise
        elif isinstance(input_array, self.SUPPORTED_NON_ARRAY_TYPES):
            input_array = np.array(input_array)
        array_type = type(input_array)
        assert target_type is not None or target_array is not None, "must specify a target"
        if target_type is not None:
            assert target_type in (np.ndarray, torch.Tensor), "invalid target type"
            if target_type == array_type:
                return input_array
            elif target_type == np.ndarray:
                # default dtype is float32
                converted_array = input_array.cpu().numpy().astype(np.float32)
            else:
                # default dtype is float32, device is 'cpu'
                converted_array = torch.tensor(input_array, dtype=torch.float32)
        else:
            assert isinstance(target_array, (np.ndarray, torch.Tensor)), "invalid target array type"
            if isinstance(target_array, array_type):
                return input_array
            elif isinstance(target_array, np.ndarray):
                converted_array = input_array.cpu().numpy().astype(target_array.dtype)
            else:
                converted_array = target_array.new_tensor(input_array)
        return converted_array

    def recover(self, input_array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor, int, float]:
        """Recover input type to original array type.

        Args:
            input_array (np.ndarray or torch.Tensor): Input array.

        Returns:
            np.ndarray or torch.Tensor or int or float: Converted array.
        """
        assert isinstance(input_array, (np.ndarray, torch.Tensor)), "invalid input array type"
        if isinstance(input_array, self.array_type):
            return input_array
        elif isinstance(input_array, torch.Tensor):
            converted_array = input_array.cpu().numpy().astype(self.dtype)
        else:
            converted_array = torch.tensor(input_array, dtype=self.dtype, device=self.device)
        if self.is_num:
            converted_array = converted_array.item()
        return converted_array


########################################################
# Adapted from: https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/bbox_3d/utils.py
########################################################
@array_converter(apply_to=("val",))
def limit_period(
    val: Union[np.ndarray, Tensor], offset: float = 0.5, period: float = np.pi
) -> Union[np.ndarray, Tensor]:
    """Limit the value into a period for periodic function.

    Args:
        val (np.ndarray or Tensor): The value to be converted.
        offset (float): Offset to set the value range. Defaults to 0.5.
        period (float): Period of the value. Defaults to np.pi.

    Returns:
        np.ndarray or Tensor: Value in the range of
        [-offset * period, (1-offset) * period].
    """
    limited_val = val - torch.floor(val / period + offset) * period
    return limited_val


@array_converter(apply_to=("points", "angles"))
def rotation_3d_in_axis(
    points: Union[np.ndarray, Tensor],
    angles: Union[np.ndarray, Tensor, float],
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor], np.ndarray, Tensor]:
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray or Tensor): Points with shape (N, M, 3).
        angles (np.ndarray or Tensor or float): Vector of angles with shape
            (N, ).
        axis (int): The axis to be rotated. Defaults to 0.
        return_mat (bool): Whether or not to return the rotation matrix
            (transposed). Defaults to False.
        clockwise (bool): Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: When the axis is not in range [-3, -2, -1, 0, 1, 2], it
            will raise ValueError.

    Returns:
        Tuple[np.ndarray, np.ndarray] or Tuple[Tensor, Tensor] or np.ndarray or
        Tensor: Rotated points with shape (N, M, 3) and rotation matrix with
        shape (N, 3, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 and points.shape[0] == angles.shape[0], (
        "Incorrect shape of points " f"angles: {points.shape}, {angles.shape}"
    )

    assert points.shape[-1] in [2, 3], f"Points size should be 2 or 3 instead of {points.shape[-1]}"

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack(
                [
                    torch.stack([rot_cos, zeros, -rot_sin]),
                    torch.stack([zeros, ones, zeros]),
                    torch.stack([rot_sin, zeros, rot_cos]),
                ]
            )
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack(
                [
                    torch.stack([rot_cos, rot_sin, zeros]),
                    torch.stack([-rot_sin, rot_cos, zeros]),
                    torch.stack([zeros, zeros, ones]),
                ]
            )
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack(
                [
                    torch.stack([ones, zeros, zeros]),
                    torch.stack([zeros, rot_cos, rot_sin]),
                    torch.stack([zeros, -rot_sin, rot_cos]),
                ]
            )
        else:
            raise ValueError(f"axis should in range [-3, -2, -1, 0, 1, 2], got {axis}")
    else:
        rot_mat_T = torch.stack([torch.stack([rot_cos, rot_sin]), torch.stack([-rot_sin, rot_cos])])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum("aij,jka->aik", points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum("jka->ajk", rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


@array_converter(apply_to=("boxes_xywhr",))
def xywhr2xyxyr(boxes_xywhr: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (Tensor or np.ndarray): Rotated boxes in XYWHR format.

    Returns:
        Tensor or np.ndarray: Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[..., 2] / 2
    half_h = boxes_xywhr[..., 3] / 2

    boxes[..., 0] = boxes_xywhr[..., 0] - half_w
    boxes[..., 1] = boxes_xywhr[..., 1] - half_h
    boxes[..., 2] = boxes_xywhr[..., 0] + half_w
    boxes[..., 3] = boxes_xywhr[..., 1] + half_h
    boxes[..., 4] = boxes_xywhr[..., 4]
    return boxes


def get_box_type(box_type: str) -> Tuple[type, int]:
    """Get the type and mode of box structure.

    Args:
        box_type (str): The type of box structure. The valid value are "LiDAR",
            "Camera" and "Depth".

    Raises:
        ValueError: A ValueError is raised when ``box_type`` does not belong to
            the three valid types.

    Returns:
        tuple: Box type and box mode.
    """
    from .box_3d_mode import Box3DMode, CameraInstance3DBoxes, DepthInstance3DBoxes, LiDARInstance3DBoxes

    box_type_lower = box_type.lower()
    if box_type_lower == "lidar":
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.LIDAR
    elif box_type_lower == "camera":
        box_type_3d = CameraInstance3DBoxes
        box_mode_3d = Box3DMode.CAM
    elif box_type_lower == "depth":
        box_type_3d = DepthInstance3DBoxes
        box_mode_3d = Box3DMode.DEPTH
    else:
        raise ValueError('Only "box_type" of "camera", "lidar", "depth" are ' f"supported, got {box_type}")

    return box_type_3d, box_mode_3d


@array_converter(apply_to=("points_3d", "proj_mat"))
def points_cam2img(
    points_3d: Union[Tensor, np.ndarray], proj_mat: Union[Tensor, np.ndarray], with_depth: bool = False
) -> Union[Tensor, np.ndarray]:
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (Tensor or np.ndarray): Points in shape (N, 3).
        proj_mat (Tensor or np.ndarray): Transformation matrix between
            coordinates.
        with_depth (bool): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        Tensor or np.ndarray: Points in image coordinates with shape [N, 2] if
        ``with_depth=False``, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, (
        "The dimension of the projection matrix should be 2 " f"instead of {len(proj_mat.shape)}."
    )
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (d1 == 4 and d2 == 4), (
        "The shape of the projection matrix " f"({d1}*{d2}) is not supported."
    )
    if d1 == 3:
        proj_mat_expanded = torch.eye(4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res


@array_converter(apply_to=("points", "cam2img"))
def points_img2cam(points: Union[Tensor, np.ndarray], cam2img: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Project points in image coordinates to camera coordinates.

    Args:
        points (Tensor or np.ndarray): 2.5D points in 2D images with shape
            [N, 3], 3 corresponds with x, y in the image and depth.
        cam2img (Tensor or np.ndarray): Camera intrinsic matrix. The shape can
            be [3, 3], [3, 4] or [4, 4].

    Returns:
        Tensor or np.ndarray: Points in 3D space with shape [N, 3], 3
        corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[: cam2img.shape[0], : cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D


def mono_cam_box2vis(cam_box):
    """This is a post-processing function on the bboxes from Mono-3D task. If
    we want to perform projection visualization, we need to:

        1. rotate the box along x-axis for np.pi / 2 (roll)
        2. change orientation from local yaw to global yaw
        3. convert yaw by (np.pi / 2 - yaw)

    After applying this function, we can project and draw it on 2D images.

    Args:
        cam_box (:obj:`CameraInstance3DBoxes`): 3D bbox in camera coordinate
            system before conversion. Could be gt bbox loaded from dataset or
            network prediction output.

    Returns:
        :obj:`CameraInstance3DBoxes`: Box after conversion.
    """
    warning.warn(
        "DeprecationWarning: The hack of yaw and dimension in the "
        "monocular 3D detection on nuScenes has been removed. The "
        "function mono_cam_box2vis will be deprecated."
    )
    from .cam_box3d import CameraInstance3DBoxes

    assert isinstance(cam_box, CameraInstance3DBoxes), "input bbox should be CameraInstance3DBoxes!"
    loc = cam_box.gravity_center
    dim = cam_box.dims
    yaw = cam_box.yaw
    feats = cam_box.tensor[:, 7:]
    # rotate along x-axis for np.pi / 2
    # see also here: https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L557  # noqa
    dim[:, [1, 2]] = dim[:, [2, 1]]
    # change local yaw to global yaw for visualization
    # refer to https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L164-L166  # noqa
    yaw += torch.atan2(loc[:, 0], loc[:, 2])
    # convert yaw by (-yaw - np.pi / 2)
    # this is because mono 3D box class such as `NuScenesBox` has different
    # definition of rotation with our `CameraInstance3DBoxes`
    yaw = -yaw - np.pi / 2
    cam_box = torch.cat([loc, dim, yaw[:, None], feats], dim=1)
    cam_box = CameraInstance3DBoxes(cam_box, box_dim=cam_box.shape[-1], origin=(0.5, 0.5, 0.5))

    return cam_box


def get_proj_mat_by_coord_type(img_meta: dict, coord_type: str) -> Tensor:
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta information.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'. Can be case-
            insensitive.

    Returns:
        Tensor: Transformation matrix.
    """
    coord_type = coord_type.upper()
    mapping = {"LIDAR": "lidar2img", "DEPTH": "depth2img", "CAMERA": "cam2img"}
    assert coord_type in mapping.keys()
    return img_meta[mapping[coord_type]]


def yaw2local(yaw: Tensor, loc: Tensor) -> Tensor:
    """Transform global yaw to local yaw (alpha in kitti) in camera
    coordinates, ranges from -pi to pi.

    Args:
        yaw (Tensor): A vector with local yaw of each box in shape (N, ).
        loc (Tensor): Gravity center of each box in shape (N, 3).

    Returns:
        Tensor: Local yaw (alpha in kitti).
    """
    local_yaw = yaw - torch.atan2(loc[:, 0], loc[:, 2])
    larger_idx = (local_yaw > np.pi).nonzero(as_tuple=False)
    small_idx = (local_yaw < -np.pi).nonzero(as_tuple=False)
    if len(larger_idx) != 0:
        local_yaw[larger_idx] -= 2 * np.pi
    if len(small_idx) != 0:
        local_yaw[small_idx] += 2 * np.pi

    return local_yaw


def get_lidar2img(cam2img: Tensor, lidar2cam: Tensor) -> Tensor:
    """Get the projection matrix of lidar2img.

    Args:
        cam2img (torch.Tensor): A 3x3 or 4x4 projection matrix.
        lidar2cam (torch.Tensor): A 3x3 or 4x4 projection matrix.

    Returns:
        Tensor: Transformation matrix with shape 4x4.
    """
    if cam2img.shape == (3, 3):
        temp = cam2img.new_zeros(4, 4)
        temp[:3, :3] = cam2img
        cam2img = temp

    if lidar2cam.shape == (3, 3):
        temp = lidar2cam.new_zeros(4, 4)
        temp[:3, :3] = lidar2cam
        lidar2cam = temp
    return torch.matmul(cam2img, lidar2cam)


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates.

    Coordinates in LiDAR:

    .. code-block:: none

                                 up z    x front (yaw=0)
                                    ^   ^
                                    |  /
                                    | /
        (yaw=0.5*pi) left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2. The yaw is 0 at
    the positive direction of x axis, and increases from the positive direction
    of x to the positive direction of y.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    YAW_AXIS = 2

    @property
    def corners(self) -> Tensor:
        """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
        x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

        .. code-block:: none

                                           up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
            left y <------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)

        Returns:
            Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
            device=dims.device, dtype=dims.dtype
        )

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin (0.5, 0.5, 0)
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=self.YAW_AXIS)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    def rotate(
        self, angle: Union[Tensor, np.ndarray, float], points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray], Tuple[BasePoints, Tensor], None]:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (Tensor or np.ndarray or float): Rotation angle or rotation
                matrix.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        if not isinstance(angle, Tensor):
            angle = self.tensor.new_tensor(angle)

        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, f"invalid rotation angle shape {angle.shape}"

        if angle.numel() == 1:
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3], angle, axis=self.YAW_AXIS, return_mat=True
            )
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        self.tensor[:, 6] += angle

        if self.tensor.shape[1] == 9:
            # rotate velo vector
            self.tensor[:, 7:9] = self.tensor[:, 7:9] @ rot_mat_T[:2, :2]

        if points is not None:
            if isinstance(points, Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(
        self, bev_direction: str = "horizontal", points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tensor, np.ndarray, BasePoints, None]:
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (str): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            Tensor or np.ndarray or :obj:`BasePoints` or None: When ``points``
            is None, the function returns None, otherwise it returns the
            flipped points.
        """
        assert bev_direction in ("horizontal", "vertical")
        if bev_direction == "horizontal":
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]
        elif bev_direction == "vertical":
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi

        if points is not None:
            assert isinstance(points, (Tensor, np.ndarray, BasePoints))
            if isinstance(points, (Tensor, np.ndarray)):
                if bev_direction == "horizontal":
                    points[:, 1] = -points[:, 1]
                elif bev_direction == "vertical":
                    points[:, 0] = -points[:, 0]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    def convert_to(
        self, dst: int, rt_mat: Optional[Union[Tensor, np.ndarray]] = None, correct_yaw: bool = False
    ) -> "BaseInstance3DBoxes":
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Box mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            correct_yaw (bool): Whether to convert the yaw angle to the target
                coordinate. Defaults to False.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type in
            the ``dst`` mode.
        """
        from .box_3d_mode import Box3DMode

        return Box3DMode.convert(box=self, src=Box3DMode.LIDAR, dst=dst, rt_mat=rt_mat, correct_yaw=correct_yaw)

    def enlarged_box(self, extra_width: Union[float, Tensor]) -> "LiDARInstance3DBoxes":
        """Enlarge the length, width and height of boxes.

        Args:
            extra_width (float or Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)
