import torch.nn as nn
import torch
from groupnorm import GroupNorm3d
from torch import rot90
from typing import Optional, List, Tuple, Union
from torch.nn import functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _reverse_repeat_tuple

# from torch.modules.utils import _triple


def rotations4(polycube, axes):
    shape = polycube.shape
    results = torch.empty(shape[0] * 4, shape[1], shape[2], shape[3], shape[4])
    for i in range(4):
        rot_results = rot90(polycube, i, dims=axes)
        results[i * shape[0] : (i + 1) * shape[0]] = rot_results
    return results


# rotate xyz tensor
def rotations24(polycube):

    # 4 rotations about axis 0
    results = rotations4(polycube, (3, 4))

    # rotate 180 about axis 1, now shape is pointing down in axis 0
    # 4 rotations about axis 0
    results = torch.cat(
        (results, rotations4(rot90(polycube, 2, dims=(2, 4)), (3, 4))), 0
    )

    # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
    # 8 rotations about axis 2
    results = torch.cat((results, rotations4(rot90(polycube, dims=(2, 4)), (2, 3))), 0)
    results = torch.cat(
        (results, rotations4(rot90(polycube, -1, dims=(2, 4)), (2, 3))), 0
    )

    # rotate about axis 2, now shape is pointing in axis 1
    # 8 rotations about axis 1
    results = torch.cat((results, rotations4(rot90(polycube, dims=(2, 3)), (2, 4))), 0)
    results = torch.cat(
        (results, rotations4(rot90(polycube, -1, dims=(2, 3)), (2, 4))), 0
    )
    return results


class groupConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = tuple([kernel_size, kernel_size, kernel_size])
        stride_ = tuple([stride, stride, stride])
        padding_ = (
            padding if isinstance(padding, str) else tuple([padding, padding, padding])
        )
        dilation_ = tuple([dilation, dilation, dilation])
        super(groupConv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )
        if bias:
            raise NotImplementedError("Does not support bias yet")

    def _conv_forward(self, input, weight, bias):
        """
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        """
        group_weight = rotations24(weight)
        print(group_weight.shape)
        return F.conv3d(
            input,
            group_weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)


# polycube = torch.arange(540).view(4,5,3,3,3)
# polycube_=torch.as_tensor(polycube)
results = rotations24(polycube)
print(results.shape)

input = torch.randn(20, 1, 10, 50, 100)

print(len(results))

g_3d = groupConv3d(1, 2, 3, padding="same")
output = g_3d(input)
print(output.shape)
