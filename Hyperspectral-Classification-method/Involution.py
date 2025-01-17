from typing import Union, Tuple, Optional
# from einops import rearrange
import torch.nn.functional as F
import torch
import torch.nn as nn


class Involution2d(nn.Module):
    """
    This class implements the 2d involution proposed in:
    https://arxiv.org/pdf/2103.06255.pdf
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 patch_size: int,
                 sigma_mapping: Optional[nn.Module] = None,
                 kernel_size: Union[int, Tuple[int, int]] = (7, 7),  # 或类型，整数或者元组都可
                 stride: Union[int, Tuple[int, int]] = (1, 1),
                 groups: int = 12,  # default 1
                 reduce_ratio: int = 4,  # default 1
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[int, Tuple[int, int]] = (3, 3),
                 bias: bool = False,
                 **kwargs) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param sigma_mapping: (nn.Module) Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized
        :param kernel_size: (Union[int, Tuple[int, int]]) Kernel size to be used
        :param stride: (Union[int, Tuple[int, int]]) Stride factor to be utilized
        :param groups: (int) Number of groups to be employed
        :param reduce_ratio: (int) Reduce ration of involution channels
        :param dilation: (Union[int, Tuple[int, int]]) Dilation in unfold to be employed
        :param padding: (Union[int, Tuple[int, int]]) Padding to be used in unfold operation
        :param bias: (bool) If true bias is utilized in each convolution layer
        :param **kwargs: Unused additional key word arguments
        """
        # Call super constructor
        super(Involution2d, self).__init__()
        # Check parameters
        # ******************先对参数check*******************************
        assert isinstance(in_channels, int) and in_channels > 0, "in channels must be a positive integer."
        assert in_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(out_channels, int) and out_channels > 0, "out channels must be a positive integer."
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(patch_size, int), "patch_size must be an integer"
        assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, \
            "Sigma mapping must be an nn.Module or None to utilize the default mapping (BN + ReLU)."
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), \
            "kernel size must be an int or a tuple of ints."
        assert isinstance(stride, int) or isinstance(stride, tuple), \
            "stride must be an int or a tuple of ints."
        assert isinstance(groups, int), "groups must be a positive integer."
        assert isinstance(reduce_ratio, int) and reduce_ratio > 0, "reduce ratio must be a positive integer."
        assert isinstance(dilation, int) or isinstance(dilation, tuple), \
            "dilation must be an int or a tuple of ints."
        assert isinstance(padding, int) or isinstance(padding, tuple), \
            "padding must be an int or a tuple of ints."
        assert isinstance(bias, bool), "bias must be a bool"
        # assert 断言函数，条件不满足直接报错
        # ****************************************************************
        # Save parameters
        self.in_channels = in_channels  # 204
        self.out_channels = out_channels  # 192
        self.patch_size = patch_size  # 11
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.bias = bias
        # Init modules
        self.sigma_mapping = sigma_mapping if sigma_mapping is not None else nn.Sequential(
            nn.BatchNorm2d(num_features=self.out_channels // self.reduce_ratio, momentum=0.3), nn.ReLU())
        self.initial_mapping = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                         bias=bias) if self.in_channels != self.out_channels else nn.Identity()
        self.o_mapping = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)  # 1,1
        self.reduce_mapping = nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.out_channels // self.reduce_ratio, kernel_size=(1, 1),
                                        stride=(1, 1), padding=(0, 0), bias=bias)  # 通道数做一变换
        self.span_mapping = nn.Conv2d(in_channels=self.out_channels // self.reduce_ratio,
                                      out_channels=self.kernel_size[0] * self.kernel_size[1] * self.groups,
                                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=dilation, padding=padding, stride=stride)
        self.flattened = self.flattened()
        self.Linear1 = nn.Sequential(
            nn.Linear(self.flattened, 512),  # 2048
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.Linear2 = nn.Sequential(
            nn.Linear(512, 128),  # 1024
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 17)  # 手动设置为17
        )

    def __repr__(self) -> str:
        """
        Method returns information about the module
        :return: (str) Info string
        """
        return ("{}({}, {},{}, kernel_size=({}, {}), stride=({}, {}), padding=({}, {}), "
                "groups={}, reduce_ratio={}, dilation=({}, {}), bias={}, sigma_mapping={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.patch_size,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.groups,
            self.reduce_mapping,
            self.dilation[0],
            self.dilation[1],
            self.bias,
            str(self.sigma_mapping)
        ))

    def flattened(self):  # 这个函数只是为全连接的时候用
        with torch.no_grad():
            x = torch.zeros((1, self.in_channels, self.patch_size, self.patch_size,))
            batch_size, _, in_height, in_width = x.shape
            out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) \
                         // self.stride[0] + 1
            out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) \
                        // self.stride[1] + 1
            # Unfold and reshape input tensor
            input_unfolded = self.unfold(self.initial_mapping(x))
            # 通道做了压缩后，再进行unfold操作，unfold操作后还需要进行reshape（此处可以调用einops操作）。
            input_unfolded = input_unfolded.view(batch_size, self.groups, self.out_channels // self.groups,
                                                 self.kernel_size[0] * self.kernel_size[1],
                                                 out_height, out_width)
            # Generate kernel
            kernel = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(x))))
            print(kernel.shape)
            # 此处也用到了初始的通道，不一定要整除G吧？？？？？？？？？？？？？
            kernel = kernel.view(batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1],
                                 kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)
            # Apply kernel to produce output
            output = (kernel * input_unfolded).sum(dim=3)
            # Reshape output
            output = output.view(batch_size, -1, output.shape[-2], output.shape[-1])
            t, w, l, b = output.size()  # 因为这里是1,所以相乘之后得到的数即是
            return t * w * l * b

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, height, width] (w/ same padding)
        """
        # Check input dimension of input tensor
        input = torch.squeeze(input)  # 降维
        # print(input.shape)  # 注释
        assert input.ndimension() == 4, \
            "Input tensor to involution must be 4d but {}d tensor is given".format(input.ndimension())
        # Save input shape and compute output shapes
        batch_size, _, in_height, in_width = input.shape
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) \
                     // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) \
                    // self.stride[1] + 1
        # Unfold and reshape input tensor
        input_unfolded = self.unfold(self.initial_mapping(input))  # 先对通道做一个降维，再unfold,k*k*c,patch*patch
        input_unfolded = input_unfolded.view(batch_size, self.groups, self.out_channels // self.groups,
                                             self.kernel_size[0] * self.kernel_size[1],
                                             out_height, out_width)  # 此处就是c/g
        # Generate kernel
        kernel = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(input))))
        kernel = kernel.view(batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1],
                             kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)
        # Apply kernel to produce output
        output = (kernel * input_unfolded).sum(dim=3)
        # Reshape output
        output = output.view(batch_size, -1, output.shape[-2], output.shape[-1])

        output = torch.flatten(output, start_dim=1)  # 将后面的几个维度直接合并
        # output = rearrange(output, 'b c h w -> b (c h w)')  # 也可以用另外一个包，einop->srearrange
        output = self.Linear1(output)  # 后面三层是我直接加的
        output = self.Linear2(output)
        return output


class Involution3d(nn.Module):
    """
    This class implements the 3d involution.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 patch_size: int,
                 sigma_mapping: Optional[nn.Module] = None,
                 kernel_size: Union[int, Tuple[int, int, int]] = (7, 7, 7),
                 stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
                 groups: int = 12,  # default：1
                 reduce_ratio: int = 4,  # default ：1
                 dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1),
                 padding: Union[int, Tuple[int, int, int]] = (3, 3, 3),
                 bias: bool = False,
                 **kwargs) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param sigma_mapping: (nn.Module) Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized
        :param kernel_size: (Union[int, Tuple[int, int, int]]) Kernel size to be used
        :param stride: (Union[int, Tuple[int, int, int]]) Stride factor to be utilized
        :param groups: (int) Number of groups to be employed
        :param reduce_ratio: (int) Reduce ration of involution channels
        :param dilation: (Union[int, Tuple[int, int, int]]) Dilation in unfold to be employed
        :param padding: (Union[int, Tuple[int, int, int]]) Padding to be used in unfold operation
        :param bias: (bool) If true bias is utilized in each convolution layer
        :param **kwargs: Unused additional key word arguments
        """
        # Call super constructor
        super(Involution3d, self).__init__()
        # Check parameters
        assert isinstance(in_channels, int) and in_channels > 0, "in channels must be a positive integer."
        assert in_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(out_channels, int) and out_channels > 0, "out channels must be a positive integer."
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(patch_size, int), "patch_size must be an integer"
        assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, \
            "Sigma mapping must be an nn.Module or None to utilize the default mapping (BN + ReLU)."
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), \
            "kernel size must be an int or a tuple of ints."
        assert isinstance(stride, int) or isinstance(stride, tuple), \
            "stride must be an int or a tuple of ints."
        assert isinstance(groups, int), "groups must be a positive integer."
        assert isinstance(reduce_ratio, int) and reduce_ratio > 0, "reduce ratio must be a positive integer."
        assert isinstance(dilation, int) or isinstance(dilation, tuple), \
            "dilation must be an int or a tuple of ints."
        assert isinstance(padding, int) or isinstance(padding, tuple), \
            "padding must be an int or a tuple of ints."
        assert isinstance(bias, bool), "bias must be a bool"
        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.bias = bias
        # Init modules
        self.sigma_mapping = sigma_mapping if sigma_mapping is not None else nn.Sequential(
            nn.BatchNorm3d(num_features=self.out_channels // self.reduce_ratio, momentum=0.3), nn.ReLU())
        self.initial_mapping = nn.Conv3d(
            in_channels=self.in_channels, out_channels=self.out_channels,
            kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
            bias=bias) if self.in_channels != self.out_channels else nn.Identity()
        self.o_mapping = nn.AvgPool3d(kernel_size=self.stride, stride=self.stride)
        self.reduce_mapping = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels // self.reduce_ratio, kernel_size=(1, 1, 1),
            stride=(1, 1, 1), padding=(0, 0, 0), bias=bias)
        self.span_mapping = nn.Conv3d(
            in_channels=self.out_channels // self.reduce_ratio,
            out_channels=self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.groups,
            kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=bias)
        self.pad = nn.ConstantPad3d(padding=(self.padding[0], self.padding[0],
                                             self.padding[1], self.padding[1],
                                             self.padding[2], self.padding[2]), value=0.)
        self.flattened = self.flattened()
        self.Linear1 = nn.Sequential(
            nn.Linear(self.flattened, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        self.Linear2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 17)  # 手动设置为17
        )

    def __repr__(self) -> str:
        """
        Method returns information about the module
        :return: (str) Info string
        """
        return ("{}({}, {}, kernel_size=({}, {}, {}), stride=({}, {}, {}), padding=({}, {}, {}), "
                "groups={}, reduce_ratio={}, dilation=({}, {}, {}), bias={}, sigma_mapping={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.groups,
            self.reduce_mapping,
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self.bias,
            str(self.sigma_mapping)
        ))

    def flattened(self):  # 这个函数只是为全连接的时候用
        with torch.no_grad():
            input = (torch.zeros((1, self.in_channels, self.patch_size, self.patch_size,))).unsqueeze(dim=2)
            # 增加一个维度[batch size, in channels, depth, height, width]
            batch_size, _, in_depth, in_height, in_width = input.shape
            out_depth = (in_depth + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) \
                        // self.stride[0] + 1
            out_height = (in_height + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) \
                         // self.stride[1] + 1
            out_width = (in_width + 2 * self.padding[2] - self.dilation[2] * (self.kernel_size[2] - 1) - 1) \
                        // self.stride[2] + 1
            # Unfold and reshape input tensor
            input_initial = self.initial_mapping(input)
            input_unfolded = self.pad(input_initial) \
                .unfold(dimension=2, size=self.kernel_size[0], step=self.stride[0]) \
                .unfold(dimension=3, size=self.kernel_size[1], step=self.stride[1]) \
                .unfold(dimension=4, size=self.kernel_size[2], step=self.stride[2])
            input_unfolded = input_unfolded.reshape(batch_size, self.groups, self.out_channels // self.groups,
                                                    self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], -1)
            input_unfolded = input_unfolded.reshape(tuple(input_unfolded.shape[:-1])
                                                    + (out_depth, out_height, out_width))
            # Generate kernel
            kernel = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(input))))
            kernel = kernel.view(
                batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2],
                kernel.shape[-3], kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)
            # Apply kernel to produce output
            output = (kernel * input_unfolded).sum(dim=3)
            # Reshape output
            output = output.view(batch_size, -1, output.shape[-3], output.shape[-2], output.shape[-1])
            _, t, w, l, b = output.size()  # 因为这里是1,所以相乘之后得到的数即是
            return t * w * l * b

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, depth, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, depth, height, width] (w/ same padding)
        """
        # Check input dimension of input tensor
        assert input.ndimension() == 5, \
            "Input tensor to involution must be 5d but {}d tensor is given".format(input.ndimension())
        # Save input shapes and compute output shapes
        # print(input.dim())  # 位置的维度要统一
        input = torch.transpose(input, 1, 2)  # 在此处对张量的维度做了一个统一
        # print(input.shape)
        batch_size, _, in_depth, in_height, in_width = input.shape
        out_depth = (in_depth + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) \
                    // self.stride[0] + 1
        out_height = (in_height + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) \
                     // self.stride[1] + 1
        out_width = (in_width + 2 * self.padding[2] - self.dilation[2] * (self.kernel_size[2] - 1) - 1) \
                    // self.stride[2] + 1
        # Unfold and reshape input tensor
        input_initial = self.initial_mapping(input)
        input_unfolded = self.pad(input_initial) \
            .unfold(dimension=2, size=self.kernel_size[0], step=self.stride[0]) \
            .unfold(dimension=3, size=self.kernel_size[1], step=self.stride[1]) \
            .unfold(dimension=4, size=self.kernel_size[2], step=self.stride[2])
        input_unfolded = input_unfolded.reshape(batch_size, self.groups, self.out_channels // self.groups,
                                                self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], -1)
        input_unfolded = input_unfolded.reshape(tuple(input_unfolded.shape[:-1])
                                                + (out_depth, out_height, out_width))
        # Generate kernel
        kernel = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(input))))
        kernel = kernel.view(
            batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2],
            kernel.shape[-3], kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)
        # Apply kernel to produce output
        output = (kernel * input_unfolded).sum(dim=3)
        # Reshape output
        output = output.view(batch_size, -1, output.shape[-3], output.shape[-2], output.shape[-1])
        output = torch.flatten(output, start_dim=1)  # 将后面的几个维度直接合并
        output = self.Linear1(output)
        output = self.Linear2(output)
        return output