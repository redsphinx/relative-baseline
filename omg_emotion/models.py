import numpy as np
import torch
import torchvision

from torch.nn.modules import conv
from torch.nn import functional as F
from torch.nn.modules.utils import _triple

from torch.nn.init import xavier_normal

from torch.nn.functional import conv3d
from torch.nn.functional import affine_grid, grid_sample


class LeNet5_2d(torch.nn.Module):

    def __init__(self):
        super(LeNet5_2d, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16 * 5 * 5,
                                   120)  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)  # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, 10)  # convert matrix with 84 features to a matrix of 10 features (columns)

    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv1(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_1(x)
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv2(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        x = x.view(-1, 16 * 5 * 5)
        # FC-1, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)

        return x


class LeNet5_3d(torch.nn.Module):

    def __init__(self):
        super(LeNet5_3d, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv3d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16 * 5 * 5,
                                   120)  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)  # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, 10)  # convert matrix with 84 features to a matrix of 10 features (columns)

    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv1(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_1(x)
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv2(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        x = x.view(-1, 16 * 5 * 5)
        # FC-1, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)

        return x


# TODO: fix the matrix/vector multiplications
# def make_affine_matrix(out_channels, scale, rotate, translate_x, translate_y, to_tensor=True):
#     """
#     :param scale: list of scale ints
#     :param rotate: list of rotation ints
#     :param translate_x: list of translation x ints
#     :param translate_y: list of translation y ints
#     :param to_tensor: boolean, defualt to True
#     :return: if to_tensor, tensor, else numpy matrix
#     """
#
#     matrix = np.zeros(shape=(out_channels, 2, 3), dtype=np.float)
#
#     # https://en.wikipedia.org/wiki/Transformation_matrix
#     matrix[:][0][0] = scale * np.cos(rotate)
#     matrix[:][1][0] = -scale * np.sin(rotate)
#     matrix[:][2][0] = translate_x * scale * np.cos(rotate) - translate_y * scale * np.sin(rotate) + np.transpose(translate_x) # t'_x
#     matrix[:][0][1] = scale * np.sin(rotate)
#     matrix[:][1][1] = -scale * np.cos(rotate)
#     matrix[:][2][1] = translate_x * scale * np.sin(rotate) + translate_y * scale * np.cos(rotate) + np.transpose(translate_y) # t'_y
#
#     if to_tensor:
#         matrix = torch.from_numpy(matrix)
#
#     return matrix

def make_affine_matrix(scale, rotate, translate_x, translate_y):
    matrix = []
    return matrix



class ConvTTN3d(conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super(ConvTTN3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode)

        # ------
        # affine parameters
        self.scale = torch.nn.init.normal(torch.nn.Parameter(torch.zeros(1)))
        self.rotate = torch.nn.init.normal(torch.nn.Parameter(torch.zeros(1)))
        self.translate_x = torch.nn.init.normal(torch.nn.Parameter(torch.zeros(1)))
        self.translate_y = torch.nn.init.normal(torch.nn.Parameter(torch.zeros(1)))
        # affine transformation matrix
        self.theta = torchvision.transforms.fun


        # self.theta = make_affine_matrix(self.scale, self.rotate, self.translate_x, self.translate_y)


        # ------



    def forward(self, input, kernels2d):

        # ------



        # ------



        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
                                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv3d(F.pad(input, expanded_padding, mode='circular'),
                            self.weight, self.bias, self.stride, _triple(0),
                            self.dilation, self.groups)



        return F.conv3d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



# assume each filter learns a different affine transform
# assume transfer learning from 2D to 3D kernel
# TTN = temporal transformer network
# out_channels = number of filters
# class ConvTTN3d(conv._ConvNd):
#
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
#         kernel_size = _triple(kernel_size)
#         stride = _triple(stride)
#         padding = _triple(padding)
#         dilation = _triple(dilation)
#
#         super(ConvTTN3d, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             False, _triple(0), groups, bias, padding_mode)
#
#         # ----
#         # implement for scale, translate, rotate
#         # ----
#         self.time_grids = []
#         # TODO: what if kernel size is int?
#         for time_point in range(kernel_size[0] - 1):  # time dimension minus 1, keep first frame fixed
#             scale = torch.nn.Parameter(np.random.random(in_channels))
#             rotate = torch.nn.Parameter(np.random.random(in_channels))  # counter clockwise
#             translate_x = torch.nn.Parameter(np.random.random(in_channels))
#             translate_y = torch.nn.Parameter(np.random.random(in_channels))
#
#             self.theta = make_affine_matrix(out_channels, scale, rotate, translate_x, translate_y)
#             # TODO: figure out size: the target output image size (N×C×H×W). Example: torch.Size((32, 3, 24, 24))
#             # TODO: how to get height and width of previous kernel
#             self.grid = affine_grid(theta=self.theta, size=(in_channels, out_channels,))
#             self.time_grids.append(self.grid)
#
#
#
#     def forward(self, input, kernels2d):
#
#         # ----
#         self.weight = grid_sample(kernels2d, grid=self.time_grids)
#         h = F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#
#         return h
#         # ----
#
#
#
#         # if self.padding_mode == 'circular':
#         #     expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
#         #                         (self.padding[1] + 1) // 2, self.padding[1] // 2,
#         #                         (self.padding[0] + 1) // 2, self.padding[0] // 2)
#         #     return F.conv3d(F.pad(input, expanded_padding, mode='circular'),
#         #                     self.weight, self.bias, self.stride, _triple(0),
#         #                     self.dilation, self.groups)
#         #
#         #
#         #
#         # return F.conv3d(input, self.weight, self.bias, self.stride,
#         #                 self.padding, self.dilation, self.groups)
#
#
#
#
#
#
