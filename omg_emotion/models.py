import numpy as np
import torch
import torchvision
from torch.nn.modules import conv
from torch.nn import functional as F
from torch.nn.modules.utils import _triple
from torch.nn.functional import conv3d



def make_affine_matrix(scale, rotate, translate_x, translate_y, use_time_N=False):
    # if out_channels is used, the shape of the matrix returned is different

    assert scale.shape == rotate.shape == translate_x.shape == translate_y.shape

    if use_time_N:
        matrix = torch.zeros((scale.shape[0], scale.shape[1], 2, 3))

        # i = number of filters
        # j = number of transformations
        for i in range(0, scale.shape[0]):
            # first transform is the identity
            matrix[i][0] = torch.eye(3)[:2]
            # https://en.wikipedia.org/wiki/Transformation_matrix
            for j in range(1, scale.shape[1]):
                matrix[i][j][0][0] = scale[i][j] * torch.cos(rotate[i][j])
                matrix[i][j][0][1] = -scale[i][j] * torch.sin(rotate[i][j])
                matrix[i][j][0][2] = translate_x[i][j] * scale[i][j] * torch.cos(rotate[i][j]) - translate_y[i][j] * \
                                     scale[i][j] * torch.sin(rotate[i][j])
                matrix[i][j][1][0] = scale[i][j] * torch.sin(rotate[i][j])
                matrix[i][j][1][1] = -scale[i][j] * torch.cos(rotate[i][j])
                matrix[i][j][1][2] = translate_x[i][j] * scale[i][j] * torch.sin(rotate[i][j]) + translate_y[i][j] * \
                                     scale[i][j] * torch.cos(rotate[i][j])
    else:
        matrix = torch.zeros((scale.shape[0], 2, 3))
        for i in range(0, scale.shape[0]):
            matrix[i][0][0] = scale[i] * torch.cos(rotate[i])
            matrix[i][0][1] = -scale[i] * torch.sin(rotate[i])
            matrix[i][0][2] = translate_x[i] * scale[i] * torch.cos(rotate[i]) - translate_y[i] * \
                              scale[i] * torch.sin(rotate[i])
            matrix[i][1][0] = scale[i] * torch.sin(rotate[i])
            matrix[i][1][1] = -scale[i] * torch.cos(rotate[i])
            matrix[i][1][2] = translate_x[i] * scale[i] * torch.sin(rotate[i]) + translate_y[i] * \
                                 scale[i] * torch.cos(rotate[i])


    return matrix

# original
class ConvTTN3d(torch.nn.Module):
    # for now: assume different transformations for each slice on the time axis

    # number of filters = out_channels
    # number of transformations needed = kernel_size[0], where the first transformation is the identity
    #
    # def __init__(self, in_channels, out_channels, kernel_size,
    #              stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(ConvTTN3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups

        self.bias = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels))))

        # (self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode):

        # if torch.__version__ == '1.0.0':
        #     super(ConvTTN3d, self).__init__(
        #         in_channels, out_channels, kernel_size, stride, padding, dilation,
        #         False, _triple(0), groups, bias)
        # else:
        #     # padding_mode = 'zeros'
        #     super(ConvTTN3d, self).__init__(
        #         in_channels, out_channels, kernel_size, stride, padding, dilation,
        #         False, _triple(0), groups, bias, padding_mode)


        # kernel_size = (5, 5, 5), type = tuple

        self.first_weight = torch.nn.init.normal(torch.nn.Parameter(torch.zeros(out_channels, in_channels,
                                                                                self.kernel_size[1], self.kernel_size[2])))

        # TODO: should be list, each list contains self.kernel_size[0] - 1 parameters (time domain)

        self.scale = torch.abs(torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, self.kernel_size[0]-1)))))
        self.rotate = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, self.kernel_size[0]-1))))
        self.translate_x = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, self.kernel_size[0]-1))))
        self.translate_y = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, self.kernel_size[0]-1))))

        # for now use different transforms for separate filters, but same transform in the time domains

        self.theta = [[] for i in range(self.kernel_size[0] - 1)]
        self.grid = [[] for i in range(self.kernel_size[0] - 1)]


        # self.theta = make_affine_matrix(self.scale, self.rotate, self.translate_x, self.translate_y,
        #                                 use_time_N=False)
        # # the_size = torch.Size([kernel_size[0], out_channels, kernel_size[1], kernel_size[2]])
        # the_size = torch.Size([out_channels, kernel_size[0], kernel_size[1], kernel_size[2]])
        #
        # self.grid = torch.nn.Parameter(F.affine_grid(self.theta, the_size))


    def update(self):

        # TODO: fix make affine matrix
        for i in range(self.kernel_size[0]-1):
            self.theta[i] = make_affine_matrix(self.scale[i], self.rotate[i], self.translate_x[i], self.translate_y[i],
                                            use_time_N=True)
            # the_size = torch.Size([kernel_size[0], out_channels, kernel_size[1], kernel_size[2]])
            the_size = torch.Size([self.out_channels, self.kernel_size[0],self.kernel_size[1], self.kernel_size[2]])

            self.grid[i] = torch.nn.Parameter(F.affine_grid(self.theta[i], the_size))


# example of how grid is used: https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522
# assuming transfer learning scenario, transfer happens in python file setup.py
# the 2d kernels are broadcasted and copied to the 3d kernels
    def forward(self, input):
        # ---
        # needed to deal with the cudnn error
        try:
            _ = F.grid_sample(self.first_weight, self.grid)
        except RuntimeError:
            torch.backends.cudnn.deterministic = True
            _ = F.grid_sample(self.first_weight, self.grid)
            print('ok cudnn')
        # ---

        my_weight = torch.Tensor(np.array((self.out_channels - 1, self.in_channels, self.kernel_size,
                                           self.kernel_size, self.kernel_size)))


        # self.weight[:, :, 0, :, :] = self.first_weight

        for i in range(my_weight.shape[2]-1):
            my_weight[:, :, i, :, :] = F.grid_sample(self.first_weight, self.grid[i])
            # self.weight[:, :, i, :, :] = F.grid_sample(self.first_weight, self.grid)

        # TODO
        # my_weight.cuda(device)

        new_weight = torch.cat((self.first_weight, my_weight), 2)
        # new_weight = np.concatenate(self.first_weight, my_weight, dimension=timedim)

        y = F.conv3d(input, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # y = F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y

#
# class ConvTTN3d(torch.nn.Module):
#     # for now: assume different transformations for each slice on the time axis
#
#     # number of filters = out_channels
#     # number of transformations needed = kernel_size[0], where the first transformation is the identity
#     #
#
#     def __init__(self, out_channels, in_channels, kernel_size):
#
#         super(ConvTTN3d, self).__init__()
#
#         # kernel_size = (5, 5, 5), type = tuple
#
#         self.first_weight = torch.nn.init.normal(torch.nn.Parameter(torch.zeros(out_channels, in_channels, # 1,
#                                                                                 kernel_size, kernel_size)))
#
#         # ------
#         # affine parameters
#         # initialize strictly positively
#         self.scale = torch.abs(torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels)))))
#         self.rotate = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels))))
#         self.translate_x = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels))))
#         self.translate_y = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels))))
#
#         # for now use different transforms for separate filters, but same transform in the time domains
#         self.theta = make_affine_matrix(self.scale, self.rotate, self.translate_x, self.translate_y,
#                                         use_time_N=False)
#         # the_size = torch.Size([kernel_size[0], out_channels, kernel_size[1], kernel_size[2]])
#         the_size = torch.Size([out_channels, kernel_size, kernel_size, kernel_size])
#
#         self.grid = torch.nn.Parameter(F.affine_grid(self.theta, the_size))
#
#         self.weight = torch.nn.Parameter(torch.zeros((out_channels, in_channels,
#                                                       kernel_size, kernel_size, kernel_size)))
#
#         self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
#
#
# # example of how grid is used: https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522
# # assuming transfer learning scenario, transfer happens in python file setup.py
# # the 2d kernels are broadcasted and copied to the 3d kernels
#     def forward(self, input, stride, padding, dilation, groups):
#         # ---
#         # needed to deal with the cudnn error
#         try:
#             _ = F.grid_sample(self.first_weight, self.grid)
#         except RuntimeError:
#             torch.backends.cudnn.deterministic = True
#             _ = F.grid_sample(self.first_weight, self.grid)
#         # ---
#
#         self.weight[:, :, 0, :, :] = self.first_weight
#         for i in range(1, self.weight.shape[2]):
#             self.weight[:, :, i, :, :] = F.grid_sample(self.first_weight, self.grid)
#
#         y = F.conv3d(input, self.weight, self.bias, stride, padding, dilation, groups)
#         return y



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
        self.fc1 = torch.nn.Linear(16 * 5 * 5 * 5,
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
        x = x.view(-1, 16 * 5 * 5 * 5)
        # FC-1, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)

        return x

# original
# class LeNet5_TTN3d(torch.nn.Module):
# 
#     def __init__(self):
#         super(LeNet5_TTN3d, self).__init__()
#         self.conv1 = ConvTTN3d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
#         self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2)
#         self.conv2 = ConvTTN3d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
#         self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2)
#         self.fc1 = torch.nn.Linear(16 * 5 * 5 * 5,
#                                    120)
#         self.fc2 = torch.nn.Linear(120, 84)
#         self.fc3 = torch.nn.Linear(84, 10)
# 
#     def forward(self, x):
#         # convolve, then perform ReLU non-linearity
#         x = torch.nn.functional.relu(self.conv1(x))
#         # max-pooling with 2x2 grid
#         x = self.max_pool_1(x)
#         # convolve, then perform ReLU non-linearity
#         x = torch.nn.functional.relu(self.conv2(x))
#         # max-pooling with 2x2 grid
#         x = self.max_pool_2(x)
#         # first flatten 'max_pool_2_out' to contain 16*5*5 columns
#         # read through https://stackoverflow.com/a/42482819/7551231
#         x = x.view(-1, 16 * 5 * 5 * 5)
#         # FC-1, then perform ReLU non-linearity
#         x = torch.nn.functional.relu(self.fc1(x))
#         # FC-2, then perform ReLU non-linearity
#         x = torch.nn.functional.relu(self.fc2(x))
#         # FC-3
#         x = self.fc3(x)
# 
#         return x

class LeNet5_TTN3d(torch.nn.Module):

    def __init__(self):
        super(LeNet5_TTN3d, self).__init__()
        self.conv1 = ConvTTN3d(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv2 = ConvTTN3d(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2)
        self.fc1 = torch.nn.Linear(16 * 5 * 5 * 5,
                                   120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16 * 5 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


