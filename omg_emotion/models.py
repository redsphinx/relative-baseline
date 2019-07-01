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

    '''
    matrix.shape = (out_channels, 2, 3)
    '''
    matrix = torch.zeros((scale.shape[0], 2, 3))
    for i in range(scale.shape[0]):
        matrix[i][0][0] = scale[i] * torch.cos(rotate[i])
        matrix[i][0][1] = -scale[i] * torch.sin(rotate[i])
        matrix[i][0][2] = translate_x[i] * scale[i] * torch.cos(rotate[i]) - translate_y[i] * \
                          scale[i] * torch.sin(rotate[i])
        matrix[i][1][0] = scale[i] * torch.sin(rotate[i])
        matrix[i][1][1] = -scale[i] * torch.cos(rotate[i])
        matrix[i][1][2] = translate_x[i] * scale[i] * torch.sin(rotate[i]) + translate_y[i] * \
                          scale[i] * torch.cos(rotate[i])

    return matrix


class ConvTTN3d_classic(conv._ConvNd):
    # for now: assume different transformations for each slice on the time axis

    # number of filters = out_channels
    # number of transformations needed = kernel_size[0], where the first transformation is the identity
    #
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        # (self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode):

        if torch.__version__ == '1.0.0':
            super(ConvTTN3d_classic, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _triple(0), groups, bias)
        else:
            # padding_mode = 'zeros'
            super(ConvTTN3d_classic, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _triple(0), groups, bias, padding_mode)

        # kernel_size = (5, 5, 5), type = tuple

        # most general formulation of the affine matrix. if False then imposes scale, rotate and translate restrictions
        most_general = False
        # if use_time_N = True, all time slices will have their own affine transformations. if False,
        # a single set of parameters is used for all time slices
        use_time_N = False
        # TODO: implement initialize with affine
        # when transferring from 2D network, copy trained weights to this parameter after initialization
        # TODO: remove the dimension with '1'
        self.first_weight = torch.nn.init.normal(torch.nn.Parameter(torch.zeros(out_channels, in_channels,  1,
                                                                                kernel_size[1], kernel_size[2])))

        # ------
        # affine parameters
        if use_time_N:
            if most_general:
                self.theta = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, kernel_size[0], 2, 3))))
            else:
                # TODO: fix for when use_time_N==True
                self.scale = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, kernel_size[0]))))
                self.rotate = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, kernel_size[0]))))
                self.translate_x = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, kernel_size[0]))))
                self.translate_y = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, kernel_size[0]))))
        else:
            # initialize strictly positively
            self.scale = torch.abs(torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels)))))
            self.rotate = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels))))
            self.translate_x = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels))))
            self.translate_y = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels))))

        # for now use different transforms for separate filters, but same transform in the time domains
        self.theta = make_affine_matrix(self.scale, self.rotate, self.translate_x, self.translate_y,
                                        use_time_N=use_time_N)
        # the_size = torch.Size([kernel_size[0], out_channels, kernel_size[1], kernel_size[2]])
        the_size = torch.Size([out_channels, kernel_size[0], kernel_size[1], kernel_size[2]])

        self.grid = torch.nn.Parameter(F.affine_grid(self.theta, the_size))
        self.grid.requires_grad = False
        # self.grid = F.affine_grid(self.theta, torch.Size([out_channels, kernel_size[0], kernel_size[1], kernel_size[2]]))
        # ------
        self.weight.requires_grad = False

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

        self.weight[:, :, 0, :, :] = self.first_weight
        for i in range(1, self.weight.shape[2]):
            self.weight[:, :, i, :, :] = F.grid_sample(self.first_weight, self.grid)

        y = F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y


# umut
# class ConvTTN3d(torch.nn.Module):
#     # for now: assume different transformations for each slice on the time axis
#
#     # number of filters = out_channels
#     # number of transformations needed = kernel_size[0], where the first transformation is the identity
#     #
#     # def __init__(self, in_channels, out_channels, kernel_size,
#     #              stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
#         super(ConvTTN3d, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _triple(kernel_size)
#         self.stride = _triple(stride)
#         self.padding = _triple(padding)
#         self.dilation = _triple(dilation)
#         self.groups = groups
#
#         self.bias = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((self.out_channels))))
#
#         self.first_weight = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros(self.out_channels, self.in_channels, 1,
#                                                                                 self.kernel_size[1], self.kernel_size[2])))
#         # self.first_weight_to_concat = self.first_weight.unsqueeze(2)
#         self.scale = torch.abs(torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((self.kernel_size[0]-1, self.out_channels)))))#, self.kernel_size[0]-1)))))
#         self.rotate = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((self.kernel_size[0]-1, self.out_channels))))#, self.kernel_size[0]-1))))
#         self.translate_x = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((self.kernel_size[0]-1, self.out_channels))))#, self.kernel_size[0]-1))))
#         self.translate_y = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((self.kernel_size[0]-1, self.out_channels))))#, self.kernel_size[0]-1))))
#
#         self.theta = torch.zeros((self.kernel_size[0]-1, self.out_channels, 2, 3))
#         self.grid = torch.zeros((self.kernel_size[0]-1, self.out_channels, self.kernel_size[1], self.kernel_size[2], 2))
#
#         # self.theta = [[] for i in range(self.kernel_size[0] - 1)]
#         # self.grid = [[] for i in range(self.kernel_size[0] - 1)]
#
#
#         # self.theta = make_affine_matrix(self.scale, self.rotate, self.translate_x, self.translate_y,
#         #                                 use_time_N=False)
#         # # the_size = torch.Size([kernel_size[0], out_channels, kernel_size[1], kernel_size[2]])
#         # the_size = torch.Size([out_channels, kernel_size[0], kernel_size[1], kernel_size[2]])
#         #
#         # self.grid = torch.nn.Parameter(F.affine_grid(self.theta, the_size))
#
#
#     def update_this(self):
#
#         for i in range(self.kernel_size[0]-1):
#             self.theta[i] = make_affine_matrix(self.scale[i], self.rotate[i], self.translate_x[i], self.translate_y[i],
#                                             use_time_N=True)
#             # the_size = torch.Size([kernel_size[0], out_channels, kernel_size[1], kernel_size[2]])
#             the_size = torch.Size([self.out_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
#
#             # theta.shape = (N x 2 x 3), N = time
#             self.grid[i] = torch.nn.Parameter(F.affine_grid(self.theta[i], the_size))
#
#             # self.grid.cuda(device)
#             # self.theta.cuda(device)
#
#
# # example of how grid is used: https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522
# # assuming transfer learning scenario, transfer happens in python file setup.py
# # the 2d kernels are broadcasted and copied to the 3d kernels
#     def forward(self, input, device):
#
#
#         # my_weight = torch.Tensor(np.array((self.out_channels - 1, self.in_channels, self.kernel_size,
#         #                                    self.kernel_size, self.kernel_size)))
#         my_weight = torch.zeros(
#             (self.out_channels, self.in_channels, self.kernel_size[0]-1, self.kernel_size[1], self.kernel_size[2]))
#
#         self.grid = self.grid.cuda(device) # torch.Size([4, 6, 5, 5, 2])
#         self.theta = self.theta.cuda(device) # torch.Size([4, 6, 2, 3])
#
#         # self.weight[:, :, 0, :, :] = self.first_weight
#
#         # ---
#         # needed to deal with the cudnn error
#         try:
#             _ = F.grid_sample(self.first_weight[:, :, 0], self.grid[0])
#         except RuntimeError:
#             torch.backends.cudnn.deterministic = True
#             _ = F.grid_sample(self.first_weight[:, :, 0], self.grid[0])
#             print('ok cudnn')
#         # ---
#
#         for i in range(my_weight.shape[2]-1):
#             my_weight[:, :, i, :, :] = F.grid_sample(self.first_weight[:, :, 0], self.grid[i])
#             # self.weight[:, :, i, :, :] = F.grid_sample(self.first_weight, self.grid)
#
#         my_weight = my_weight.cuda(device)
#
#         '''
#         kernel_size = (time, h, w)
#         first_weight.shape = (out_channels, in_channels, 1, h, w), 1 = first timeslice
#         my_weight.shape = (out_channels, in_channels, time-1, h, w)
#
#         '''
#         new_weight = torch.cat((self.first_weight, my_weight), 2)
#         # new_weight = np.concatenate(self.first_weight, my_weight, dimension=timedim)
#
#         y = F.conv3d(input, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         # y = F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         return y

# CURRENT
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
#         self.first_weight = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros(out_channels, in_channels, 1,
#                                                                                  kernel_size[1], kernel_size[2])))
#
#
#         self.scale = 1+torch.abs(torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels)))))
#         self.rotate = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels))))
#         self.translate_x = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels))))
#         self.translate_y = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels))))
#
#
#         # Don't init with zeros
#         self.theta = torch.zeros((kernel_size[0] - 1, out_channels, 2, 3))
#         self.grid = torch.zeros((kernel_size[0] - 1, out_channels, kernel_size[1], kernel_size[2], 2))
#
#
#     def update_this(self):
#
#         for i in range(self.kernel_size[0] - 1):
#             self.theta[i] = make_affine_matrix(self.scale[i], self.rotate[i], self.translate_x[i], self.translate_y[i],
#                                                use_time_N=True)
#             the_size = torch.Size([self.out_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
#
#             self.grid[i] = torch.nn.Parameter(F.affine_grid(self.theta[i], the_size))
#
#             # self.grid.cuda(device)
#             # self.theta.cuda(device)
#
#     def forward(self, input, device):
#
#         # self.update_this()
#
#
#         my_weight = torch.zeros(
#             (self.out_channels, self.in_channels, self.kernel_size[0] - 1, self.kernel_size[1], self.kernel_size[2]))
#
#         self.grid = self.grid.cuda(device)  # torch.Size([4, 6, 5, 5, 2])
#         # self.theta = self.theta.cuda(device)  # torch.Size([4, 6, 2, 3])
#
#         # self.weight[:, :, 0, :, :] = self.first_weight
#
#         # ---
#         # needed to deal with the cudnn error
#         try:
#             _ = F.grid_sample(self.first_weight[:, :, 0], self.grid[0])
#         except RuntimeError:
#             torch.backends.cudnn.deterministic = True
#             _ = F.grid_sample(self.first_weight[:, :, 0], self.grid[0])
#             print('ok cudnn')
#             del _
#         # ---
#
#         for i in range(my_weight.shape[2] - 1):
#             my_weight[:, :, i, :, :] = F.grid_sample(self.first_weight[:, :, 0], self.grid[i])
#             # self.weight[:, :, i, :, :] = F.grid_sample(self.first_weight, self.grid)
#
#         my_weight = my_weight.cuda(device)
#
#         '''
#         kernel_size = (time, h, w)
#         first_weight.shape = (out_channels, in_channels, 1, h, w), 1 = first timeslice
#         my_weight.shape = (out_channels, in_channels, time-1, h, w)
#
#         '''
#         new_weight = torch.cat((self.first_weight, my_weight), 2)
#         # new_weight = np.concatenate(self.first_weight, my_weight, dimension=timedim)
#
#         y = F.conv3d(input, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         # y = F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         return y


# simple
class ConvTTN3d(conv._ConvNd):
    '''
    basic version where theta is sampled, so we avoid needing the matrix def
    '''

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super(ConvTTN3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode)

        self.first_weight = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros(out_channels, in_channels, 1,
                                                                                 kernel_size[1], kernel_size[2])))

        self.theta = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels, 2, 3))))

        # for cudnn issue
        self.grid = torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels, kernel_size[1], kernel_size[2], 2)))

        # self.grid = torch.zeros((kernel_size[0] - 1, out_channels, kernel_size[1], kernel_size[2], 2))
        # self.the_size = torch.Size([self.out_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])


    def update_this(self):
        # cudnn error
        try:
            _ = F.affine_grid(self.theta[0],
                              [self.out_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
        except RuntimeError:
            torch.backends.cudnn.deterministic = True
            _ = F.affine_grid(self.theta[0],
                              [self.out_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
            print('ok cudnn')
            del _

        for i in range(self.kernel_size[0] - 1):
            self.grid[i] = F.affine_grid(self.theta[i], [self.out_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])


    def forward(self, input, device):

        # self.update_this()



        my_weight = torch.zeros(
            (self.out_channels, self.in_channels, self.kernel_size[0] - 1, self.kernel_size[1], self.kernel_size[2]))

        # self.grid = self.grid.cuda(device)  # torch.Size([4, 6, 5, 5, 2])
        # self.theta = self.theta.cuda(device)  # torch.Size([4, 6, 2, 3])

        # self.weight[:, :, 0, :, :] = self.first_weight

        # ---
        # needed to deal with the cudnn error
        try:
            _ = F.grid_sample(self.first_weight[:, :, 0], self.grid[0])
        except RuntimeError:
            torch.backends.cudnn.deterministic = True
            _ = F.grid_sample(self.first_weight[:, :, 0], self.grid[0])
            print('ok cudnn')
            del _
        # ---

        new_weight = self.first_weight

        for i in range(my_weight.shape[2]):
            tmp = F.grid_sample(self.first_weight[:, :, 0], self.grid[i])
            new_weight = torch.cat((new_weight, tmp.unsqueeze(2)), 2)
            # my_weight[:, :, i, :, :] = F.grid_sample(self.first_weight[:, :, 0], self.grid[i])

        # my_weight = my_weight.cuda(device)

        # new_weight = torch.cat((self.first_weight, my_weight), 2)

        y = F.conv3d(input, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y


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
        self.conv1 = ConvTTN3d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv2 = ConvTTN3d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2)
        self.fc1 = torch.nn.Linear(16 * 5 * 5 * 5,
                                   120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x, device):
        if torch.isnan(x).sum() > 0:
            print('NAN found 1')

        x = self.conv1(x, device)

        if torch.isnan(x).sum() > 0:
            print('NAN found 2')

        x = torch.nn.functional.relu(x)

        if torch.isnan(x).sum() > 0:
            print('NAN found 3')

        x = self.max_pool_1(x)

        if torch.isnan(x).sum() > 0:
            print('NAN found 4')

        x = self.conv2(x, device)

        if torch.isnan(x).sum() > 0:
            print('NAN found 5')

        x = torch.nn.functional.relu(x)

        if torch.isnan(x).sum() > 0:
            print('NAN found 6')

        x = self.max_pool_2(x)

        if torch.isnan(x).sum() > 0:
            print('NAN found 7 ')

        x = x.view(-1, 16 * 5 * 5 * 5)
        x = self.fc1(x)

        if torch.isnan(x).sum() > 0:
            print('NAN found 8')

        x = torch.nn.functional.relu(x)

        if torch.isnan(x).sum() > 0:
            print('NAN found 9')

        x = self.fc2(x)

        if torch.isnan(x).sum() > 0:
            print('NAN found 10')

        x = torch.nn.functional.relu(x)

        if torch.isnan(x).sum() > 0:
            print('NAN found 11')

        x = self.fc3(x)

        if torch.isnan(x).sum() > 0:
            print('NAN found 12')

        return x


