import numpy as np
import torch
import torchvision
from torch.nn.modules import conv
from torch.nn import functional as F
from torch.nn.modules.utils import _triple
from torch.nn.functional import conv3d


def make_affine_matrix(scale, rotate, translate_x, translate_y, use_out_channels=False):
    # if out_channels is used, the shape of the matrix returned is different

    assert scale.shape == rotate.shape == translate_x.shape == translate_y.shape

    if use_out_channels:
        matrix = torch.zeros((scale.shape[0], scale.shape[1], 2, 3))

        # i = number of filters
        # j = number of transformations
        for i in range(0, scale.shape[0]):
            # first transform is the identity
            matrix[i][0] = torch.eye(3)[:2]
            # TODO: maybe torch.transpose is missing some parameters
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
        matrix[0] = torch.eye(3)[:2]
        for i in range(1, scale.shape[0]):
            matrix[i][0][0] = scale[i] * torch.cos(rotate[i])
            matrix[i][0][1] = -scale[i] * torch.sin(rotate[i])
            matrix[i][0][2] = translate_x[i] * scale[i] * torch.cos(rotate[i]) - translate_y[i] * \
                              scale[i] * torch.sin(rotate[i])
            matrix[i][1][0] = scale[i] * torch.sin(rotate[i])
            matrix[i][1][1] = -scale[i] * torch.cos(rotate[i])
            matrix[i][1][2] = translate_x[i] * scale[i] * torch.sin(rotate[i]) + translate_y[i] * \
                                 scale[i] * torch.cos(rotate[i])


    return matrix


class ConvTTN3d(conv._ConvNd):
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

        super(ConvTTN3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode)

        # kernel_size = (5, 5, 5), type = tuple

        # most general formulation of the affine matrix. if False then imposes scale, rotate and translate restrictions
        most_general = False
        # if use_out_channels = True, all filters in out_channels will have their own affine transformations. if False, 
        # a single set of parameters is used for all filters in out_channels
        use_out_channels = False
        # TODO: implement initialize with affine
        # when transferring from 2D network, copy trained weights to this parameter after initialization
        self.first_weight = torch.nn.init.normal(torch.nn.Parameter(torch.zeros(out_channels, in_channels, 1,
                                                                                kernel_size[1], kernel_size[2])))

        # ------
        # affine parameters
        if use_out_channels:
            if most_general:
                 self.theta = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, kernel_size[0], 2, 3))))
            else:
                # TODO: fix for when use_out_channels==True
                self.scale = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, kernel_size[0]))))
                self.rotate = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, kernel_size[0]))))
                self.translate_x = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, kernel_size[0]))))
                self.translate_y = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, kernel_size[0]))))
        else:
            # initialize strictly positively
            self.scale = torch.abs(torch.nn.init.normal(torch.nn.Parameter(torch.zeros((kernel_size[0])))))
            self.rotate = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((kernel_size[0]))))
            self.translate_x = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((kernel_size[0]))))
            self.translate_y = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((kernel_size[0]))))

        # for now just use the same transformation for all filters in out_channels
        # affine transformation matrix
        self.theta = make_affine_matrix(self.scale, self.rotate, self.translate_x, self.translate_y,
                                        use_out_channels=use_out_channels)
        # TODO: fix RuntimeError: Expected tensor to have size 6 at dimension 0, but got size 5 for argument #2 'batch2' (while checking arguments for bmm)
        self.grid = F.affine_grid(self.theta, torch.Size([kernel_size[0], in_channels, kernel_size[1], kernel_size[2]]))
        # self.grid = F.affine_grid(self.theta, torch.Size([out_channels, kernel_size[0], kernel_size[1], kernel_size[2]]))
        # ------
        self.weight.requires_grad = False

# example of how grid is used: https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522
# assuming transfer learning scenario, transfer happens in python file setup.py
# the 2d kernels are broadcasted and copied to the 3d kernels
    def forward(self, input):
        # print(self.weight.shape)
        # TODO: are shapes correct?
        self.weight = F.grid_sample(self.first_weight, self.grid)

        # TODO: check what happens when weight.requires_grad == False
        # TODO: is this the correct place to set gradient to False?



        y = F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
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


class LeNet5_TTN3d(torch.nn.Module):

    def __init__(self):
        super(LeNet5_TTN3d, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = ConvTTN3d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2)
        # Convolution
        self.conv2 = ConvTTN3d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
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
