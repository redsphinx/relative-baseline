import numpy as np

from relative_baseline.omg_emotion.convttn3d import ConvTTN3d

import torch
import torch.nn as nn
from torch.nn.modules import conv
from torch.nn import functional as F
from torch.nn.modules.utils import _triple
from torch.nn.functional import conv3d


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

    def __init__(self, project_variable):
        self.return_ind = False
        if project_variable.return_ind:
            self.return_ind = True

        super(LeNet5_3d, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=project_variable.num_out_channels[0], kernel_size=project_variable.k_shape, stride=1, padding=2, bias=True)
        # conv is initialized with uniformly sampled weights
        if project_variable.k0_init == 'normal':
            self.conv1.weight = torch.nn.init.normal_(self.conv1.weight)
            self.conv1.bias = torch.nn.init.normal_(self.conv1.bias)
        elif project_variable.k0_init == 'ones':
            self.conv1.weight = torch.nn.init.constant_(self.conv1.weight, 1)
            self.conv1.bias = torch.nn.init.constant_(self.conv1.bias, 1)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2, return_indices=self.return_ind)
        if project_variable.do_batchnorm[0]:
            self.bn1 = torch.nn.BatchNorm3d(project_variable.num_out_channels[0])
        
        # Convolution
        self.conv2 = torch.nn.Conv3d(in_channels=project_variable.num_out_channels[0], out_channels=project_variable.num_out_channels[1], kernel_size=project_variable.k_shape, stride=1, padding=0, bias=True)
        if project_variable.k0_init == 'normal':
            self.conv2.weight = torch.nn.init.normal_(self.conv2.weight)
            self.conv2.bias = torch.nn.init.normal_(self.conv2.bias)
        elif project_variable.k0_init == 'ones':
            self.conv2.weight = torch.nn.init.constant_(self.conv2.weight, 1)
            self.conv2.bias = torch.nn.init.constant_(self.conv2.bias, 1)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2, return_indices=self.return_ind)
        
        if project_variable.do_batchnorm[1]:
            self.bn2 = torch.nn.BatchNorm3d(project_variable.num_out_channels[1])

        # Fully connected layer
        if project_variable.k_shape == (5, 5, 5):
            _fc_in = [5, 5, 5]
        elif project_variable.k_shape == (3, 4, 4):
            _fc_in = [7, 5, 5]
        elif project_variable.k_shape == (5, 6, 6):
            _fc_in = [5, 4, 4]
        elif project_variable.k_shape == (4, 6, 6):
            _fc_in = (6, 4, 4)
        else:
            print('ERROR: k_shape %s not supported' % str(project_variable.k_shape))
            _fc_in = [1, 1, 1]

        if project_variable.dataset == 'kth_actions':
            _fc_in = [73, 28, 38]
        elif project_variable.dataset == 'omg_emotions':
            # TODO
            pass


        self.fc1 = torch.nn.Linear(project_variable.num_out_channels[1] * _fc_in[0] * _fc_in[1] * _fc_in[2],
                                   120)  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)  # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, project_variable.label_size)  # convert matrix with 84 features to a matrix of 10 features (columns)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn1(x)
        except AttributeError:
            pass
        if self.return_ind:
            x, ind1 = self.max_pool_1(x)
        else:
            x = self.max_pool_1(x)


        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn2(x)
        except AttributeError:
            pass
        if self.return_ind:
            x, ind2 = self.max_pool_2(x)
        else:
            x = self.max_pool_2(x)


        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# works with data with 1 color channel
class LeNet5_TTN3d(torch.nn.Module):
# transformation_groups, k0_groups, transformations_per_filter
    def __init__(self, project_variable):
        super(LeNet5_TTN3d, self).__init__()
        self.conv1 = ConvTTN3d(in_channels=1, out_channels=project_variable.num_out_channels[0], kernel_size=5, padding=2,
                               project_variable=project_variable, 
                               transformation_groups=project_variable.transformation_groups[0],
                               k0_groups=project_variable.k0_groups[0],
                               transformations_per_filter=project_variable.transformations_per_filter)
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2)
        if project_variable.do_batchnorm[0]:
            self.bn1 = torch.nn.BatchNorm3d(project_variable.num_out_channels[0])
        
        self.conv2 = ConvTTN3d(in_channels=project_variable.num_out_channels[0], out_channels=project_variable.num_out_channels[1], kernel_size=5, padding=0,
                               project_variable=project_variable, 
                               transformation_groups=project_variable.transformation_groups[1],
                               k0_groups=project_variable.k0_groups[1],
                               transformations_per_filter=project_variable.transformations_per_filter)
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2)
        if project_variable.do_batchnorm[1]:
            self.bn2 = torch.nn.BatchNorm3d(project_variable.num_out_channels[1])

        if project_variable.dataset == 'kth_actions':
            _fc_in = [73, 28, 38]
        else:
            _fc_in = [5, 5, 5]

        self.fc1 = torch.nn.Linear(project_variable.num_out_channels[1] * _fc_in[0] * _fc_in[1] * _fc_in[2],
                                   120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, project_variable.label_size)

    def forward(self, x, device):
        x = self.conv1(x, device)
        x = torch.nn.functional.relu(x)
        x = self.max_pool_1(x)
        x = self.conv2(x, device)
        x = torch.nn.functional.relu(x)
        x = self.max_pool_2(x)
        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        # _shape = x.shape
        # x = x.view(-1, _shape[1] * 5 * 5 * 5)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return x


# adapted from https://arxiv.org/abs/1907.11272
class Sota_3d(torch.nn.Module):
    def __init__(self, input_shape):
        t, h, w = input_shape
        super(Sota_3d, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=1,
                                      out_channels=16,
                                      kernel_size=(10, 32, 32),
                                      stride=1,
                                      padding=0,
                                      bias=True)

        t = t - 10 + 1
        h = h - 32 + 1
        w = w - 32 + 1

        self.prelu_1 = torch.nn.PReLU()
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=(3, 10, 10))

        t = int(np.floor(t / 3))
        h = int(np.floor(h / 10))
        w = int(np.floor(w / 10))

        if t == 0:
            t += 1
        if h == 0:
            h += 1
        if w == 0:
            w += 1

        self.conv2 = torch.nn.Conv3d(in_channels=16,
                                      out_channels=32,
                                      kernel_size=(3, 10, 10),
                                      stride=1,
                                      padding=1,
                                      bias=True)

        t = t+2 - 3 + 1
        h = h+2 - 10 + 1
        w = w+2 - 10 + 1

        self.prelu_2 = torch.nn.PReLU()
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=(1, 3, 3))

        t = int(np.floor(t / 1))
        h = int(np.floor(h / 3))
        w = int(np.floor(w / 3))

        if t == 0:
            t += 1
        if h == 0:
            h += 1
        if w == 0:
            w += 1

        in_features = t * h * w * 32
        self.fc1 = torch.nn.Linear(in_features=in_features,
                                    out_features=128)
        self.drop_1 = torch.nn.Dropout3d(p=0.5)
        self.fc2 = torch.nn.Linear(in_features=128,
                                    out_features=6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu_1(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = self.prelu_2(x)
        x = self.max_pool_2(x)
        _shape = x.shape
        x = x.view(-1, _shape[1]*_shape[2]*_shape[3]*_shape[4])
        x = self.fc1(x)
        x = self.drop_1(x)
        x = self.fc2(x)

        return x


def auto_in_features(input_shape, type, params, floor=True):
    t, h, w = input_shape

    assert type in ['conv', 'pool']

    # TODO: implement with stride
    if type == 'conv':
        k_t, k_h, k_w, pad = params
        t = t + 2 * pad - k_t + 1
        h = h + 2 * pad - k_h + 1
        w = w + 2 * pad - k_w + 1
    elif type == 'pool':
        k_t, k_h, k_w = params
        if floor:
            t = int(np.floor(t / k_t))
            h = int(np.floor(h / k_h))
            w = int(np.floor(w / k_w))
        else:
            t = int(np.ceil(t / k_t))
            h = int(np.ceil(h / k_h))
            w = int(np.ceil(w / k_w))

    return t, h, w


class C3D_experiment(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        print(t, h, w)
        channels = project_variable.num_out_channels
        # k_t, k_h, k_w = project_variable.k_shape
        do_batchnorm = project_variable.do_batchnorm

        super(C3D_experiment, self).__init__()

        self.conv1 = torch.nn.Conv3d(in_channels=1,
                                      out_channels=channels[0],
                                      kernel_size=(project_variable.conv1_k_t, 3, 3),
                                      stride=1,
                                      padding=0,
                                      bias=True)
        t, h, w = auto_in_features((t, h, w), 'conv', (project_variable.conv1_k_t, 3, 3, 0))
        print(t, h, w)
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
        print(t, h, w)
        if do_batchnorm[0]:
            self.bn1 = torch.nn.BatchNorm3d(channels[0])

        self.conv2 = torch.nn.Conv3d(in_channels=channels[0],
                                      out_channels=channels[1],
                                      kernel_size=(3, 3, 3),
                                      stride=1,
                                      padding=0,
                                      bias=True)
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
        print(t, h, w)
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
        print(t, h, w)
        if do_batchnorm[1]:
            self.bn2 = torch.nn.BatchNorm3d(channels[1])

        self.conv3 = torch.nn.Conv3d(in_channels=channels[1],
                                      out_channels=channels[2],
                                      kernel_size=(3, 3, 3),
                                      stride=1,
                                      padding=0,
                                      bias=True)
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
        print(t, h, w)
        # self.max_pool_3 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        # t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
        if do_batchnorm[2]:
            self.bn3 = torch.nn.BatchNorm3d(channels[2])

        self.conv4 = torch.nn.Conv3d(in_channels=channels[2],
                                       out_channels=channels[3],
                                       kernel_size=(3, 3, 3),
                                       stride=1,
                                       padding=0,
                                       bias=True)
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))

        # t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
        print(t, h, w, int(np.floor(t / 2)))
        if int(np.floor(t / 2)) == 0:
            print('set 1')
            self.max_pool_4 = torch.nn.MaxPool3d(kernel_size=(1, 2, 2))
            t, h, w = auto_in_features((t, h, w), 'pool', (1, 2, 2))
        else:
            self.max_pool_4 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
            t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))

        print(t, h, w)

        if do_batchnorm[3]:
            self.bn4 = torch.nn.BatchNorm3d(channels[3])

        in_features = t * h * w * channels[3]
        self.fc1 = torch.nn.Linear(in_features=in_features,
                                    out_features=in_features+128)
        if do_batchnorm[4]:
            self.bn5 = torch.nn.BatchNorm1d(in_features+128)

        self.fc2 = torch.nn.Linear(in_features=in_features+128,
                                    out_features=6)


    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn1(x)
        except AttributeError:
            pass

        x = self.conv2(x)
        x = self.max_pool_2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn2(x)
        except AttributeError:
            pass

        x = self.conv3(x)
        # x = self.max_pool_3(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn3(x)
        except AttributeError:
            pass

        x = self.conv4(x)
        x = self.max_pool_4(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn4(x)
        except AttributeError:
            pass

        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn5(x)
        except AttributeError:
            pass

        x = self.fc2(x)

        return x


class C3D(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        channels = project_variable.num_out_channels

        super(C3D, self).__init__()

        self.conv1 = torch.nn.Conv3d(in_channels=1,
                                      out_channels=channels[0],
                                      kernel_size=(3, 3, 3),
                                      stride=1,
                                      padding=0,
                                      bias=True)
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
        self.bn1 = torch.nn.BatchNorm3d(channels[0])

        self.conv2 = torch.nn.Conv3d(in_channels=channels[0],
                                      out_channels=channels[1],
                                      kernel_size=(3, 3, 3),
                                      stride=1,
                                      padding=0,
                                      bias=True)
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))

        self.conv3 = torch.nn.Conv3d(in_channels=channels[1],
                                      out_channels=channels[2],
                                      kernel_size=(3, 3, 3),
                                      stride=1,
                                      padding=0,
                                      bias=True)
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))

        self.conv4 = torch.nn.Conv3d(in_channels=channels[2],
                                       out_channels=channels[3],
                                       kernel_size=(3, 3, 3),
                                       stride=1,
                                       padding=0,
                                       bias=True)
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))

        self.max_pool_4 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))

        in_features = t * h * w * channels[3]
        self.fc1 = torch.nn.Linear(in_features=in_features,
                                    out_features=in_features+128)

        self.fc2 = torch.nn.Linear(in_features=in_features+128,
                                    out_features=6)


    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.max_pool_2(x)
        x = torch.nn.functional.relu(x)

        x = self.conv3(x)
        # x = self.max_pool_3(x)
        x = torch.nn.functional.relu(x)

        x = self.conv4(x)
        x = self.max_pool_4(x)
        x = torch.nn.functional.relu(x)

        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)

        return x


class C3DTTN_after(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        channels = project_variable.num_out_channels
        trafo_groups = project_variable.transformation_groups
        k0_groups = project_variable.k0_groups

        super(C3DTTN_after, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=1,
                                out_channels=channels[0],
                                kernel_size=(3, 3, 3),
                                stride=1,
                                padding=0,
                                bias=True,
                                project_variable=project_variable,
                                transformation_groups=trafo_groups[0],
                                k0_groups=k0_groups[0])
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
        self.bn1 = torch.nn.BatchNorm3d(channels[0])

        self.conv2 = ConvTTN3d(in_channels=channels[0],
                                out_channels=channels[1],
                                kernel_size=(3, 3, 3),
                                stride=1,
                                padding=0,
                                bias=True,
                                project_variable=project_variable,
                                transformation_groups=trafo_groups[1],
                                k0_groups=k0_groups[1])
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))

        self.conv3 = ConvTTN3d(in_channels=channels[1],
                                out_channels=channels[2],
                                kernel_size=(3, 3, 3),
                                stride=1,
                                padding=0,
                                bias=True,
                                project_variable=project_variable,
                                transformation_groups=trafo_groups[2],
                                k0_groups=k0_groups[2])
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))

        self.conv4 = ConvTTN3d(in_channels=channels[2],
                                out_channels=channels[3],
                                kernel_size=(3, 3, 3),
                                stride=1,
                                padding=0,
                                bias=True,
                                project_variable=project_variable,
                                transformation_groups=trafo_groups[3],
                                k0_groups=k0_groups[3])
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))

        self.max_pool_4 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))

        in_features = t * h * w * channels[3]
        self.fc1 = torch.nn.Linear(in_features=in_features,
                                    out_features=in_features + 128)

        self.fc2 = torch.nn.Linear(in_features=in_features + 128,
                                    out_features=6)


    def forward(self, x, device):
        x = self.conv1(x, device)
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn1(x)

        x = self.conv2(x, device)
        x = self.max_pool_2(x)
        x = torch.nn.functional.relu(x)

        x = self.conv3(x, device)
        x = torch.nn.functional.relu(x)

        x = self.conv4(x, device)
        x = self.max_pool_4(x)
        x = torch.nn.functional.relu(x)

        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)

        return x

class C3DTTN(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        print(t, h, w)
        channels = project_variable.num_out_channels
        do_batchnorm = project_variable.do_batchnorm
        trafo_groups = project_variable.transformation_groups
        k0_groups = project_variable.k0_groups

        super(C3DTTN, self).__init__()


        self.conv1 = ConvTTN3d(in_channels=1,
                                      out_channels=channels[0],
                                      kernel_size=(project_variable.conv1_k_t, 3, 3),
                                      stride=1,
                                      padding=0,
                                      bias=True,
                                project_variable=project_variable,
                                transformation_groups=trafo_groups[0],
                                k0_groups=k0_groups[0])
        t, h, w = auto_in_features((t, h, w), 'conv', (project_variable.conv1_k_t, 3, 3, 0))
        print(t, h, w)
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
        print(t, h, w)
        if do_batchnorm[0]:
            self.bn1 = torch.nn.BatchNorm3d(channels[0])

        self.conv2 = ConvTTN3d(in_channels=channels[0],
                                      out_channels=channels[1],
                                      kernel_size=(3, 3, 3),
                                      stride=1,
                                      padding=0,
                                      bias=True,
                                project_variable=project_variable,
                                transformation_groups=trafo_groups[1],
                                k0_groups=k0_groups[1])
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
        print(t, h, w)
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
        print(t, h, w)
        if do_batchnorm[1]:
            self.bn2 = torch.nn.BatchNorm3d(channels[1])

        self.conv3 = ConvTTN3d(in_channels=channels[1],
                                      out_channels=channels[2],
                                      kernel_size=(3, 3, 3),
                                      stride=1,
                                      padding=0,
                                      bias=True,
                                project_variable=project_variable,
                                transformation_groups=trafo_groups[2],
                                k0_groups=k0_groups[2])
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
        print(t, h, w)
        # self.max_pool_3 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        # t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
        if do_batchnorm[2]:
            self.bn3 = torch.nn.BatchNorm3d(channels[2])

        self.conv4 = ConvTTN3d(in_channels=channels[2],
                                       out_channels=channels[3],
                                       kernel_size=(3, 3, 3),
                                       stride=1,
                                       padding=0,
                                       bias=True,
                                project_variable=project_variable,
                                transformation_groups=trafo_groups[3],
                                k0_groups=k0_groups[3])
        t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))

        # t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
        print(t, h, w, int(np.floor(t / 2)))
        if int(np.floor(t / 2)) == 0:
            print('set 1')
            self.max_pool_4 = torch.nn.MaxPool3d(kernel_size=(1, 2, 2))
            t, h, w = auto_in_features((t, h, w), 'pool', (1, 2, 2))
        else:
            self.max_pool_4 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
            t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))

        print(t, h, w)

        if do_batchnorm[3]:
            self.bn4 = torch.nn.BatchNorm3d(channels[3])

        in_features = t * h * w * channels[3]
        self.fc1 = torch.nn.Linear(in_features=in_features,
                                    out_features=in_features+128)
        if do_batchnorm[4]:
            self.bn5 = torch.nn.BatchNorm1d(in_features+128)

        self.fc2 = torch.nn.Linear(in_features=in_features+128,
                                    out_features=6)


    def forward(self, x, device):
        x = self.conv1(x, device)
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn1(x)
        except AttributeError:
            pass

        x = self.conv2(x, device)
        x = self.max_pool_2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn2(x)
        except AttributeError:
            pass

        x = self.conv3(x, device)
        # x = self.max_pool_3(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn3(x)
        except AttributeError:
            pass

        x = self.conv4(x, device)
        x = self.max_pool_4(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn4(x)
        except AttributeError:
            pass

        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn5(x)
        except AttributeError:
            pass

        x = self.fc2(x)

        return x


# 30, 3
class C3TTN1(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        print(t, h, w)
        channels = project_variable.num_out_channels
        do_batchnorm = project_variable.do_batchnorm
        trafo_groups = project_variable.transformation_groups
        k0_groups = project_variable.k0_groups
        max_pool_temp = project_variable.max_pool_temporal
        kt, kh, kw = project_variable.k_shape

        super(C3TTN1, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=1,
                               out_channels=channels[0],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[0],
                               k0_groups=k0_groups[0])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        print(t, h, w)

        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[0]:
            self.bn1 = torch.nn.BatchNorm3d(channels[0])

        self.conv2 = ConvTTN3d(in_channels=channels[0],
                               out_channels=channels[1],
                               kernel_size=(kt*2, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[1],
                               k0_groups=k0_groups[1])

        t, h, w = auto_in_features((t, h, w), 'conv', (kt*2, kh, kw, 0))
        print(t, h, w)

        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[1]:
            self.bn2 = torch.nn.BatchNorm3d(channels[1])

        self.conv3 = ConvTTN3d(in_channels=channels[1],
                               out_channels=channels[2],
                               kernel_size=(kt*3, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[2],
                               k0_groups=k0_groups[2])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt*3, kh, kw, 0))
        print(t, h, w)
        if do_batchnorm[2]:
            self.bn3 = torch.nn.BatchNorm3d(channels[2])

        self.conv4 = ConvTTN3d(in_channels=channels[2],
                               out_channels=channels[3],
                               kernel_size=(kt*4, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[3],
                               k0_groups=k0_groups[3])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt*4, kh, kw, 0))

        self.max_pool_3 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[3]:
            self.bn4 = torch.nn.BatchNorm3d(channels[3])

        in_features = t * h * w * channels[3]  # 4096
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=1024
                                   )
        if do_batchnorm[4]:
            self.bn5 = torch.nn.BatchNorm1d(1024)

        self.fc2 = torch.nn.Linear(in_features=1024,
                                   out_features=512)

        if do_batchnorm[5]:
            self.bn6 = torch.nn.BatchNorm1d(512)

        self.fc3 = torch.nn.Linear(in_features=512,
                                   out_features=6)



    def forward(self, x, device):
        x = self.conv1(x, device)
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn1(x)
        except AttributeError:
            pass

        x = self.conv2(x, device)
        x = self.max_pool_2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn2(x)
        except AttributeError:
            pass

        x = self.conv3(x, device)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn3(x)
        except AttributeError:
            pass

        x = self.conv4(x, device)
        x = self.max_pool_3(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn4(x)
        except AttributeError:
            pass

        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn5(x)
        except AttributeError:
            pass

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn6(x)
        except AttributeError:
            pass

        x = self.fc3(x)

        return x


# 30, 5
class C3TTN2(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        print(t, h, w)
        channels = project_variable.num_out_channels
        do_batchnorm = project_variable.do_batchnorm
        trafo_groups = project_variable.transformation_groups
        k0_groups = project_variable.k0_groups
        max_pool_temp = project_variable.max_pool_temporal
        kt, kh, kw = project_variable.k_shape

        super(C3TTN2, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=1,
                               out_channels=channels[0],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[0],
                               k0_groups=k0_groups[0])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        print(t, h, w)

        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[0]:
            self.bn1 = torch.nn.BatchNorm3d(channels[0])

        self.conv2 = ConvTTN3d(in_channels=channels[0],
                               out_channels=channels[1],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[1],
                               k0_groups=k0_groups[1])

        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        print(t, h, w)

        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[1]:
            self.bn2 = torch.nn.BatchNorm3d(channels[1])

        self.conv3 = ConvTTN3d(in_channels=channels[1],
                               out_channels=channels[2],
                               kernel_size=(kt * 2, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[2],
                               k0_groups=k0_groups[2])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt * 2, kh, kw, 0))
        print(t, h, w)
        if do_batchnorm[2]:
            self.bn3 = torch.nn.BatchNorm3d(channels[2])

        self.conv4 = ConvTTN3d(in_channels=channels[2],
                               out_channels=channels[3],
                               kernel_size=(kt * 2, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[3],
                               k0_groups=k0_groups[3])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt * 2, kh, kw, 0))

        self.max_pool_3 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[3]:
            self.bn4 = torch.nn.BatchNorm3d(channels[3])

        in_features = t * h * w * channels[3] # 4096
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=1024
                                   )
        if do_batchnorm[4]:
            self.bn5 = torch.nn.BatchNorm1d(1024)

        self.fc2 = torch.nn.Linear(in_features=1024,
                                   out_features=512)

        if do_batchnorm[5]:
            self.bn6 = torch.nn.BatchNorm1d(512)

        self.fc3 = torch.nn.Linear(in_features=512,
                                   out_features=6)

    def forward(self, x, device):
        x = self.conv1(x, device)
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn1(x)
        except AttributeError:
            pass

        x = self.conv2(x, device)
        x = self.max_pool_2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn2(x)
        except AttributeError:
            pass

        x = self.conv3(x, device)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn3(x)
        except AttributeError:
            pass

        x = self.conv4(x, device)
        x = self.max_pool_3(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn4(x)
        except AttributeError:
            pass

        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn5(x)
        except AttributeError:
            pass

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn6(x)
        except AttributeError:
            pass

        x = self.fc3(x)

        return x


# 30, 7
class C3TTN3(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        print(t, h, w)
        channels = project_variable.num_out_channels
        do_batchnorm = project_variable.do_batchnorm
        trafo_groups = project_variable.transformation_groups
        k0_groups = project_variable.k0_groups
        max_pool_temp = project_variable.max_pool_temporal
        kt, kh, kw = project_variable.k_shape

        super(C3TTN3, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=1,
                               out_channels=channels[0],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[0],
                               k0_groups=k0_groups[0])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        print(t, h, w)

        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[0]:
            self.bn1 = torch.nn.BatchNorm3d(channels[0])

        self.conv2 = ConvTTN3d(in_channels=channels[0],
                               out_channels=channels[1],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[1],
                               k0_groups=k0_groups[1])

        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        print(t, h, w)

        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[1]:
            self.bn2 = torch.nn.BatchNorm3d(channels[1])

        self.conv3 = ConvTTN3d(in_channels=channels[1],
                               out_channels=channels[2],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[2],
                               k0_groups=k0_groups[2])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        print(t, h, w)
        if do_batchnorm[2]:
            self.bn3 = torch.nn.BatchNorm3d(channels[2])

        self.conv4 = ConvTTN3d(in_channels=channels[2],
                               out_channels=channels[3],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[3],
                               k0_groups=k0_groups[3])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))

        self.max_pool_3 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[3]:
            self.bn4 = torch.nn.BatchNorm3d(channels[3])

        in_features = t * h * w * channels[3]  # 6144
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=1024
                                   )
        if do_batchnorm[4]:
            self.bn5 = torch.nn.BatchNorm1d(1024)

        self.fc2 = torch.nn.Linear(in_features=1024,
                                   out_features=512)

        if do_batchnorm[5]:
            self.bn6 = torch.nn.BatchNorm1d(512)

        self.fc3 = torch.nn.Linear(in_features=512,
                                   out_features=6)

    def forward(self, x, device):
        x = self.conv1(x, device)
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn1(x)
        except AttributeError:
            pass

        x = self.conv2(x, device)
        x = self.max_pool_2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn2(x)
        except AttributeError:
            pass

        x = self.conv3(x, device)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn3(x)
        except AttributeError:
            pass

        x = self.conv4(x, device)
        x = self.max_pool_3(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn4(x)
        except AttributeError:
            pass

        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn5(x)
        except AttributeError:
            pass

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn6(x)
        except AttributeError:
            pass

        x = self.fc3(x)

        return x


# 30, 9
class C3TTN4(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        print(t, h, w)
        channels = project_variable.num_out_channels
        do_batchnorm = project_variable.do_batchnorm
        trafo_groups = project_variable.transformation_groups
        k0_groups = project_variable.k0_groups
        max_pool_temp = project_variable.max_pool_temporal
        kt, kh, kw = project_variable.k_shape

        super(C3TTN4, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=1,
                               out_channels=channels[0],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[0],
                               k0_groups=k0_groups[0])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        print(t, h, w)

        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[0]:
            self.bn1 = torch.nn.BatchNorm3d(channels[0])

        self.conv2 = ConvTTN3d(in_channels=channels[0],
                               out_channels=channels[1],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[1],
                               k0_groups=k0_groups[1])

        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        print(t, h, w)

        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[1]:
            self.bn2 = torch.nn.BatchNorm3d(channels[1])

        self.conv3 = ConvTTN3d(in_channels=channels[1],
                               out_channels=channels[2],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[2],
                               k0_groups=k0_groups[2])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        print(t, h, w)

        self.max_pool_3 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[2]:
            self.bn3 = torch.nn.BatchNorm3d(channels[2])

        in_features = t * h * w * channels[2] # 9600
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=1024
                                   )
        if do_batchnorm[4]:
            self.bn4 = torch.nn.BatchNorm1d(1024)

        self.fc2 = torch.nn.Linear(in_features=1024,
                                   out_features=512)

        if do_batchnorm[5]:
            self.bn5 = torch.nn.BatchNorm1d(512)

        self.fc3 = torch.nn.Linear(in_features=512,
                                   out_features=6)

    def forward(self, x, device):
        x = self.conv1(x, device)
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn1(x)
        except AttributeError:
            pass

        x = self.conv2(x, device)
        x = self.max_pool_2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn2(x)
        except AttributeError:
            pass

        x = self.conv3(x, device)
        x = self.max_pool_3(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn3(x)
        except AttributeError:
            pass

        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn4(x)
        except AttributeError:
            pass

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn5(x)
        except AttributeError:
            pass

        x = self.fc3(x)

        return x

# 100, 7
class C3TTN5(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        print(t, h, w)
        channels = project_variable.num_out_channels
        do_batchnorm = project_variable.do_batchnorm
        trafo_groups = project_variable.transformation_groups
        k0_groups = project_variable.k0_groups
        max_pool_temp = project_variable.max_pool_temporal
        kt, kh, kw = project_variable.k_shape

        super(C3TTN5, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=1,
                               out_channels=channels[0],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[0],
                               k0_groups=k0_groups[0])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        print(t, h, w)

        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[0]:
            self.bn1 = torch.nn.BatchNorm3d(channels[0])

        self.conv2 = ConvTTN3d(in_channels=channels[0],
                               out_channels=channels[1],
                               kernel_size=(kt*2, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[1],
                               k0_groups=k0_groups[1])

        t, h, w = auto_in_features((t, h, w), 'conv', (kt*2, kh, kw, 0))
        print(t, h, w)

        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[1]:
            self.bn2 = torch.nn.BatchNorm3d(channels[1])

        self.conv3 = ConvTTN3d(in_channels=channels[1],
                               out_channels=channels[2],
                               kernel_size=(kt*5, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[2],
                               k0_groups=k0_groups[2])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt*5, kh, kw, 0))
        print(t, h, w)

        if do_batchnorm[2]:
            self.bn3 = torch.nn.BatchNorm3d(channels[2])

        self.conv4 = ConvTTN3d(in_channels=channels[2],
                               out_channels=channels[3],
                               kernel_size=(kt * 6, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[3],
                               k0_groups=k0_groups[3])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt * 6, kh, kw, 0))

        self.max_pool_3 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[3]:
            self.bn4 = torch.nn.BatchNorm3d(channels[3])

        in_features = t * h * w * channels[3] # 6144
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=1024
                                   )
        if do_batchnorm[4]:
            self.bn5 = torch.nn.BatchNorm1d(1024)

        self.fc2 = torch.nn.Linear(in_features=1024,
                                   out_features=512)

        if do_batchnorm[5]:
            self.bn6 = torch.nn.BatchNorm1d(512)

        self.fc3 = torch.nn.Linear(in_features=512,
                                   out_features=6)

    def forward(self, x, device):
        x = self.conv1(x, device)
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn1(x)
        except AttributeError:
            pass

        x = self.conv2(x, device)
        x = self.max_pool_2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn2(x)
        except AttributeError:
            pass

        x = self.conv3(x, device)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn3(x)
        except AttributeError:
            pass

        x = self.conv4(x, device)
        x = self.max_pool_3(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn4(x)
        except AttributeError:
            pass

        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn5(x)
        except AttributeError:
            pass

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn6(x)
        except AttributeError:
            pass

        x = self.fc3(x)

        return x


# 100, 9
class C3TTN6(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        print(t, h, w)
        channels = project_variable.num_out_channels
        do_batchnorm = project_variable.do_batchnorm
        trafo_groups = project_variable.transformation_groups
        k0_groups = project_variable.k0_groups
        max_pool_temp = project_variable.max_pool_temporal
        kt, kh, kw = project_variable.k_shape

        super(C3TTN6, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=1,
                               out_channels=channels[0],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[0],
                               k0_groups=k0_groups[0])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        print(t, h, w)

        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[0]:
            self.bn1 = torch.nn.BatchNorm3d(channels[0])

        self.conv2 = ConvTTN3d(in_channels=channels[0],
                               out_channels=channels[1],
                               kernel_size=(kt*2, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[1],
                               k0_groups=k0_groups[1])

        t, h, w = auto_in_features((t, h, w), 'conv', (kt*2, kh, kw, 0))
        print(t, h, w)

        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[1]:
            self.bn2 = torch.nn.BatchNorm3d(channels[1])

        self.conv3 = ConvTTN3d(in_channels=channels[1],
                               out_channels=channels[2],
                               kernel_size=(kt*4, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[2],
                               k0_groups=k0_groups[2])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt*4, kh, kw, 0))
        print(t, h, w)

        if do_batchnorm[2]:
            self.bn3 = torch.nn.BatchNorm3d(channels[2])

        self.conv4 = ConvTTN3d(in_channels=channels[2],
                               out_channels=channels[3],
                               kernel_size=(kt * 4, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[3],
                               k0_groups=k0_groups[3])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt * 4, kh, kw, 0))

        self.max_pool_3 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[3]:
            self.bn4 = torch.nn.BatchNorm3d(channels[3])

        in_features = t * h * w * channels[3] # 5120
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=1024
                                   )
        if do_batchnorm[4]:
            self.bn5 = torch.nn.BatchNorm1d(1024)

        self.fc2 = torch.nn.Linear(in_features=1024,
                                   out_features=512)

        if do_batchnorm[5]:
            self.bn6 = torch.nn.BatchNorm1d(512)

        self.fc3 = torch.nn.Linear(in_features=512,
                                   out_features=6)

    def forward(self, x, device):
        x = self.conv1(x, device)
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn1(x)
        except AttributeError:
            pass

        x = self.conv2(x, device)
        x = self.max_pool_2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn2(x)
        except AttributeError:
            pass

        x = self.conv3(x, device)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn3(x)
        except AttributeError:
            pass

        x = self.conv4(x, device)
        x = self.max_pool_3(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn4(x)
        except AttributeError:
            pass

        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn5(x)
        except AttributeError:
            pass

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn6(x)
        except AttributeError:
            pass

        x = self.fc3(x)

        return x


# 100, 11
class C3TTN7(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        print(t, h, w)
        channels = project_variable.num_out_channels
        do_batchnorm = project_variable.do_batchnorm
        trafo_groups = project_variable.transformation_groups
        k0_groups = project_variable.k0_groups
        max_pool_temp = project_variable.max_pool_temporal
        kt, kh, kw = project_variable.k_shape

        super(C3TTN7, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=1,
                               out_channels=channels[0],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[0],
                               k0_groups=k0_groups[0])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        print(t, h, w)

        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[0]:
            self.bn1 = torch.nn.BatchNorm3d(channels[0])

        self.conv2 = ConvTTN3d(in_channels=channels[0],
                               out_channels=channels[1],
                               kernel_size=(kt*2, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[1],
                               k0_groups=k0_groups[1])

        t, h, w = auto_in_features((t, h, w), 'conv', (kt*2, kh, kw, 0))
        print(t, h, w)

        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[1]:
            self.bn2 = torch.nn.BatchNorm3d(channels[1])

        self.conv3 = ConvTTN3d(in_channels=channels[1],
                               out_channels=channels[2],
                               kernel_size=(kt*3, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[2],
                               k0_groups=k0_groups[2])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt*3, kh, kw, 0))
        print(t, h, w)

        if do_batchnorm[2]:
            self.bn3 = torch.nn.BatchNorm3d(channels[2])

        self.conv4 = ConvTTN3d(in_channels=channels[2],
                               out_channels=channels[3],
                               kernel_size=(kt * 3, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[3],
                               k0_groups=k0_groups[3])
        t, h, w = auto_in_features((t, h, w), 'conv', (kt * 3, kh, kw, 0))

        self.max_pool_3 = torch.nn.MaxPool3d(kernel_size=(max_pool_temp, 2, 2))
        t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp, 2, 2))
        print(t, h, w)

        if do_batchnorm[3]:
            self.bn4 = torch.nn.BatchNorm3d(channels[3])

        in_features = t * h * w * channels[3] # 5120
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=1024
                                   )
        if do_batchnorm[4]:
            self.bn5 = torch.nn.BatchNorm1d(1024)

        self.fc2 = torch.nn.Linear(in_features=1024,
                                   out_features=512)

        if do_batchnorm[5]:
            self.bn6 = torch.nn.BatchNorm1d(512)

        self.fc3 = torch.nn.Linear(in_features=512,
                                   out_features=6)

    def forward(self, x, device):
        x = self.conv1(x, device)
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn1(x)
        except AttributeError:
            pass

        x = self.conv2(x, device)
        x = self.max_pool_2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn2(x)
        except AttributeError:
            pass

        x = self.conv3(x, device)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn3(x)
        except AttributeError:
            pass

        x = self.conv4(x, device)
        x = self.max_pool_3(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn4(x)
        except AttributeError:
            pass

        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn5(x)
        except AttributeError:
            pass

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        try:
            x = self.bn6(x)
        except AttributeError:
            pass

        x = self.fc3(x)

        return x


class C3DTTN_1L(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        channels = project_variable.num_out_channels
        trafo_groups = project_variable.transformation_groups
        k0_groups = project_variable.k0_groups
        kt, kh, kw = project_variable.k_shape
        trafo_per_filter = project_variable.transformations_per_filter

        super(C3DTTN_1L, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=1,
                               out_channels=channels[0],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True,
                               project_variable=project_variable,
                               transformation_groups=trafo_groups[0],
                               k0_groups=k0_groups[0],
                               transformations_per_filter=trafo_per_filter)

        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))

        in_features = t * h * w * channels[0]
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=project_variable.label_size)


    def forward(self, x, device):
        x = self.conv1(x, device)
        x = torch.nn.functional.relu(x)
        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)

        return x


class C3D_1L(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        channels = project_variable.num_out_channels
        kt, kh, kw = project_variable.k_shape

        super(C3D_1L, self).__init__()

        self.conv1 = torch.nn.Conv3d(in_channels=1,
                               out_channels=channels[0],
                               kernel_size=(kt, kh, kw),
                               stride=1,
                               padding=0,
                               bias=True)

        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))

        in_features = t * h * w * channels[0]
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=project_variable.label_size)


    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        return x


# works for data with more than one channel
class LeNet5_TTN3d_xD(torch.nn.Module):
    def __init__(self, input_shape, project_variable):
        t, h, w = input_shape
        kt, kh, kw = project_variable.k_shape
        
        self.return_ind = False
        if project_variable.return_ind:
            self.return_ind = True
        
        super(LeNet5_TTN3d_xD, self).__init__()
        self.conv1 = ConvTTN3d(in_channels=project_variable.num_in_channels, out_channels=project_variable.num_out_channels[0], kernel_size=5,
                               padding=2,
                               project_variable=project_variable,
                               transformation_groups=project_variable.transformation_groups[0],
                               k0_groups=project_variable.k0_groups[0],
                               transformations_per_filter=project_variable.transformations_per_filter)

        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))

        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2, return_indices=self.return_ind)

        t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))

        self.conv2 = ConvTTN3d(in_channels=project_variable.num_out_channels[0],
                               out_channels=project_variable.num_out_channels[1], kernel_size=5, padding=0,
                               project_variable=project_variable,
                               transformation_groups=project_variable.transformation_groups[1],
                               k0_groups=project_variable.k0_groups[1],
                               transformations_per_filter=project_variable.transformations_per_filter)

        t, h, w = auto_in_features((t, h, w), 'conv', (kt, kh, kw, 0))
        
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2, return_indices=self.return_ind)

        t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))

        if project_variable.dataset == 'kth_actions':
            _fc_in = [73, 28, 38]
        elif project_variable.dataset == 'omg_emotion':
            if project_variable.num_out_channels == [6, 16]:
                in_features = 100672
            elif project_variable.num_out_channels == [12, 22]:
                in_features = 138424
        elif project_variable.dataset == 'mov_mnist':
            if project_variable.num_out_channels == [6, 16]:
                # TODO
                in_features = 2000
        elif project_variable.dataset == 'dhg':
            if project_variable.num_out_channels == [6, 16]:
                in_features = 4000
            elif project_variable.num_out_channels == [4, 14]:
                in_features = 3500
            elif project_variable.num_out_channels == [8, 18]:
                in_features = 4500
            elif project_variable.num_out_channels == [12, 22]:
                in_features = 5500
        elif project_variable.dataset == 'jester':
            if project_variable.num_out_channels == [6, 16]:
                in_features = 12800
            elif project_variable.num_out_channels == [8, 18]:
                in_features = 14400
            elif project_variable.num_out_channels == [12, 22]:
                in_features = 17600
        else:
            _fc_in = [5, 5, 5]

        in_features_tmp = t * h * w * project_variable.num_out_channels[1]
        self.fc1 = torch.nn.Linear(in_features, 120)
        # self.fc1 = torch.nn.Linear(project_variable.num_out_channels[1] * _fc_in[0] * _fc_in[1] * _fc_in[2],
        #                            120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, project_variable.label_size)

    def forward(self, x, device):
        x = self.conv1(x, device)
        x = torch.nn.functional.relu(x)
        
        if self.return_ind:
            x, ind1 = self.max_pool_1(x)
        else:
            x = self.max_pool_1(x)
            
        x = self.conv2(x, device)
        x = torch.nn.functional.relu(x)
        
        if self.return_ind:
            x, ind2 = self.max_pool_2(x)
        else:
            x = self.max_pool_2(x)
            
        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        # _shape = x.shape
        # x = x.view(-1, _shape[1] * 5 * 5 * 5)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)

        return x


def isnan(x, name):
    if True in np.ravel(np.array(torch.isnan(x).cpu())) or True in np.ravel(np.array(torch.isinf(x).cpu())):
        print('NaN or Inf in %s' % name)


class LeNet5_3d_xD(torch.nn.Module):

    def __init__(self, project_variable):
        self.return_ind = False
        if project_variable.return_ind:
            self.return_ind = True

        super(LeNet5_3d_xD, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=project_variable.num_in_channels, out_channels=project_variable.num_out_channels[0],
                                     kernel_size=project_variable.k_shape, stride=1, padding=2, bias=True)
        if project_variable.k0_init == 'normal':
            self.conv1.weight = torch.nn.init.normal_(self.conv1.weight)
            self.conv1.bias = torch.nn.init.normal_(self.conv1.bias)
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2, return_indices=self.return_ind)

        self.conv2 = torch.nn.Conv3d(in_channels=project_variable.num_out_channels[0],
                                     out_channels=project_variable.num_out_channels[1],
                                     kernel_size=project_variable.k_shape, stride=1, padding=0, bias=True)
        if project_variable.k0_init == 'normal':
            self.conv2.weight = torch.nn.init.normal_(self.conv2.weight)
            self.conv2.bias = torch.nn.init.normal_(self.conv2.bias)

        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2, return_indices=self.return_ind)

        in_features = 420.69

        if project_variable.dataset == 'kth_actions':
            _fc_in = [73, 28, 38]
        elif project_variable.dataset == 'omg_emotion':
            if project_variable.num_out_channels == [6, 16]:
                in_features = 100672
            elif project_variable.num_out_channels == [12, 22]:
                in_features = 138424
        elif project_variable.dataset == 'mov_mnist':
            if project_variable.num_out_channels == [6, 16]:
                in_features = 2000
        elif project_variable.dataset == 'dhg':
            if project_variable.num_out_channels == [6, 16]:
                in_features = 4000
            elif project_variable.num_out_channels == [4, 14]:
                in_features = 3500
            elif project_variable.num_out_channels == [8, 18]:
                in_features = 4500
            elif project_variable.num_out_channels == [12, 22]:
                in_features = 5500
        elif project_variable.dataset == 'jester':
            if project_variable.num_out_channels == [6, 16]:
                in_features = 12800
            elif project_variable.num_out_channels == [8, 18]:
                in_features = 14400
            elif project_variable.num_out_channels == [12, 22]:
                in_features = 17600

        self.fc1 = torch.nn.Linear(in_features,
                                   120)  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)  # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84,
                                   project_variable.label_size)  # convert matrix with 84 features to a matrix of 10 features (columns)

    def forward(self, x):
        x = self.conv1(x)
        # if sum(np.isnan(x)) + sum(np.inf(x)) > 0:
        #     print('NaN or inf in conv1')
        # maybe shouldnt check here, too many input

        x = torch.nn.functional.relu(x)
        if self.return_ind:
            x, ind1 = self.max_pool_1(x)
        else:
            x = self.max_pool_1(x)
        isnan(x, 'max_pool_1')

        x = self.conv2(x)
        isnan(x, 'conv2')
        if self.return_ind:
            x, ind2 = self.max_pool_2(x)
        else:
            x = self.max_pool_2(x)
        isnan(x, 'max_pool_2')

        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = torch.nn.functional.relu(self.fc1(x))
        isnan(x, 'fc1')
        x = torch.nn.functional.relu(self.fc2(x))
        isnan(x, 'fc2')
        x = self.fc3(x)
        isnan(x, 'fc3')

        return x


class LeNet5_2d_xD(torch.nn.Module):

    def __init__(self, project_variable):
        super(LeNet5_2d_xD, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=project_variable.num_in_channels, out_channels=project_variable.num_out_channels[0],
                                     kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(in_channels=project_variable.num_out_channels[0], out_channels=project_variable.num_out_channels[1],
                                     kernel_size=5, stride=1, padding=0, bias=True)
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


class deconv_3DTTN(torch.nn.Module):

    def __init__(self, which_conv):
        super(deconv_3DTTN, self).__init__()

        if which_conv == 'conv2':
            # relu
            # unpool 2 5x5
            self.unpool2 = torch.nn.MaxUnpool3d(kernel_size=2)
            # deconv 2
            self.deconv2 = torch.nn.ConvTranspose3d(in_channels=16,
                                                    out_channels=6,
                                                    kernel_size=5,
                                                    padding=0,
                                                    bias=False)
            # 10x10
            # self.deconv2 = torch.nn.Conv3d(in_channels=16,
            #                                out_channels=6,
            #                                kernel_size=5, # flor k/2
            #                                padding=4,
            #                                bias=False)

        # relu
        # unpool 1 14x14
        self.unpool1 = torch.nn.MaxUnpool3d(kernel_size=2)
        # deconv 1
        self.deconv1 = torch.nn.ConvTranspose3d(in_channels=6,
                                                out_channels=1,
                                                kernel_size=5,
                                                padding=2,
                                                bias=False)
        #  28x28
        # self.deconv1 = torch.nn.Conv3d(in_channels=6,
        #                                out_channels=1,
        #                                kernel_size=5,
        #                                padding=2,
        #                                bias=False) #28x28 + 2

        self.which_conv = which_conv

    def forward(self, x, pool_switches):
        if self.which_conv == 'conv2':
            x = torch.nn.functional.relu(x)
            x = self.unpool2(x, pool_switches[1], torch.Size([1, 16, 11, 10, 10]))  # video mnist
            # x = self.unpool2(x, pool_switches[1], torch.Size([1, 16, 21, 10, 10])) # dhg
            x = self.deconv2(x)

        x = torch.nn.functional.relu(x)
        x = self.unpool1(x, pool_switches[0])
        x = self.deconv1(x)

        return x

class deconv_3D(torch.nn.Module):

    def __init__(self, which_conv):
        super(deconv_3D, self).__init__()

        if which_conv == 'conv2':
            # relu
            # unpool 2 5x5
            self.unpool2 = torch.nn.MaxUnpool3d(kernel_size=2)
            # deconv 2
            self.deconv2 = torch.nn.ConvTranspose3d(in_channels=16,
                                                    out_channels=6,
                                                    kernel_size=5,
                                                    padding=0,
                                                    bias=False)
            # 10x10
            # self.deconv2 = torch.nn.Conv3d(in_channels=16,
            #                                out_channels=6,
            #                                kernel_size=5, # flor k/2
            #                                padding=4,
            #                                bias=False)

        # relu
        # unpool 1 14x14
        self.unpool1 = torch.nn.MaxUnpool3d(kernel_size=2)
        # deconv 1
        self.deconv1 = torch.nn.ConvTranspose3d(in_channels=6,
                                                out_channels=1,
                                                kernel_size=5,
                                                padding=2,
                                                bias=False)
        #  28x28
        # self.deconv1 = torch.nn.Conv3d(in_channels=6,
        #                                out_channels=1,
        #                                kernel_size=5,
        #                                padding=2,
        #                                bias=False) #28x28 + 2

        self.which_conv = which_conv

    def forward(self, x, pool_switches):
        if self.which_conv == 'conv2':
            x = torch.nn.functional.relu(x)
            x = self.unpool2(x, pool_switches[1], torch.Size([1, 16, 11, 10, 10])) # video mnist
            # x = self.unpool2(x, pool_switches[1], torch.Size([1, 16, 21, 10, 10])) # dhg
            x = self.deconv2(x)

        x = torch.nn.functional.relu(x)
        x = self.unpool1(x, pool_switches[0])
        x = self.deconv1(x)

        return x


# TODO: create method that give you the input shape depending on the dataset
def get_input_shape():
    pass


def make_ConvTTN3d_layer(project_variable, which_layer, k, p):
    # start counting at 1
    assert which_layer > 0

    if which_layer == 1:
        the_in_channels = project_variable.num_in_channels
    else:
        the_in_channels = project_variable.num_out_channels[which_layer - 2]

    # TODO: get rid of the transformation_groups, k0_groups
    the_out_channels = project_variable.num_out_channels[which_layer - 1]
    # the_transformation_groups = project_variable.transformation_groups[which_layer - 1]
    # the_k0_groups = project_variable.k0_groups[which_layer - 1]  # <- correct

    return ConvTTN3d(in_channels=the_in_channels,
                     out_channels=the_out_channels,
                     kernel_size=k,
                     padding=p,
                     project_variable=project_variable
                     # transformation_groups=the_transformation_groups,
                     # k0_groups=the_k0_groups,
                     # transformations_per_filter=project_variable.transformations_per_filter
                     )


def auto_in_features_2(t, h, w, layer_type, k, p, s, div):
    assert layer_type in ['conv', 'pool']

    if layer_type == 'conv':
        t = (t - k + 2 * p) / s + 1
        h = (h - k + 2 * p) / s + 1
        w = (w - k + 2 * p) / s + 1
    elif layer_type == 'pool':
        t = int(np.floor(t / div))
        h = int(np.floor(h / div))
        w = int(np.floor(w / div))

    # print("after %s: t=%d, h=%d, w=%d" % (layer_type, t, h, w))

    return t, h, w


# model_number = 14
class Experimental_TTN3d_xD(torch.nn.Module):
    def __init__(self, project_variable):

        if project_variable.dataset == 'jester':
            t, h, w = 30, 50, 75
        else:
            t, h, w = None, None, None

        self.return_ind = False
        if project_variable.return_ind:
            self.return_ind = True

        super(Experimental_TTN3d_xD, self).__init__()

        self.conv1 = make_ConvTTN3d_layer(project_variable, which_layer=1, k=5, p=1)
        t, h, w = auto_in_features_2(t, h, w, 'conv', k=5, p=1, s=1, div=None)

        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2, return_indices=self.return_ind)
        t, h, w = auto_in_features_2(t, h, w, 'pool', k=None, p=None, s=None, div=2)

        self.conv2 = make_ConvTTN3d_layer(project_variable, which_layer=2, k=5, p=1)
        t, h, w = auto_in_features_2(t, h, w, 'conv', k=5, p=1, s=1, div=None)

        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2, return_indices=self.return_ind)
        t, h, w = auto_in_features_2(t, h, w, 'pool', k=None, p=None, s=None, div=2)

        self.conv3 = make_ConvTTN3d_layer(project_variable, which_layer=3, k=5, p=1)
        t, h, w = auto_in_features_2(t, h, w, 'conv', k=5, p=1, s=1, div=None)

        self.max_pool_3 = torch.nn.MaxPool3d(kernel_size=2, return_indices=self.return_ind)
        t, h, w = auto_in_features_2(t, h, w, 'pool', k=None, p=None, s=None, div=2)

        # out_channels of last conv layer - 1
        in_features = t * h * w * project_variable.num_out_channels[3 - 1]
        self.fc1 = torch.nn.Linear(in_features, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, project_variable.label_size)

    def forward(self, x, device):
        # print('input shape: ', x.shape)
        x = self.conv1(x, device)
        # print('conv1: ', x.shape)

        x = torch.nn.functional.relu(x)

        if self.return_ind:
            x, ind1 = self.max_pool_1(x)
        else:
            x = self.max_pool_1(x)
        # print('mp1: ', x.shape)

        x = self.conv2(x, device)
        # print('conv2: ', x.shape)
        x = torch.nn.functional.relu(x)

        if self.return_ind:
            x, ind2 = self.max_pool_2(x)
        else:
            x = self.max_pool_2(x)
        # print('mp2: ', x.shape)

        x = self.conv3(x, device)
        # print('conv3: ', x.shape)
        x = torch.nn.functional.relu(x)
        
        if self.return_ind:
            x, ind3 = self.max_pool_3(x)
        else:
            x = self.max_pool_3(x)
        # print('mp3: ', x.shape)

        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)

        return x
