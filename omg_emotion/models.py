import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules import conv
from torch.nn import functional as F
from torch.nn.modules.utils import _triple
from torch.nn.functional import conv3d


class ConvTTN3d(conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, project_variable, transformation_groups, k0_groups,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super(ConvTTN3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode)

        self.project_variable = project_variable

        assert (0 < transformation_groups <= out_channels)
        self.transformation_groups = transformation_groups

        assert (0 < k0_groups <= out_channels)
        self.k0_groups = k0_groups

        # replaced out_channels with k0_groups
        first_w = torch.nn.Parameter(torch.zeros(self.k0_groups, in_channels, 1, kernel_size[1], kernel_size[2]))

        # when transfer learning this will be an issue? set k0_init to None when transfer learning
        if project_variable.k0_init == 'normal':
            self.first_weight = torch.nn.init.normal_(first_w)
        elif project_variable.k0_init == 'ones':
            self.first_weight = torch.nn.init.constant(first_w, 1)
        elif project_variable.k0_init == 'ones_var':
            self.first_weight = torch.nn.init.normal_(first_w, mean=1., std=0.5)
        elif project_variable.k0_init == 'uniform':
            self.first_weight = torch.nn.init.uniform(first_w)

        if self.project_variable.theta_init is not None:
            # self.theta = torch.zeros((kernel_size[0] - 1, out_channels, 2, 3))
            self.theta = torch.zeros((kernel_size[0] - 1, self.transformation_groups, 2, 3))
            if self.project_variable.theta_init == 'eye':
                for i in range(kernel_size[0] - 1):
                    # for j in range(out_channels):
                    for j in range(self.transformation_groups):
                        self.theta[i][j] = torch.eye(3)[:2, ]
            elif self.project_variable.theta_init == 'normal':
                self.theta = torch.nn.init.normal_(self.theta)
            else:
                print("ERROR: theta_init mode '%s' not supported" % self.project_variable.theta_init)
                self.theta = None

            self.theta = torch.nn.Parameter(self.theta)
        
        # replaced 'out_channels' with 'transformation_groups'
        else:
            if self.project_variable.srxy_init == 'normal':
            # use 4 parameters
                self.scale = torch.nn.Parameter(
                    torch.abs(torch.nn.init.normal_(torch.zeros((kernel_size[0] - 1, self.transformation_groups)))))
                self.rotate = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, self.transformation_groups))))
                self.translate_x = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, self.transformation_groups))))
                self.translate_y = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, self.transformation_groups))))
            elif self.project_variable.srxy_init == 'eye':
                self.scale = torch.nn.Parameter(torch.nn.init.ones_(torch.zeros((kernel_size[0] - 1, self.transformation_groups))))
                self.rotate = torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, self.transformation_groups)))
                self.translate_x = torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, self.transformation_groups)))
                self.translate_y = torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, self.transformation_groups)))
            elif self.project_variable.srxy_init == 'eye-like':
                self.scale = torch.nn.Parameter(torch.abs(torch.nn.init.normal_(torch.zeros((kernel_size[0] - 1, self.transformation_groups)), mean=1, std=1e-5)))
                self.rotate = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, self.transformation_groups))), mean=0, std=1e-5)
                self.translate_x = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, self.transformation_groups))), mean=0, std=1e-5)
                self.translate_y = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, self.transformation_groups))), mean=0, std=1e-5)
            else:
                print("ERROR: srxy_init mode '%s' not supported" % self.project_variable.srxy_init)
                self.scale, self.rotate, self.translate_x, self.translate_y = None, None, None, None


    def make_affine_matrix(self, scale, rotate, translate_x, translate_y):
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
            matrix[i][1][1] = scale[i] * torch.cos(rotate[i])
            matrix[i][1][2] = translate_x[i] * scale[i] * torch.sin(rotate[i]) + translate_y[i] * \
                              scale[i] * torch.cos(rotate[i])

        return matrix


    # replace out_channels with self.transformation_groups
    def update_2(self, grid, theta, device):
        # deal with updating s r x y

        if theta is not None:

            for i in range(self.kernel_size[0] - 1):
                tmp = self.make_affine_matrix(self.scale[i], self.rotate[i], self.translate_x[i], self.translate_y[i])
                tmp = tmp.cuda(device)
                theta = torch.cat((theta, tmp.unsqueeze(0)), 0)
            theta = theta[1:]
        
            try:
                _ = F.affine_grid(theta[0],
                                  [self.transformation_groups, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
            except RuntimeError:
                torch.backends.cudnn.deterministic = True
                print('ok cudnn')

            for i in range(self.kernel_size[0] - 1):
                tmp = F.affine_grid(theta[i],
                                    [self.transformation_groups, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
                grid = torch.cat((grid, tmp.unsqueeze(0)), 0)

            return grid

        else:
            # cudnn error
            try:
                _ = F.affine_grid(self.theta[0],
                                  [self.transformation_groups, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
            except RuntimeError:
                torch.backends.cudnn.deterministic = True
                print('ok cudnn')

            for i in range(self.kernel_size[0] - 1):
                tmp = F.affine_grid(self.theta[i],
                                    [self.transformation_groups, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
                grid = torch.cat((grid, tmp.unsqueeze(0)), 0)

            return grid

    # replace out_channels with transformation_groups
    def forward(self, input, device):

        grid = torch.zeros((1, self.transformation_groups, self.kernel_size[1], self.kernel_size[2], 2))

        if self.project_variable.theta_init is None:
            # add smoothness constraint for SRXY
            if self.project_variable.srxy_smoothness == 'sigmoid':
                self.scale.data = (2 - 0) * torch.nn.functional.sigmoid(-self.scale) + 0  # between 0 and 2
                self.rotate.data = (1 - -1) * torch.nn.functional.sigmoid(-self.rotate) - 1  # between -1 and 1
                self.translate_x.data = (1 - -1) * torch.nn.functional.sigmoid(-self.translate_x) - 1  # between -1 and 1
                self.translate_y.data = (1 - -1) * torch.nn.functional.sigmoid(-self.translate_y) - 1  # between -1 and 1
            elif self.project_variable.srxy_smoothness == 'sigmoid_small':
                self.scale.data = (1.1 - 0.9) * torch.nn.functional.sigmoid(-self.scale) + 0.9  # between 0.9 and 1.1
                self.rotate.data = (0.5 - -0.5) * torch.nn.functional.sigmoid(-self.rotate) - 0.5  # between -0.5 and 0.5
                self.translate_x.data = (0.25 - -0.25) * torch.nn.functional.sigmoid(-self.translate_x) - 0.25  # between -0.25 and 0.25
                self.translate_y.data = (0.25 - -0.25) * torch.nn.functional.sigmoid(-self.translate_y) - 0.25  # between -0.25 and 0.25

            theta = torch.zeros((1, self.transformation_groups, 2, 3))
            theta = theta.cuda(device)
        else:
            theta = None

        grid = grid.cuda(device)
        grid = self.update_2(grid, theta, device)[1:]

        # check if shape of grid is compatible with out_channels. if not, correct it
        if self.out_channels != grid.shape[1]:
            if self.out_channels % self.transformation_groups == 0:
                grid = grid.repeat_interleave(self.out_channels//self.transformation_groups, 1)
            else:
                # print('working on it')
                grid = grid.repeat_interleave(self.out_channels//self.transformation_groups+1, 1)
                grid = grid[:, :self.out_channels, :, :, :]

        # check if first_weight is compatible with out_channels. if not, correct it
        if self.out_channels % self.k0_groups == 0:
            times = self.out_channels // self.k0_groups
        else:
            times = self.out_channels // self.k0_groups + 1

        # ---
        # needed to deal with the cudnn error
        try:
            _ = F.grid_sample(self.first_weight[:, :, 0].repeat_interleave(times, 0)[:self.out_channels], grid[0])
            # _ = F.grid_sample(self.first_weight[:, :, 0], grid[0])
        except RuntimeError:
            torch.backends.cudnn.deterministic = True
            print('ok cudnn')
        # ---

        new_weight = self.first_weight.repeat_interleave(times, 0)[:self.out_channels]

        if self.project_variable.weight_transform == 'naive':
            for i in range(self.kernel_size[0] - 1):
                tmp = F.grid_sample(self.first_weight[:, :, 0].repeat_interleave(times, 0)[:self.out_channels], grid[i])
                new_weight = torch.cat((new_weight, tmp.unsqueeze(2)), 2)
        elif self.project_variable.weight_transform == 'seq':
            for i in range(self.kernel_size[0] - 1):
                tmp = F.grid_sample(new_weight[:,:,-1], grid[i])
                new_weight = torch.cat((new_weight, tmp.unsqueeze(2)), 2)
        else:
            print('ERROR: weight_transform with value %s not supported' % self.project_variable.weight_transform)


        y = F.conv3d(input, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # self.weight = torch.nn.Parameter(new_weight) # TODO: why is this here??
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

    def __init__(self, project_variable):
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
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv3d(in_channels=project_variable.num_out_channels[0], out_channels=project_variable.num_out_channels[1], kernel_size=project_variable.k_shape, stride=1, padding=0, bias=True)
        if project_variable.k0_init == 'normal':
            self.conv2.weight = torch.nn.init.normal_(self.conv2.weight)
            self.conv2.bias = torch.nn.init.normal_(self.conv2.bias)
        elif project_variable.k0_init == 'ones':
            self.conv2.weight = torch.nn.init.constant_(self.conv2.weight, 1)
            self.conv2.bias = torch.nn.init.constant_(self.conv2.bias, 1)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2)

        # FIX: automatically determine the required size for fc1
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

        self.fc1 = torch.nn.Linear(project_variable.num_out_channels[1] * _fc_in[0] * _fc_in[1] * _fc_in[2],
                                   120)  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)  # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, project_variable.label_size)  # convert matrix with 84 features to a matrix of 10 features (columns)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class LeNet5_TTN3d(torch.nn.Module):

    def __init__(self, project_variable):
        super(LeNet5_TTN3d, self).__init__()
        self.conv1 = ConvTTN3d(in_channels=1, out_channels=project_variable.num_out_channels[0], kernel_size=5, padding=2,
                               project_variable=project_variable, 
                               transformation_groups=project_variable.transformation_groups[0],
                               k0_groups=project_variable.k0_groups[0])
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv2 = ConvTTN3d(in_channels=project_variable.num_out_channels[0], out_channels=project_variable.num_out_channels[1], kernel_size=5, padding=0,
                               project_variable=project_variable, 
                               transformation_groups=project_variable.transformation_groups[1],
                               k0_groups=project_variable.k0_groups[1])
        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2)

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


class AlexNet_classic(nn.Module):

    def __init__(self, num_classes=6):
        super(AlexNet_classic, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    

class AlexNet_2d(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet_2d, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        # self. = nn.ReLU(inplace=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        # self. = nn.ReLU(inplace=True)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        # self. = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        # self. = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self. = nn.ReLU(inplace=True)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # TODO: finish alexnet implementation, do the RELUs
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


# adapted from https://arxiv.org/abs/1907.11272
class Sota_3d(torch.nn.Module):
    def __init__(self, input_shape):
        t, h, w = input_shape
        super(Sota_3d, self).__init__()
        self.conv_1 = torch.nn.Conv3d(in_channels=1,
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

        self.conv_2 = torch.nn.Conv3d(in_channels=16,
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
        self.fc_1 = torch.nn.Linear(in_features=in_features,
                                    out_features=128)
        self.drop_1 = torch.nn.Dropout3d(p=0.5)
        self.fc_2 = torch.nn.Linear(in_features=128,
                                    out_features=6)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.prelu_1(x)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.prelu_2(x)
        x = self.max_pool_2(x)
        _shape = x.shape
        x = x.view(-1, _shape[1]*_shape[2]*_shape[3]*_shape[4])
        x = self.fc_1(x)
        x = self.drop_1(x)
        x = self.fc_2(x)

        return x






















