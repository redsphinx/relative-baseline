import torch
from torch.nn.modules import conv
from torch.nn import functional as F
from torch.nn.modules.utils import _triple
from torch.nn.functional import conv3d


class ConvTTN3d(conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, project_variable,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        super(ConvTTN3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode)

        self.project_variable = project_variable

        self.first_weight = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros(out_channels, in_channels, 1,
                                                                                 kernel_size[1], kernel_size[2])))

        if self.project_variable.theta_init is not None:
            self.theta = torch.zeros((kernel_size[0] - 1, out_channels, 2, 3))
            if self.project_variable.theta_init == 'eye':
                for i in range(kernel_size[0] - 1):
                    for j in range(out_channels):
                        self.theta[i][j] = torch.eye(3)[:2, ]
            elif self.project_variable.theta_init == 'normal':
                self.theta = torch.nn.init.normal_(self.theta)
            else:
                print("ERROR: theta_init mode '%s' not supported" % self.project_variable.theta_init)
                self.theta = None

            self.theta = torch.nn.Parameter(self.theta)

        else:
            if self.project_variable.srxy_init == 'normal':
            # use 4 parameters
                self.scale = torch.nn.Parameter(
                    torch.abs(torch.nn.init.normal_(torch.zeros((kernel_size[0] - 1, out_channels)))))
                self.rotate = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels))))
                self.translate_x = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels))))
                self.translate_y = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels))))
            elif self.project_variable.srxy_init == 'eye':
                self.scale = torch.nn.Parameter(torch.nn.init.ones_(torch.zeros((kernel_size[0] - 1, out_channels))))
                self.rotate = torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels)))
                self.translate_x = torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels)))
                self.translate_y = torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels)))
            elif self.project_variable.srxy_init == 'eye-like':
                self.scale = torch.nn.Parameter(torch.abs(torch.nn.init.normal_(torch.zeros((kernel_size[0] - 1, out_channels)), mean=1, std=1e-5)))
                self.rotate = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels))), mean=0, std=1e-5)
                self.translate_x = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels))), mean=0, std=1e-5)
                self.translate_y = torch.nn.init.normal_(torch.nn.Parameter(torch.zeros((kernel_size[0] - 1, out_channels))), mean=0, std=1e-5)
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
            matrix[i][1][1] = -scale[i] * torch.cos(rotate[i])
            matrix[i][1][2] = translate_x[i] * scale[i] * torch.sin(rotate[i]) + translate_y[i] * \
                              scale[i] * torch.cos(rotate[i])

        return matrix

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
                                  [self.out_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
            except RuntimeError:
                torch.backends.cudnn.deterministic = True
                print('ok cudnn')

            for i in range(self.kernel_size[0] - 1):
                tmp = F.affine_grid(theta[i],
                                    [self.out_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
                grid = torch.cat((grid, tmp.unsqueeze(0)), 0)

            return grid

        else:
            # cudnn error
            try:
                _ = F.affine_grid(self.theta[0],
                                  [self.out_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
            except RuntimeError:
                torch.backends.cudnn.deterministic = True
                print('ok cudnn')

            for i in range(self.kernel_size[0] - 1):
                tmp = F.affine_grid(self.theta[i],
                                    [self.out_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
                grid = torch.cat((grid, tmp.unsqueeze(0)), 0)

            return grid

    def forward(self, input, device):

        grid = torch.zeros((1, self.out_channels, self.kernel_size[1], self.kernel_size[2], 2))

        if self.project_variable.theta_init is None:
            theta = torch.zeros((1, self.out_channels, 2, 3))
            theta = theta.cuda(device)
        else:
            theta = None

        grid = grid.cuda(device)
        grid = self.update_2(grid, theta, device)[1:]

        # ---
        # needed to deal with the cudnn error
        try:
            _ = F.grid_sample(self.first_weight[:, :, 0], grid[0])
        except RuntimeError:
            torch.backends.cudnn.deterministic = True
            print('ok cudnn')
        # ---

        new_weight = self.first_weight

        if self.project_variable.weight_transform == 'naive':
            for i in range(self.kernel_size[0] - 1):
                tmp = F.grid_sample(self.first_weight[:, :, 0], grid[i])
                new_weight = torch.cat((new_weight, tmp.unsqueeze(2)), 2)
        elif self.project_variable.weight_transform == 'seq':
            for i in range(self.kernel_size[0] - 1):
                tmp = F.grid_sample(new_weight[:,:,-1], grid[i])
                new_weight = torch.cat((new_weight, tmp.unsqueeze(2)), 2)
        else:
            print('ERROR: weight_transform with value %s not supported' % self.project_variable.weight_transform)


        y = F.conv3d(input, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.weight = torch.nn.Parameter(new_weight)
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


class LeNet5_TTN3d(torch.nn.Module):

    def __init__(self, project_variable):
        super(LeNet5_TTN3d, self).__init__()
        self.conv1 = ConvTTN3d(in_channels=1, out_channels=6, kernel_size=5, padding=2,
                               project_variable=project_variable)
        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv2 = ConvTTN3d(in_channels=6, out_channels=16, kernel_size=5, padding=0,
                               project_variable=project_variable)
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
