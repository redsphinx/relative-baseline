import numpy as np
import torch
from torch.nn.modules import conv
from torch.nn import functional as F
from torch.nn.modules.utils import _triple
from torch.nn.functional import conv3d

'''
from factorized_convolution import LeNet5_TTN3d

my_model = LeNet5_TTN3d()
my_model.conv1.weight.requires_grad = False
my_model.conv2.weight.requires_grad = False

run as:     my_model(data, device)
where 'device' is a torch.device
'''


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


        first_w = torch.nn.Parameter(torch.zeros(out_channels, in_channels, 1, kernel_size[1], kernel_size[2]))
        self.first_weight = torch.nn.init.normal_(first_w)


        self.scale = torch.nn.Parameter(
            torch.nn.init.ones_(torch.zeros((kernel_size[0]-1, out_channels))))
        self.rotate = torch.nn.Parameter(
            torch.zeros((kernel_size[0]-1, out_channels)))
        self.translate_x = torch.nn.Parameter(
            torch.zeros((kernel_size[0]-1, out_channels)))
        self.translate_y = torch.nn.Parameter(
            torch.zeros((kernel_size[0]-1, out_channels)))
        

    def make_affine_matrix(self, scale, rotate, translate_x, translate_y):
        # if out_channels is used, the shape of the matrix returned is different

        assert scale.shape == rotate.shape == translate_x.shape == translate_y.shape

        '''
        matrix.shape = (out_channels, 2, 3)
        '''
        matrix = torch.zeros((scale.shape[0], 2, 3))

        matrix[:, 0, 0] = scale[:] * torch.cos(rotate[:])
        matrix[:, 0, 1] = -scale[:] * torch.sin(rotate[:])
        matrix[:, 0, 2] = translate_x[:] * scale[:] * torch.cos(rotate[:]) - translate_y[:] * \
                          scale[:] * torch.sin(rotate[:])
        matrix[:, 1, 0] = scale[:] * torch.sin(rotate[:])
        matrix[:, 1, 1] = scale[:] * torch.cos(rotate[:])
        matrix[:, 1, 2] = translate_x[:] * scale[:] * torch.sin(rotate[:]) + translate_y[:] * \
                          scale[:] * torch.cos(rotate[:])

        return matrix


    def update_grid(self, grid, theta, device):
        # deal with updating s r x y

        for i in range(self.kernel_size[0]-1):
            tmp = self.make_affine_matrix(self.scale[i], self.rotate[i], self.translate_x[i], self.translate_y[i])
            tmp = tmp.cuda(device)
            theta = torch.cat((theta, tmp.unsqueeze(0)), 0)
        theta = theta[1:]
        
        # deal with cudnn error
        try:
            _ = F.affine_grid(theta[0],
                              [self.out_channels, self.kernel_size[0]-1, self.kernel_size[1],
                               self.kernel_size[2]])
        except RuntimeError:
            torch.backends.cudnn.deterministic = True
            print('ok cudnn')
        # ---

        for i in range(self.kernel_size[0] - 1):
            tmp = F.affine_grid(theta[i],
                                [self.out_channels, self.kernel_size[0], self.kernel_size[1],
                                 self.kernel_size[2]])
            grid = torch.cat((grid, tmp.unsqueeze(0)), 0)

        return grid



    def forward(self, input, device):

        theta = torch.zeros((1, self.out_channels, 2, 3))
        theta = theta.cuda(device)

        grid = torch.zeros((1, self.out_channels, self.kernel_size[1], self.kernel_size[2], 2))
        grid = grid.cuda(device)
        grid = self.update_grid(grid, theta, device)
        grid = grid[1:]

        times = 2

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

        for i in range(self.kernel_size[0] - 1):
            tmp = F.grid_sample(new_weight[:, :, -1], grid[i])
            new_weight = torch.cat((new_weight, tmp.unsqueeze(2)), 2)


        y = F.conv3d(input, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y


# practical example using LeNet-5, the 3DTTN version

class LeNet5_TTN3d(torch.nn.Module):
    def __init__(self):
        super(LeNet5_TTN3d, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=1,
                               out_channels=6,
                               kernel_size=5,
                               padding=2)

        self.max_pool_1 = torch.nn.MaxPool3d(kernel_size=2)

        self.conv2 = ConvTTN3d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               padding=0)

        self.max_pool_2 = torch.nn.MaxPool3d(kernel_size=2)

        self.fc1 = torch.nn.Linear(16 * 5 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x, device):
        x = self.conv1(x, device)
        x = torch.nn.functional.relu(x)
        x = self.max_pool_1(x)
        x = self.conv2(x, device)
        x = torch.nn.functional.relu(x)
        x = self.max_pool_2(x)
        _shape = x.shape
        x = x.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return x

