import numpy as np

from relative_baseline.omg_emotion.convttn3d import ConvTTN3d

import torch
from torch.nn.functional import relu
from torch.nn import MaxPool3d, MaxPool3d, AdaptiveAvgPool3d, AdaptiveAvgPool3d, Conv3d, Conv3d, BatchNorm3d, BatchNorm3d


class ConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pv):
        super(ConvolutionBlock, self).__init__()
        self.conv = ConvTTN3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=7,
                              stride=2,
                              padding=3,
                              project_variable=pv,
                              bias=False)
        self.bn_conv = BatchNorm3d(out_channels)

    def forward(self, x, device):
        h = self.conv(x, device)
        h = self.bn_conv(h)
        y = relu(h)
        return y


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pv):
        super(ResidualBlock, self).__init__()
        self.res_branch2a = ConvTTN3d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      padding=1,
                                      project_variable=pv,
                                      bias=False)
        self.bn_branch2a = BatchNorm3d(out_channels)
        self.res_branch2b = ConvTTN3d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      padding=1,
                                      project_variable=pv,
                                      bias=False)
        self.bn_branch2b = BatchNorm3d(out_channels)

    def forward(self, x, device):
        h = self.res_branch2a(x, device)
        h = self.bn_branch2a(h)
        h = relu(h)
        h = self.res_branch2b(h, device)
        h = self.bn_branch2b(h)
        h = h + x
        y = relu(h)
        return y


class ResidualBlockB(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pv):
        super(ResidualBlockB, self).__init__()
        self.res_branch1 = Conv3d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=2,
                                  bias=False)
        self.bn_branch1 = BatchNorm3d(out_channels)
        self.res_branch2a = ConvTTN3d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      project_variable=pv,
                                      bias=False)
        self.bn_branch2a = BatchNorm3d(out_channels)
        self.res_branch2b = ConvTTN3d(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      padding=1,
                                      project_variable=pv,
                                      bias=False)
        self.bn_branch2b = BatchNorm3d(out_channels)

    def forward(self, x, device):
        temp = self.res_branch1(x)
        temp = self.bn_branch1(temp)

        h = self.res_branch2a(x, device)
        h = self.bn_branch2a(h)
        h = relu(h)
        h = self.res_branch2b(h, device)
        h = self.bn_branch2b(h)
        h = temp + h
        y = relu(h)
        return y


class ResNet18(torch.nn.Module):
    def __init__(self, pv):
        super(ResNet18, self).__init__()
        self.conv1_relu = ConvolutionBlock(3, 64, pv)
        self.maxpool = MaxPool3d(kernel_size=3, padding=1, stride=2, dilation=1)
        self.res2a_relu = ResidualBlock(64, 64, pv)
        self.res2b_relu = ResidualBlock(64, 64, pv)
        self.res3a_relu = ResidualBlockB(64, 128, pv)
        self.res3b_relu = ResidualBlock(128, 128, pv)
        self.res4a_relu = ResidualBlockB(128, 256, pv)
        self.res4b_relu = ResidualBlock(256, 256, pv)
        self.res5a_relu = ResidualBlockB(256, 512, pv)
        self.res5b_relu = ResidualBlock(512, 512, pv)
        self.avgpool = AdaptiveAvgPool3d(output_size=1)
        self.fc = torch.nn.Linear(512, 27)

    def forward(self, x, device):
        print(x.shape)
        h = self.conv1_relu(x, device)
        print(h.shape)
        h = self.maxpool(h)
        print(h.shape)
        h = self.res2a_relu(h, device)
        print(h.shape)
        h = self.res2b_relu(h, device)
        print(h.shape)
        h = self.res3a_relu(h, device)
        print(h.shape)
        h = self.res3b_relu(h, device)
        print(h.shape)
        h = self.res4a_relu(h, device)
        print(h.shape)
        h = self.res4b_relu(h, device)
        print(h.shape)
        h = self.res5a_relu(h, device)
        print(h.shape)
        h = self.res5b_relu(h, device)
        print(h.shape)
        h = self.avgpool(h)
        print(h.shape)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        y = self.fc(h)
        print(h.shape)
        return y