import numpy as np

from relative_baseline.omg_emotion.convttn3d import ConvTTN3d

import torch
from torch.nn.functional import relu
from torch.nn import MaxPool3d, AdaptiveAvgPool3d, Conv3d, BatchNorm3d


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
        h = self.conv1_relu(x, device)
        h = self.maxpool(h)
        h = self.res2a_relu(h, device)
        h = self.res2b_relu(h, device)
        h = self.res3a_relu(h, device)
        h = self.res3b_relu(h, device)
        h = self.res4a_relu(h, device)
        h = self.res4b_relu(h, device)
        h = self.res5a_relu(h, device)
        h = self.res5b_relu(h, device)
        h = self.avgpool(h)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        y = self.fc(h)
        return y


class ResNet18Explicit(torch.nn.Module):
    def __init__(self, pv):
        super(ResNet18Explicit, self).__init__()
        # self.conv1_relu = ConvolutionBlock(3, 64, pv)
        self.conv1 = ConvTTN3d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, project_variable=pv, bias=False)
        self.bn1 = BatchNorm3d(64)
        
        self.maxpool = MaxPool3d(kernel_size=3, padding=1, stride=2, dilation=1)
        
        # self.res2a_relu = ResidualBlock(64, 64, pv)
        self.conv2 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn2 = BatchNorm3d(64)
        self.conv3 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn3 = BatchNorm3d(64)
        
        # self.res2b_relu = ResidualBlock(64, 64, pv)
        self.conv4 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn4 = BatchNorm3d(64)
        self.conv5 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn5 = BatchNorm3d(64)
        
        # self.res3a_relu = ResidualBlockB(64, 128, pv)
        self.conv6 = Conv3d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False)
        self.bn6 = BatchNorm3d(128)
        self.conv7 = ConvTTN3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, project_variable=pv, bias=False)
        self.bn7 = BatchNorm3d(128)
        self.conv8 = ConvTTN3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn8 = BatchNorm3d(128)
        
        # self.res3b_relu = ResidualBlock(128, 128, pv)
        self.conv9 = ConvTTN3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn9 = BatchNorm3d(128)
        self.conv10 = ConvTTN3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn10 = BatchNorm3d(128)
        
        # self.res4a_relu = ResidualBlockB(128, 256, pv)
        self.conv11 = Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False)
        self.bn11 = BatchNorm3d(256)
        self.conv12 = ConvTTN3d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, project_variable=pv, bias=False)
        self.bn12 = BatchNorm3d(256)
        self.conv13 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn13 = BatchNorm3d(256)
        
        # self.res4b_relu = ResidualBlock(256, 256, pv)
        self.conv14 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn14 = BatchNorm3d(256)
        self.conv15 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn15 = BatchNorm3d(256)
        
        # self.res5a_relu = ResidualBlockB(256, 512, pv)
        self.conv16 = Conv3d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False)
        self.bn16 = BatchNorm3d(512)
        self.conv17 = ConvTTN3d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, project_variable=pv, bias=False)
        self.bn17 = BatchNorm3d(512)
        self.conv18 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn18 = BatchNorm3d(512)
        
        # self.res5b_relu = ResidualBlock(512, 512, pv)
        self.conv19 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn19 = BatchNorm3d(512)
        self.conv20 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, project_variable=pv, bias=False)
        self.bn20 = BatchNorm3d(512)
        
        self.avgpool = AdaptiveAvgPool3d(output_size=1)
        self.fc = torch.nn.Linear(512, 27)

    def forward(self, x, device, stop_at=None):
        # h = self.conv1_relu(x, device)

        num = 1
        h = self.conv1(x, device)
        if stop_at == num:
            return h
        h = self.bn1(h)
        h = relu(h)

        h = self.maxpool(h)
        
        # h = self.res2a_relu(h, device)
        num = 2
        h1 = self.conv2(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn2(h1)
        h1 = relu(h1)
        num = 3
        h1 = self.conv3(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn3(h1)
        h = h1 + h
        h = relu(h)
        
        # h = self.res2b_relu(h, device)
        num = 4
        h1 = self.conv4(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn4(h1)
        h1 = relu(h1)
        num = 5
        h1 = self.conv5(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn5(h1)
        h = h1 + h
        h = relu(h)
        
        # h = self.res3a_relu(h, device)
        temp = self.conv6(h)
        temp = self.bn6(temp)
        num = 7
        h1 = self.conv7(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn7(h1)
        h1 = relu(h1)
        num = 8
        h1 = self.conv8(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn8(h1)
        h = temp + h1
        h = relu(h)
        
        # h = self.res3b_relu(h, device)
        num = 9
        h1 = self.conv9(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn9(h1)
        h1 = relu(h1)
        num = 10
        h1 = self.conv10(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn10(h1)
        h = h1 + h
        h = relu(h)
        
        # h = self.res4a_relu(h, device)
        temp = self.conv11(h)
        temp = self.bn11(temp)
        num = 12
        h1 = self.conv12(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn12(h1)
        h1 = relu(h1)
        num = 13
        h1 = self.conv13(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn13(h1)
        h = temp + h1
        h = relu(h)
        
        # h = self.res4b_relu(h, device)
        num = 14
        h1 = self.conv14(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn14(h1)
        h1 = relu(h1)
        num = 15
        h1 = self.conv15(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn15(h1)
        h = h1 + h
        h = relu(h)
        
        # h = self.res5a_relu(h, device)
        temp = self.conv16(h)
        temp = self.bn16(temp)
        num = 17
        h1 = self.conv17(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn17(h1)
        h1 = relu(h1)
        num = 18
        h1 = self.conv18(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn18(h1)
        h = temp + h1
        h = relu(h)
        
        # h = self.res5b_relu(h, device)
        num = 19
        h1 = self.conv19(h, device)
        if stop_at == num:
            return h1
        h1 = self.bn19(h1)
        h1 = relu(h1)
        num = 20
        h1 = self.conv20(h1, device)
        if stop_at == num:
            return h1
        h1 = self.bn20(h1)
        h = h1 + h
        h = relu(h)
        
        h = self.avgpool(h)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        y = self.fc(h)
        return y

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# resnet18 3DConv
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

class ResNet18Explicit3DConv(torch.nn.Module):
    def __init__(self):
        super(ResNet18Explicit3DConv, self).__init__()
        # self.conv1_relu = ConvolutionBlock(3, 64, pv)
        self.conv1 = Conv3d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm3d(64)

        self.maxpool = MaxPool3d(kernel_size=3, padding=1, stride=2, dilation=1)

        # self.res2a_relu = ResidualBlock(64, 64, pv)
        self.conv2 = Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm3d(64)
        self.conv3 = Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn3 = BatchNorm3d(64)

        # self.res2b_relu = ResidualBlock(64, 64, pv)
        self.conv4 = Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn4 = BatchNorm3d(64)
        self.conv5 = Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn5 = BatchNorm3d(64)

        # self.res3a_relu = ResidualBlockB(64, 128, pv)
        self.conv6 = Conv3d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False)
        self.bn6 = BatchNorm3d(128)
        self.conv7 = Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = BatchNorm3d(128)
        self.conv8 = Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn8 = BatchNorm3d(128)

        # self.res3b_relu = ResidualBlock(128, 128, pv)
        self.conv9 = Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn9 = BatchNorm3d(128)
        self.conv10 = Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn10 = BatchNorm3d(128)

        # self.res4a_relu = ResidualBlockB(128, 256, pv)
        self.conv11 = Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False)
        self.bn11 = BatchNorm3d(256)
        self.conv12 = Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn12 = BatchNorm3d(256)
        self.conv13 = Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False)
        self.bn13 = BatchNorm3d(256)

        # self.res4b_relu = ResidualBlock(256, 256, pv)
        self.conv14 = Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False)
        self.bn14 = BatchNorm3d(256)
        self.conv15 = Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False)
        self.bn15 = BatchNorm3d(256)

        # self.res5a_relu = ResidualBlockB(256, 512, pv)
        self.conv16 = Conv3d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False)
        self.bn16 = BatchNorm3d(512)
        self.conv17 = Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn17 = BatchNorm3d(512)
        self.conv18 = Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.bn18 = BatchNorm3d(512)

        # self.res5b_relu = ResidualBlock(512, 512, pv)
        self.conv19 = Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.bn19 = BatchNorm3d(512)
        self.conv20 = Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False)
        self.bn20 = BatchNorm3d(512)

        self.avgpool = AdaptiveAvgPool3d(output_size=1)
        self.fc = torch.nn.Linear(512, 27)


    def forward(self, x, stop_at=None):
        # h = self.conv1_relu(x, device)

        num = 1
        h = self.conv1(x)
        if stop_at == num:
            return h
        h = self.bn1(h)
        h = relu(h)

        h = self.maxpool(h)

        # h = self.res2a_relu(h)
        num = 2
        h1 = self.conv2(h)
        if stop_at == num:
            return h1
        h1 = self.bn2(h1)
        h1 = relu(h1)
        num = 3
        h1 = self.conv3(h1)
        if stop_at == num:
            return h1
        h1 = self.bn3(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res2b_relu(h)
        num = 4
        h1 = self.conv4(h)
        if stop_at == num:
            return h1
        h1 = self.bn4(h1)
        h1 = relu(h1)
        num = 5
        h1 = self.conv5(h1)
        if stop_at == num:
            return h1
        h1 = self.bn5(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res3a_relu(h)
        temp = self.conv6(h)
        temp = self.bn6(temp)
        num = 7
        h1 = self.conv7(h)
        if stop_at == num:
            return h1
        h1 = self.bn7(h1)
        h1 = relu(h1)
        num = 8
        h1 = self.conv8(h1)
        if stop_at == num:
            return h1
        h1 = self.bn8(h1)
        h = temp + h1
        h = relu(h)

        # h = self.res3b_relu(h)
        num = 9
        h1 = self.conv9(h)
        if stop_at == num:
            return h1
        h1 = self.bn9(h1)
        h1 = relu(h1)
        num = 10
        h1 = self.conv10(h1)
        if stop_at == num:
            return h1
        h1 = self.bn10(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res4a_relu(h)
        temp = self.conv11(h)
        temp = self.bn11(temp)
        num = 12
        h1 = self.conv12(h)
        if stop_at == num:
            return h1
        h1 = self.bn12(h1)
        h1 = relu(h1)
        num = 13
        h1 = self.conv13(h1)
        if stop_at == num:
            return h1
        h1 = self.bn13(h1)
        h = temp + h1
        h = relu(h)

        # h = self.res4b_relu(h)
        num = 14
        h1 = self.conv14(h)
        if stop_at == num:
            return h1
        h1 = self.bn14(h1)
        h1 = relu(h1)
        num = 15
        h1 = self.conv15(h1)
        if stop_at == num:
            return h1
        h1 = self.bn15(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res5a_relu(h)
        temp = self.conv16(h)
        temp = self.bn16(temp)
        num = 17
        h1 = self.conv17(h)
        if stop_at == num:
            return h1
        h1 = self.bn17(h1)
        h1 = relu(h1)
        num = 18
        h1 = self.conv18(h1)
        if stop_at == num:
            return h1
        h1 = self.bn18(h1)
        h = temp + h1
        h = relu(h)

        # h = self.res5b_relu(h)
        num = 19
        h1 = self.conv19(h)
        if stop_at == num:
            return h1
        h1 = self.bn19(h1)
        h1 = relu(h1)
        num = 20
        h1 = self.conv20(h1)
        if stop_at == num:
            return h1
        h1 = self.bn20(h1)
        h = h1 + h
        h = relu(h)

        h = self.avgpool(h)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        y = self.fc(h)
        return y


class ResNet18Explicit3DConvReduced(torch.nn.Module):
    def __init__(self):
        super(ResNet18Explicit3DConvReduced, self).__init__()
        # self.conv1_relu = ConvolutionBlock(3, 64, pv)
        self.conv1 = Conv3d(in_channels=3, out_channels=int(64/1.718), kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm3d(int(64/1.718))

        self.maxpool = MaxPool3d(kernel_size=3, padding=1, stride=2, dilation=1)

        # self.res2a_relu = ResidualBlock(int(64/1.718), int(64/1.718), pv)
        self.conv2 = Conv3d(in_channels=int(64/1.718), out_channels=int(64/1.718), kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm3d(int(64/1.718))
        self.conv3 = Conv3d(in_channels=int(64/1.718), out_channels=int(64/1.718), kernel_size=3, padding=1, bias=False)
        self.bn3 = BatchNorm3d(int(64/1.718))

        # self.res2b_relu = ResidualBlock(int(64/1.718), int(64/1.718), pv)
        self.conv4 = Conv3d(in_channels=int(64/1.718), out_channels=int(64/1.718), kernel_size=3, padding=1, bias=False)
        self.bn4 = BatchNorm3d(int(64/1.718))
        self.conv5 = Conv3d(in_channels=int(64/1.718), out_channels=int(64/1.718), kernel_size=3, padding=1, bias=False)
        self.bn5 = BatchNorm3d(int(64/1.718))

        # self.res3a_relu = ResidualBlockB(int(64/1.718), int(128/1.718), pv)
        self.conv6 = Conv3d(in_channels=int(64/1.718), out_channels=int(128/1.718), kernel_size=1, stride=2, bias=False)
        self.bn6 = BatchNorm3d(int(128/1.718))
        self.conv7 = Conv3d(in_channels=int(64/1.718), out_channels=int(128/1.718), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = BatchNorm3d(int(128/1.718))
        self.conv8 = Conv3d(in_channels=int(128/1.718), out_channels=int(128/1.718), kernel_size=3, padding=1, bias=False)
        self.bn8 = BatchNorm3d(int(128/1.718))

        # self.res3b_relu = ResidualBlock(int(128/1.718), int(128/1.718), pv)
        self.conv9 = Conv3d(in_channels=int(128/1.718), out_channels=int(128/1.718), kernel_size=3, padding=1, bias=False)
        self.bn9 = BatchNorm3d(int(128/1.718))
        self.conv10 = Conv3d(in_channels=int(128/1.718), out_channels=int(128/1.718), kernel_size=3, padding=1, bias=False)
        self.bn10 = BatchNorm3d(int(128/1.718))

        # self.res4a_relu = ResidualBlockB(int(128/1.718), int(256/1.718), pv)
        self.conv11 = Conv3d(in_channels=int(128/1.718), out_channels=int(256/1.718), kernel_size=1, stride=2, bias=False)
        self.bn11 = BatchNorm3d(int(256/1.718))
        self.conv12 = Conv3d(in_channels=int(128/1.718), out_channels=int(256/1.718), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn12 = BatchNorm3d(int(256/1.718))
        self.conv13 = Conv3d(in_channels=int(256/1.718), out_channels=int(256/1.718), kernel_size=3, padding=1, bias=False)
        self.bn13 = BatchNorm3d(int(256/1.718))

        # self.res4b_relu = ResidualBlock(int(256/1.718), int(256/1.718), pv)
        self.conv14 = Conv3d(in_channels=int(256/1.718), out_channels=int(256/1.718), kernel_size=3, padding=1, bias=False)
        self.bn14 = BatchNorm3d(int(256/1.718))
        self.conv15 = Conv3d(in_channels=int(256/1.718), out_channels=int(256/1.718), kernel_size=3, padding=1, bias=False)
        self.bn15 = BatchNorm3d(int(256/1.718))

        # self.res5a_relu = ResidualBlockB(int(256/1.718), int(512/1.718), pv)
        self.conv16 = Conv3d(in_channels=int(256/1.718), out_channels=int(512/1.718), kernel_size=1, stride=2, bias=False)
        self.bn16 = BatchNorm3d(int(512/1.718))
        self.conv17 = Conv3d(in_channels=int(256/1.718), out_channels=int(512/1.718), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn17 = BatchNorm3d(int(512/1.718))
        self.conv18 = Conv3d(in_channels=int(512/1.718), out_channels=int(512/1.718), kernel_size=3, padding=1, bias=False)
        self.bn18 = BatchNorm3d(int(512/1.718))

        # self.res5b_relu = ResidualBlock(int(512/1.718), int(512/1.718), pv)
        self.conv19 = Conv3d(in_channels=int(512/1.718), out_channels=int(512/1.718), kernel_size=3, padding=1, bias=False)
        self.bn19 = BatchNorm3d(int(512/1.718))
        self.conv20 = Conv3d(in_channels=int(512/1.718), out_channels=int(512/1.718), kernel_size=3, padding=1, bias=False)
        self.bn20 = BatchNorm3d(int(512/1.718))

        self.avgpool = AdaptiveAvgPool3d(output_size=1)
        self.fc = torch.nn.Linear(int(512/1.718), 27)


    def forward(self, x, stop_at=None):
        # h = self.conv1_relu(x, device)

        num = 1
        h = self.conv1(x)
        if stop_at == num:
            return h
        h = self.bn1(h)
        h = relu(h)

        h = self.maxpool(h)

        # h = self.res2a_relu(h)
        num = 2
        h1 = self.conv2(h)
        if stop_at == num:
            return h1
        h1 = self.bn2(h1)
        h1 = relu(h1)
        num = 3
        h1 = self.conv3(h1)
        if stop_at == num:
            return h1
        h1 = self.bn3(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res2b_relu(h)
        num = 4
        h1 = self.conv4(h)
        if stop_at == num:
            return h1
        h1 = self.bn4(h1)
        h1 = relu(h1)
        num = 5
        h1 = self.conv5(h1)
        if stop_at == num:
            return h1
        h1 = self.bn5(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res3a_relu(h)
        temp = self.conv6(h)
        temp = self.bn6(temp)
        num = 7
        h1 = self.conv7(h)
        if stop_at == num:
            return h1
        h1 = self.bn7(h1)
        h1 = relu(h1)
        num = 8
        h1 = self.conv8(h1)
        if stop_at == num:
            return h1
        h1 = self.bn8(h1)
        h = temp + h1
        h = relu(h)

        # h = self.res3b_relu(h)
        num = 9
        h1 = self.conv9(h)
        if stop_at == num:
            return h1
        h1 = self.bn9(h1)
        h1 = relu(h1)
        num = 10
        h1 = self.conv10(h1)
        if stop_at == num:
            return h1
        h1 = self.bn10(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res4a_relu(h)
        temp = self.conv11(h)
        temp = self.bn11(temp)
        num = 12
        h1 = self.conv12(h)
        if stop_at == num:
            return h1
        h1 = self.bn12(h1)
        h1 = relu(h1)
        num = 13
        h1 = self.conv13(h1)
        if stop_at == num:
            return h1
        h1 = self.bn13(h1)
        h = temp + h1
        h = relu(h)

        # h = self.res4b_relu(h)
        num = 14
        h1 = self.conv14(h)
        if stop_at == num:
            return h1
        h1 = self.bn14(h1)
        h1 = relu(h1)
        num = 15
        h1 = self.conv15(h1)
        if stop_at == num:
            return h1
        h1 = self.bn15(h1)
        h = h1 + h
        h = relu(h)

        # h = self.res5a_relu(h)
        temp = self.conv16(h)
        temp = self.bn16(temp)
        num = 17
        h1 = self.conv17(h)
        if stop_at == num:
            return h1
        h1 = self.bn17(h1)
        h1 = relu(h1)
        num = 18
        h1 = self.conv18(h1)
        if stop_at == num:
            return h1
        h1 = self.bn18(h1)
        h = temp + h1
        h = relu(h)

        # h = self.res5b_relu(h)
        num = 19
        h1 = self.conv19(h)
        if stop_at == num:
            return h1
        h1 = self.bn19(h1)
        h1 = relu(h1)
        num = 20
        h1 = self.conv20(h1)
        if stop_at == num:
            return h1
        h1 = self.bn20(h1)
        h = h1 + h
        h = relu(h)

        h = self.avgpool(h)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        y = self.fc(h)
        return y