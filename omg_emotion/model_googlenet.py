import numpy as np

from relative_baseline.omg_emotion.convttn3d import ConvTTN3d

import torch
from torch.nn.functional import relu
from torch.nn import MaxPool3d, AdaptiveAvgPool3d, Conv3d, BatchNorm3d, AvgPool3d, Linear, Dropout3d


class Googlenet3TConv_explicit(torch.nn.Module):
    def __init__(self, pv):
        super(Googlenet3TConv_explicit, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2, project_variable=pv, bias=False)
        self.bn1 = BatchNorm3d(64)
        self.maxpool1 = MaxPool3d(kernel_size=3, padding=0, stride=2)
        self.conv2 = Conv3d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn2 = BatchNorm3d(64)
        self.conv3 = ConvTTN3d(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn3 = BatchNorm3d(192)
        self.maxpool2 = MaxPool3d(kernel_size=3, padding=0, stride=2)

        # inception 3a
        self.conv4 = Conv3d(in_channels=192, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn4 = BatchNorm3d(64)
        self.conv5 = Conv3d(in_channels=192, out_channels=96, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn5 = BatchNorm3d(64)
        self.conv6 = ConvTTN3d(in_channels=96, out_channels=128, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn6 = BatchNorm3d(64)
        self.conv7 = Conv3d(in_channels=192, out_channels=16, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn7 = BatchNorm3d(16)
        self.conv8 = ConvTTN3d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn8 = BatchNorm3d(32)
        self.maxpool3 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv9 = Conv3d(in_channels=192, out_channels=32, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn9 = BatchNorm3d(32)

        # inception 3b
        self.conv10 = Conv3d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn10 = BatchNorm3d(128)
        self.conv11 = Conv3d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn11 = BatchNorm3d(128)
        self.conv12 = ConvTTN3d(in_channels=128, out_channels=192, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn12 = BatchNorm3d(192)
        self.conv13 = Conv3d(in_channels=256, out_channels=32, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn13 = BatchNorm3d(32)
        self.conv14 = ConvTTN3d(in_channels=32, out_channels=96, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn14 = BatchNorm3d(96)
        self.maxpool4 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv15 = Conv3d(in_channels=256, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn15 = BatchNorm3d(64)

        self.maxpool5 = MaxPool3d(kernel_size=3, padding=0, stride=2)

        # inception 4a
        self.conv16 = Conv3d(in_channels=480, out_channels=192, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn16 = BatchNorm3d(192)
        self.conv17 = Conv3d(in_channels=480, out_channels=96, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn17 = BatchNorm3d(96)
        self.conv18 = ConvTTN3d(in_channels=96, out_channels=208, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn18 = BatchNorm3d(208)
        self.conv19 = Conv3d(in_channels=480, out_channels=16, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn19 = BatchNorm3d(16)
        self.conv20 = ConvTTN3d(in_channels=16, out_channels=48, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn20 = BatchNorm3d(48)
        self.maxpool6 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv21 = Conv3d(in_channels=480, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn21 = BatchNorm3d(64)

        # inception 4b
        self.conv22 = Conv3d(in_channels=512, out_channels=160, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn22 = BatchNorm3d(160)
        self.conv23 = Conv3d(in_channels=512, out_channels=112, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn23 = BatchNorm3d(112)
        self.conv24 = ConvTTN3d(in_channels=112, out_channels=224, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn24 = BatchNorm3d(224)
        self.conv25 = Conv3d(in_channels=512, out_channels=24, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn25 = BatchNorm3d(24)
        self.conv26 = ConvTTN3d(in_channels=24, out_channels=64, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn26 = BatchNorm3d(64)
        self.maxpool7 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv27 = Conv3d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn27 = BatchNorm3d(128)

        self.avgpool1 = AvgPool3d(kernel_size=5, padding=0, stride=3)
        self.conv28 = Conv3d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn28 = BatchNorm3d(128)
        self.fc1 = Linear(in_features=2048, out_features=1024)
        self.dropout1 = Dropout3d(p=0.7)
        self.fc2 = Linear(in_features=1024, out_features=27)

        # inception 4c
        self.conv29 = Conv3d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn29 = BatchNorm3d(128)
        self.conv30 = Conv3d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn30 = BatchNorm3d(128)
        self.conv31 = ConvTTN3d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn31 = BatchNorm3d(256)
        self.conv32 = Conv3d(in_channels=512, out_channels=24, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn32 = BatchNorm3d(24)
        self.conv33 = ConvTTN3d(in_channels=24, out_channels=64, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn33 = BatchNorm3d(64)
        self.maxpool8 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv34 = Conv3d(in_channels=512, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn34 = BatchNorm3d(64)

        # inception 4d
        self.conv35 = Conv3d(in_channels=512, out_channels=112, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn35 = BatchNorm3d(112)
        self.conv36 = Conv3d(in_channels=512, out_channels=144, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn36 = BatchNorm3d(144)
        self.conv37 = ConvTTN3d(in_channels=144, out_channels=288, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn37 = BatchNorm3d(288)
        self.conv38 = Conv3d(in_channels=512, out_channels=32, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn38 = BatchNorm3d(32)
        self.conv39 = ConvTTN3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn39 = BatchNorm3d(64)
        self.maxpool9 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv40 = Conv3d(in_channels=512, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn40 = BatchNorm3d(64)

        # inception 4e
        self.conv41 = Conv3d(in_channels=528, out_channels=256, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn41 = BatchNorm3d(256)
        self.conv42 = Conv3d(in_channels=528, out_channels=160, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn42 = BatchNorm3d(160)
        self.conv43 = ConvTTN3d(in_channels=160, out_channels=320, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn43 = BatchNorm3d(320)
        self.conv44 = Conv3d(in_channels=528, out_channels=32, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn44 = BatchNorm3d(32)
        self.conv45 = ConvTTN3d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn45 = BatchNorm3d(128)
        self.maxpool10 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv46 = Conv3d(in_channels=528, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn46 = BatchNorm3d(128)

        self.avgpool2 = AvgPool3d(kernel_size=5, padding=0, stride=3)
        self.conv47 = Conv3d(in_channels=528, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn47 = BatchNorm3d(128)
        self.fc3 = Linear(in_features=2048, out_features=1024)
        self.dropout2 = Dropout3d(p=0.7)
        self.fc4 = Linear(in_features=1024, out_features=27)

        self.maxpool11 = MaxPool3d(kernel_size=3, padding=0, stride=2)

        # inception 5a
        self.conv48 = Conv3d(in_channels=832, out_channels=256, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn48 = BatchNorm3d(256)
        self.conv49 = Conv3d(in_channels=832, out_channels=160, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn49 = BatchNorm3d(160)
        self.conv50 = ConvTTN3d(in_channels=160, out_channels=320, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn50 = BatchNorm3d(320)
        self.conv51 = Conv3d(in_channels=832, out_channels=32, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn51 = BatchNorm3d(32)
        self.conv52 = ConvTTN3d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn52 = BatchNorm3d(128)
        self.maxpool12 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv53 = Conv3d(in_channels=832, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn53 = BatchNorm3d(128)

        # inception 5b
        self.conv54 = Conv3d(in_channels=832, out_channels=384, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn54 = BatchNorm3d(384)
        self.conv55 = Conv3d(in_channels=832, out_channels=192, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn55 = BatchNorm3d(192)
        self.conv56 = ConvTTN3d(in_channels=192, out_channels=384, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn56 = BatchNorm3d(384)
        self.conv57 = Conv3d(in_channels=832, out_channels=48, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn57 = BatchNorm3d(48)
        self.conv58 = ConvTTN3d(in_channels=48, out_channels=128, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn58 = BatchNorm3d(128)
        self.maxpool13 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv59 = Conv3d(in_channels=832, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn59 = BatchNorm3d(128)

        self.avgpool3 = AdaptiveAvgPool3d(1)
        self.dropout3 = Dropout3d(p=0.4)
        self.fc5 = Linear(in_features=1024, out_features=27)

    def forward(self, x, device, stop_at=None):

        h = self.conv1(x, device)
        h = self.bn1(h)
        h = relu(h)
        h = self.maxpool1(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = relu(h)

        h = self.conv3(h, device)
        h = self.bn3(h)
        h = relu(h)
        h = self.maxpool2(h)

        # inception 3a


