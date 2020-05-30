import numpy as np

from relative_baseline.omg_emotion.convttn3d import ConvTTN3d
from relative_baseline.omg_emotion.convttn3d_dynamic import ConvTTN3d_dynamic

import torch
from torch.nn.functional import relu
from torch.nn import MaxPool3d, AdaptiveAvgPool3d, Conv3d, BatchNorm3d, AvgPool3d, Linear, Dropout3d


class Googlenet3TConv_explicit(torch.nn.Module):
    def __init__(self, pv):
        super(Googlenet3TConv_explicit, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2, project_variable=pv, bias=False)
        self.bn1 = BatchNorm3d(64)
        self.maxpool1 = MaxPool3d(kernel_size=(1, 3, 3), padding=0, stride=(1, 2, 2))
        self.conv2 = Conv3d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn2 = BatchNorm3d(64)
        self.conv3 = ConvTTN3d(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn3 = BatchNorm3d(192)
        self.maxpool2 = MaxPool3d(kernel_size=(1, 3, 3), padding=0, stride=(1, 2, 2))

        # inception 3a
        self.conv4 = Conv3d(in_channels=192, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn4 = BatchNorm3d(64)
        self.conv5 = Conv3d(in_channels=192, out_channels=96, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn5 = BatchNorm3d(96)
        self.conv6 = ConvTTN3d(in_channels=96, out_channels=128, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn6 = BatchNorm3d(128)
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
        self.conv27 = Conv3d(in_channels=512, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn27 = BatchNorm3d(64)

        self.avgpool1 = AvgPool3d(kernel_size=5, padding=0, stride=3)
        self.conv28 = Conv3d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn28 = BatchNorm3d(128)
        # self.fc1 = Linear(in_features=2304, out_features=1024)
        self.fc1 = Linear(in_features=768, out_features=1024)  # 768
        self.dropout1 = Dropout3d(p=0.7)
        self.fc2 = Linear(in_features=1024, out_features=pv.label_size)

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
        # self.fc3 = Linear(in_features=2304, out_features=1024)
        self.fc3 = Linear(in_features=768, out_features=1024)
        self.dropout2 = Dropout3d(p=0.7)
        self.fc4 = Linear(in_features=1024, out_features=pv.label_size)

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
        self.fc5 = Linear(in_features=1024, out_features=pv.label_size)

    def forward(self, x, device, stop_at=None, aux=True):
        # set aux=False during inference

        num = 1
        h = self.conv1(x, device)
        if stop_at == num:
            return h
        h = self.bn1(h)
        h = relu(h)
        h = self.maxpool1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = relu(h)
        num = 3
        h = self.conv3(h, device)
        if stop_at == num:
            return h
        h = self.bn3(h)
        h = relu(h)
        h = self.maxpool2(h)

        # inception 3a
        h1 = self.conv4(h)          # branch 1
        h1 = self.bn4(h1)
        h2 = self.conv5(h)          # branch 2.0
        h2 = self.bn5(h2)
        h2 = relu(h2)
        num = 6
        h2 = self.conv6(h2, device) # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn6(h2)
        h3 = self.conv7(h)          # branch 3.0
        h3 = self.bn7(h3)
        h3 = relu(h3)
        num = 8
        h3 = self.conv8(h3, device) # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn8(h3)
        h4 = self.maxpool3(h)       # branch 4.0
        h4 = self.conv9(h4)         # branch 4.1
        h4 = self.bn9(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 3b
        h1 = self.conv10(h)  # branch 1
        h1 = self.bn10(h1)
        h2 = self.conv11(h)  # branch 2.0
        h2 = self.bn11(h2)
        h2 = relu(h2)
        num = 12
        h2 = self.conv12(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn12(h2)
        h3 = self.conv13(h)  # branch 3.0
        h3 = self.bn13(h3)
        h3 = relu(h3)
        num = 14
        h3 = self.conv14(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn14(h3)
        h4 = self.maxpool4(h)  # branch 4.0
        h4 = self.conv15(h4)  # branch 4.1
        h4 = self.bn15(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)
        h = self.maxpool5(h)

        # inception 4a
        h1 = self.conv16(h)  # branch 1
        h1 = self.bn16(h1)
        h2 = self.conv17(h)  # branch 2.0
        h2 = self.bn17(h2)
        h2 = relu(h2)
        num = 18
        h2 = self.conv18(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn18(h2)
        h3 = self.conv19(h)  # branch 3.0
        h3 = self.bn19(h3)
        h3 = relu(h3)
        num = 20
        h3 = self.conv20(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn20(h3)
        h4 = self.maxpool6(h)  # branch 4.0
        h4 = self.conv21(h4)  # branch 4.1
        h4 = self.bn21(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 4b
        h1 = self.conv22(h)  # branch 1
        h1 = self.bn22(h1)
        h2 = self.conv23(h)  # branch 2.0
        h2 = self.bn23(h2)
        h2 = relu(h2)
        num = 24
        h2 = self.conv24(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn24(h2)
        h3 = self.conv25(h)  # branch 3.0
        h3 = self.bn25(h3)
        h3 = relu(h3)
        num = 26
        h3 = self.conv26(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn26(h3)
        h4 = self.maxpool7(h)  # branch 4.0
        h4 = self.conv27(h4)  # branch 4.1
        h4 = self.bn27(h4)

        if aux:
            aux1 = self.avgpool1(h)
            aux1 = self.conv28(aux1)
            aux1 = self.bn28(aux1)
            aux1 = relu(aux1)
            _shape = aux1.shape
            aux1 = aux1.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
            aux1 = self.fc1(aux1)
            aux1 = relu(aux1)
            aux1 = self.dropout1(aux1)
            aux1 = self.fc2(aux1)
        else:
            aux1 = None

        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 4c
        h1 = self.conv29(h)  # branch 1
        h1 = self.bn29(h1)
        h2 = self.conv30(h)  # branch 2.0
        h2 = self.bn30(h2)
        h2 = relu(h2)
        num = 31
        h2 = self.conv31(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn31(h2)
        h3 = self.conv32(h)  # branch 3.0
        h3 = self.bn32(h3)
        h3 = relu(h3)
        num = 33
        h3 = self.conv33(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn33(h3)
        h4 = self.maxpool8(h)  # branch 4.0
        h4 = self.conv34(h4)  # branch 4.1
        h4 = self.bn34(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 4d
        h1 = self.conv35(h)  # branch 1
        h1 = self.bn35(h1)
        h2 = self.conv36(h)  # branch 2.0
        h2 = self.bn36(h2)
        h2 = relu(h2)
        num = 37
        h2 = self.conv37(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn37(h2)
        h3 = self.conv38(h)  # branch 3.0
        h3 = self.bn38(h3)
        h3 = relu(h3)
        num = 39
        h3 = self.conv39(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn39(h3)
        h4 = self.maxpool9(h)  # branch 4.0
        h4 = self.conv40(h4)  # branch 4.1
        h4 = self.bn40(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 4e
        h1 = self.conv41(h)  # branch 1
        h1 = self.bn41(h1)
        h2 = self.conv42(h)  # branch 2.0
        h2 = self.bn42(h2)
        h2 = relu(h2)
        num = 43
        h2 = self.conv43(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn43(h2)
        h3 = self.conv44(h)  # branch 3.0
        h3 = self.bn44(h3)
        h3 = relu(h3)
        num = 45
        h3 = self.conv45(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn45(h3)
        h4 = self.maxpool10(h)  # branch 4.0
        h4 = self.conv46(h4)  # branch 4.1
        h4 = self.bn46(h4)

        if aux:
            aux2 = self.avgpool2(h)
            aux2 = self.conv47(aux2)
            aux2 = self.bn47(aux2)
            aux2 = relu(aux2)
            _shape = aux2.shape
            aux2 = aux2.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
            aux2 = self.fc3(aux2)
            aux2 = relu(aux2)
            aux2 = self.dropout2(aux2)
            aux2 = self.fc4(aux2)
        else:
            aux2 = None

        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)
        h = self.maxpool11(h)

        # inception 5a
        h1 = self.conv48(h)  # branch 1
        h1 = self.bn48(h1)
        h2 = self.conv49(h)  # branch 2.0
        h2 = self.bn49(h2)
        h2 = relu(h2)
        num = 50
        h2 = self.conv50(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn50(h2)
        h3 = self.conv51(h)  # branch 3.0
        h3 = self.bn51(h3)
        h3 = relu(h3)
        num = 52
        h3 = self.conv52(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn52(h3)
        h4 = self.maxpool12(h)  # branch 4.0
        h4 = self.conv53(h4)  # branch 4.1
        h4 = self.bn53(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 5b
        h1 = self.conv54(h)  # branch 1
        h1 = self.bn54(h1)
        h2 = self.conv55(h)  # branch 2.0
        h2 = self.bn55(h2)
        h2 = relu(h2)
        num = 56
        h2 = self.conv56(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn56(h2)
        h3 = self.conv57(h)  # branch 3.0
        h3 = self.bn57(h3)
        h3 = relu(h3)
        num = 58
        h3 = self.conv58(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn58(h3)
        h4 = self.maxpool13(h)  # branch 4.0
        h4 = self.conv59(h4)  # branch 4.1
        h4 = self.bn59(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        h = self.avgpool3(h)
        h = self.dropout3(h)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        y = self.fc5(h)
        return aux1, aux2, y

# [1, 3, 6, 8, 12, 14, 18, 20, 24, 26, 31, 33, 37, 39, 43, 45, 50, 52, 56, 58]


class Googlenet3DConv_explicit(torch.nn.Module):
    def __init__(self, pv):
        super(Googlenet3DConv_explicit, self).__init__()

        self.conv1 = Conv3d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = BatchNorm3d(64)
        self.maxpool1 = MaxPool3d(kernel_size=(1, 3, 3), padding=0, stride=(1, 2, 2))
        self.conv2 = Conv3d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn2 = BatchNorm3d(64)
        self.conv3 = Conv3d(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn3 = BatchNorm3d(192)
        self.maxpool2 = MaxPool3d(kernel_size=(1, 3, 3), padding=0, stride=(1, 2, 2))

        # inception 3a
        self.conv4 = Conv3d(in_channels=192, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn4 = BatchNorm3d(64)
        self.conv5 = Conv3d(in_channels=192, out_channels=96, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn5 = BatchNorm3d(96)
        self.conv6 = Conv3d(in_channels=96, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn6 = BatchNorm3d(128)
        self.conv7 = Conv3d(in_channels=192, out_channels=16, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn7 = BatchNorm3d(16)
        self.conv8 = Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn8 = BatchNorm3d(32)
        self.maxpool3 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv9 = Conv3d(in_channels=192, out_channels=32, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn9 = BatchNorm3d(32)

        # inception 3b
        self.conv10 = Conv3d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn10 = BatchNorm3d(128)
        self.conv11 = Conv3d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn11 = BatchNorm3d(128)
        self.conv12 = Conv3d(in_channels=128, out_channels=192, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn12 = BatchNorm3d(192)
        self.conv13 = Conv3d(in_channels=256, out_channels=32, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn13 = BatchNorm3d(32)
        self.conv14 = Conv3d(in_channels=32, out_channels=96, kernel_size=3, padding=1, stride=1, bias=False)
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
        self.conv18 = Conv3d(in_channels=96, out_channels=208, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn18 = BatchNorm3d(208)
        self.conv19 = Conv3d(in_channels=480, out_channels=16, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn19 = BatchNorm3d(16)
        self.conv20 = Conv3d(in_channels=16, out_channels=48, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn20 = BatchNorm3d(48)
        self.maxpool6 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv21 = Conv3d(in_channels=480, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn21 = BatchNorm3d(64)

        # inception 4b
        self.conv22 = Conv3d(in_channels=512, out_channels=160, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn22 = BatchNorm3d(160)
        self.conv23 = Conv3d(in_channels=512, out_channels=112, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn23 = BatchNorm3d(112)
        self.conv24 = Conv3d(in_channels=112, out_channels=224, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn24 = BatchNorm3d(224)
        self.conv25 = Conv3d(in_channels=512, out_channels=24, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn25 = BatchNorm3d(24)
        self.conv26 = Conv3d(in_channels=24, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn26 = BatchNorm3d(64)
        self.maxpool7 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv27 = Conv3d(in_channels=512, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn27 = BatchNorm3d(64)

        self.avgpool1 = AvgPool3d(kernel_size=5, padding=0, stride=3)
        self.conv28 = Conv3d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn28 = BatchNorm3d(128)
        # self.fc1 = Linear(in_features=2304, out_features=1024)
        self.fc1 = Linear(in_features=768, out_features=1024)  # 768
        self.dropout1 = Dropout3d(p=0.7)
        self.fc2 = Linear(in_features=1024, out_features=pv.label_size)

        # inception 4c
        self.conv29 = Conv3d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn29 = BatchNorm3d(128)
        self.conv30 = Conv3d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn30 = BatchNorm3d(128)
        self.conv31 = Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn31 = BatchNorm3d(256)
        self.conv32 = Conv3d(in_channels=512, out_channels=24, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn32 = BatchNorm3d(24)
        self.conv33 = Conv3d(in_channels=24, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn33 = BatchNorm3d(64)
        self.maxpool8 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv34 = Conv3d(in_channels=512, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn34 = BatchNorm3d(64)

        # inception 4d
        self.conv35 = Conv3d(in_channels=512, out_channels=112, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn35 = BatchNorm3d(112)
        self.conv36 = Conv3d(in_channels=512, out_channels=144, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn36 = BatchNorm3d(144)
        self.conv37 = Conv3d(in_channels=144, out_channels=288, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn37 = BatchNorm3d(288)
        self.conv38 = Conv3d(in_channels=512, out_channels=32, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn38 = BatchNorm3d(32)
        self.conv39 = Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn39 = BatchNorm3d(64)
        self.maxpool9 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv40 = Conv3d(in_channels=512, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn40 = BatchNorm3d(64)

        # inception 4e
        self.conv41 = Conv3d(in_channels=528, out_channels=256, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn41 = BatchNorm3d(256)
        self.conv42 = Conv3d(in_channels=528, out_channels=160, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn42 = BatchNorm3d(160)
        self.conv43 = Conv3d(in_channels=160, out_channels=320, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn43 = BatchNorm3d(320)
        self.conv44 = Conv3d(in_channels=528, out_channels=32, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn44 = BatchNorm3d(32)
        self.conv45 = Conv3d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn45 = BatchNorm3d(128)
        self.maxpool10 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv46 = Conv3d(in_channels=528, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn46 = BatchNorm3d(128)

        self.avgpool2 = AvgPool3d(kernel_size=5, padding=0, stride=3)
        self.conv47 = Conv3d(in_channels=528, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn47 = BatchNorm3d(128)
        # self.fc3 = Linear(in_features=2304, out_features=1024)
        self.fc3 = Linear(in_features=768, out_features=1024)
        self.dropout2 = Dropout3d(p=0.7)
        self.fc4 = Linear(in_features=1024, out_features=pv.label_size)

        self.maxpool11 = MaxPool3d(kernel_size=3, padding=0, stride=2)

        # inception 5a
        self.conv48 = Conv3d(in_channels=832, out_channels=256, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn48 = BatchNorm3d(256)
        self.conv49 = Conv3d(in_channels=832, out_channels=160, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn49 = BatchNorm3d(160)
        self.conv50 = Conv3d(in_channels=160, out_channels=320, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn50 = BatchNorm3d(320)
        self.conv51 = Conv3d(in_channels=832, out_channels=32, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn51 = BatchNorm3d(32)
        self.conv52 = Conv3d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn52 = BatchNorm3d(128)
        self.maxpool12 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv53 = Conv3d(in_channels=832, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn53 = BatchNorm3d(128)

        # inception 5b
        self.conv54 = Conv3d(in_channels=832, out_channels=384, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn54 = BatchNorm3d(384)
        self.conv55 = Conv3d(in_channels=832, out_channels=192, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn55 = BatchNorm3d(192)
        self.conv56 = Conv3d(in_channels=192, out_channels=384, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn56 = BatchNorm3d(384)
        self.conv57 = Conv3d(in_channels=832, out_channels=48, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn57 = BatchNorm3d(48)
        self.conv58 = Conv3d(in_channels=48, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn58 = BatchNorm3d(128)
        self.maxpool13 = MaxPool3d(kernel_size=3, padding=1, stride=1)
        self.conv59 = Conv3d(in_channels=832, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn59 = BatchNorm3d(128)

        self.avgpool3 = AdaptiveAvgPool3d(1)
        self.dropout3 = Dropout3d(p=0.4)
        self.fc5 = Linear(in_features=1024, out_features=pv.label_size)

    def forward(self, x, stop_at=None, aux=True):
        # set aux=False during inference

        num = 1
        h = self.conv1(x)
        if stop_at == num:
            return h
        h = self.bn1(h)
        h = relu(h)
        h = self.maxpool1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = relu(h)
        num = 3
        h = self.conv3(h)
        if stop_at == num:
            return h
        h = self.bn3(h)
        h = relu(h)
        h = self.maxpool2(h)

        # inception 3a
        h1 = self.conv4(h)          # branch 1
        h1 = self.bn4(h1)
        h2 = self.conv5(h)          # branch 2.0
        h2 = self.bn5(h2)
        h2 = relu(h2)
        num = 6
        h2 = self.conv6(h2) # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn6(h2)
        h3 = self.conv7(h)          # branch 3.0
        h3 = self.bn7(h3)
        h3 = relu(h3)
        num = 8
        h3 = self.conv8(h3) # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn8(h3)
        h4 = self.maxpool3(h)       # branch 4.0
        h4 = self.conv9(h4)         # branch 4.1
        h4 = self.bn9(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 3b
        h1 = self.conv10(h)  # branch 1
        h1 = self.bn10(h1)
        h2 = self.conv11(h)  # branch 2.0
        h2 = self.bn11(h2)
        h2 = relu(h2)
        num = 12
        h2 = self.conv12(h2)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn12(h2)
        h3 = self.conv13(h)  # branch 3.0
        h3 = self.bn13(h3)
        h3 = relu(h3)
        num = 14
        h3 = self.conv14(h3)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn14(h3)
        h4 = self.maxpool4(h)  # branch 4.0
        h4 = self.conv15(h4)  # branch 4.1
        h4 = self.bn15(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)
        h = self.maxpool5(h)

        # inception 4a
        h1 = self.conv16(h)  # branch 1
        h1 = self.bn16(h1)
        h2 = self.conv17(h)  # branch 2.0
        h2 = self.bn17(h2)
        h2 = relu(h2)
        num = 18
        h2 = self.conv18(h2)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn18(h2)
        h3 = self.conv19(h)  # branch 3.0
        h3 = self.bn19(h3)
        h3 = relu(h3)
        num = 20
        h3 = self.conv20(h3)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn20(h3)
        h4 = self.maxpool6(h)  # branch 4.0
        h4 = self.conv21(h4)  # branch 4.1
        h4 = self.bn21(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 4b
        h1 = self.conv22(h)  # branch 1
        h1 = self.bn22(h1)
        h2 = self.conv23(h)  # branch 2.0
        h2 = self.bn23(h2)
        h2 = relu(h2)
        num = 24
        h2 = self.conv24(h2)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn24(h2)
        h3 = self.conv25(h)  # branch 3.0
        h3 = self.bn25(h3)
        h3 = relu(h3)
        num = 26
        h3 = self.conv26(h3)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn26(h3)
        h4 = self.maxpool7(h)  # branch 4.0
        h4 = self.conv27(h4)  # branch 4.1
        h4 = self.bn27(h4)

        if aux:
            aux1 = self.avgpool1(h)
            aux1 = self.conv28(aux1)
            aux1 = self.bn28(aux1)
            aux1 = relu(aux1)
            _shape = aux1.shape
            aux1 = aux1.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
            aux1 = self.fc1(aux1)
            aux1 = relu(aux1)
            aux1 = self.dropout1(aux1)
            aux1 = self.fc2(aux1)
        else:
            aux1 = None

        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 4c
        h1 = self.conv29(h)  # branch 1
        h1 = self.bn29(h1)
        h2 = self.conv30(h)  # branch 2.0
        h2 = self.bn30(h2)
        h2 = relu(h2)
        num = 31
        h2 = self.conv31(h2)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn31(h2)
        h3 = self.conv32(h)  # branch 3.0
        h3 = self.bn32(h3)
        h3 = relu(h3)
        num = 33
        h3 = self.conv33(h3)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn33(h3)
        h4 = self.maxpool8(h)  # branch 4.0
        h4 = self.conv34(h4)  # branch 4.1
        h4 = self.bn34(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 4d
        h1 = self.conv35(h)  # branch 1
        h1 = self.bn35(h1)
        h2 = self.conv36(h)  # branch 2.0
        h2 = self.bn36(h2)
        h2 = relu(h2)
        num = 37
        h2 = self.conv37(h2)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn37(h2)
        h3 = self.conv38(h)  # branch 3.0
        h3 = self.bn38(h3)
        h3 = relu(h3)
        num = 39
        h3 = self.conv39(h3)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn39(h3)
        h4 = self.maxpool9(h)  # branch 4.0
        h4 = self.conv40(h4)  # branch 4.1
        h4 = self.bn40(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 4e
        h1 = self.conv41(h)  # branch 1
        h1 = self.bn41(h1)
        h2 = self.conv42(h)  # branch 2.0
        h2 = self.bn42(h2)
        h2 = relu(h2)
        num = 43
        h2 = self.conv43(h2)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn43(h2)
        h3 = self.conv44(h)  # branch 3.0
        h3 = self.bn44(h3)
        h3 = relu(h3)
        num = 45
        h3 = self.conv45(h3)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn45(h3)
        h4 = self.maxpool10(h)  # branch 4.0
        h4 = self.conv46(h4)  # branch 4.1
        h4 = self.bn46(h4)

        if aux:
            aux2 = self.avgpool2(h)
            aux2 = self.conv47(aux2)
            aux2 = self.bn47(aux2)
            aux2 = relu(aux2)
            _shape = aux2.shape
            aux2 = aux2.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
            aux2 = self.fc3(aux2)
            aux2 = relu(aux2)
            aux2 = self.dropout2(aux2)
            aux2 = self.fc4(aux2)
        else:
            aux2 = None

        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)
        h = self.maxpool11(h)

        # inception 5a
        h1 = self.conv48(h)  # branch 1
        h1 = self.bn48(h1)
        h2 = self.conv49(h)  # branch 2.0
        h2 = self.bn49(h2)
        h2 = relu(h2)
        num = 50
        h2 = self.conv50(h2)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn50(h2)
        h3 = self.conv51(h)  # branch 3.0
        h3 = self.bn51(h3)
        h3 = relu(h3)
        num = 52
        h3 = self.conv52(h3)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn52(h3)
        h4 = self.maxpool12(h)  # branch 4.0
        h4 = self.conv53(h4)  # branch 4.1
        h4 = self.bn53(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 5b
        h1 = self.conv54(h)  # branch 1
        h1 = self.bn54(h1)
        h2 = self.conv55(h)  # branch 2.0
        h2 = self.bn55(h2)
        h2 = relu(h2)
        num = 56
        h2 = self.conv56(h2)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn56(h2)
        h3 = self.conv57(h)  # branch 3.0
        h3 = self.bn57(h3)
        h3 = relu(h3)
        num = 58
        h3 = self.conv58(h3)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn58(h3)
        h4 = self.maxpool13(h)  # branch 4.0
        h4 = self.conv59(h4)  # branch 4.1
        h4 = self.bn59(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        h = self.avgpool3(h)
        h = self.dropout3(h)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        y = self.fc5(h)
        return aux1, aux2, y

'''
        h = self.avgpool3(h)  # such that the D dimension is untouched
        
        h = self.dropout3(h)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        y = self.conv60(h)  # conv only in the D dimension
        return aux1, aux2, y
'''

class Googlenet3TConv_explicit_dyn(torch.nn.Module):
    def __init__(self, pv):
        super(Googlenet3TConv_explicit_dyn, self).__init__()

        self.conv1 = ConvTTN3d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2, project_variable=pv, bias=False)
        self.bn1 = BatchNorm3d(64)
        self.maxpool1 = MaxPool3d(kernel_size=(1, 3, 3), padding=0, stride=(1, 2, 2))
        self.conv2 = Conv3d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn2 = BatchNorm3d(64)
        self.conv3 = ConvTTN3d(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn3 = BatchNorm3d(192)
        self.maxpool2 = MaxPool3d(kernel_size=(1, 3, 3), padding=0, stride=(1, 2, 2))

        # inception 3a
        self.conv4 = Conv3d(in_channels=192, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn4 = BatchNorm3d(64)
        self.conv5 = Conv3d(in_channels=192, out_channels=96, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn5 = BatchNorm3d(96)
        self.conv6 = ConvTTN3d(in_channels=96, out_channels=128, kernel_size=3, padding=1, stride=1, project_variable=pv, bias=False)
        self.bn6 = BatchNorm3d(128)
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
        self.conv27 = Conv3d(in_channels=512, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn27 = BatchNorm3d(64)

        self.avgpool1 = AvgPool3d(kernel_size=5, padding=0, stride=3)
        self.conv28 = Conv3d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn28 = BatchNorm3d(128)
        # self.fc1 = Linear(in_features=2304, out_features=1024)
        self.fc1 = Linear(in_features=768, out_features=1024)  # 768
        self.dropout1 = Dropout3d(p=0.7)
        self.fc2 = Linear(in_features=1024, out_features=pv.label_size)

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
        # self.fc3 = Linear(in_features=2304, out_features=1024)
        self.fc3 = Linear(in_features=768, out_features=1024)
        self.dropout2 = Dropout3d(p=0.7)
        self.fc4 = Linear(in_features=1024, out_features=pv.label_size)

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

        self.avgpool3 = AdaptiveAvgPool3d((3, 1, 1))
        self.dropout3 = Dropout3d(p=0.4)
        self.conv60 = ConvTTN3d_dynamic(in_channels=1024, out_channels=pv.label_size, kernel_size=(3, 1, 1),
                                        padding=0, stride=1, project_variable=pv, bias=False)


    def forward(self, x, device, stop_at=None, aux=True):
        # set aux=False during inference

        num = 1
        h = self.conv1(x, device)
        if stop_at == num:
            return h
        h = self.bn1(h)
        h = relu(h)
        h = self.maxpool1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = relu(h)
        num = 3
        h = self.conv3(h, device)
        if stop_at == num:
            return h
        h = self.bn3(h)
        h = relu(h)
        h = self.maxpool2(h)

        # inception 3a
        h1 = self.conv4(h)          # branch 1
        h1 = self.bn4(h1)
        h2 = self.conv5(h)          # branch 2.0
        h2 = self.bn5(h2)
        h2 = relu(h2)
        num = 6
        h2 = self.conv6(h2, device) # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn6(h2)
        h3 = self.conv7(h)          # branch 3.0
        h3 = self.bn7(h3)
        h3 = relu(h3)
        num = 8
        h3 = self.conv8(h3, device) # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn8(h3)
        h4 = self.maxpool3(h)       # branch 4.0
        h4 = self.conv9(h4)         # branch 4.1
        h4 = self.bn9(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 3b
        h1 = self.conv10(h)  # branch 1
        h1 = self.bn10(h1)
        h2 = self.conv11(h)  # branch 2.0
        h2 = self.bn11(h2)
        h2 = relu(h2)
        num = 12
        h2 = self.conv12(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn12(h2)
        h3 = self.conv13(h)  # branch 3.0
        h3 = self.bn13(h3)
        h3 = relu(h3)
        num = 14
        h3 = self.conv14(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn14(h3)
        h4 = self.maxpool4(h)  # branch 4.0
        h4 = self.conv15(h4)  # branch 4.1
        h4 = self.bn15(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)
        h = self.maxpool5(h)

        # inception 4a
        h1 = self.conv16(h)  # branch 1
        h1 = self.bn16(h1)
        h2 = self.conv17(h)  # branch 2.0
        h2 = self.bn17(h2)
        h2 = relu(h2)
        num = 18
        h2 = self.conv18(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn18(h2)
        h3 = self.conv19(h)  # branch 3.0
        h3 = self.bn19(h3)
        h3 = relu(h3)
        num = 20
        h3 = self.conv20(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn20(h3)
        h4 = self.maxpool6(h)  # branch 4.0
        h4 = self.conv21(h4)  # branch 4.1
        h4 = self.bn21(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 4b
        h1 = self.conv22(h)  # branch 1
        h1 = self.bn22(h1)
        h2 = self.conv23(h)  # branch 2.0
        h2 = self.bn23(h2)
        h2 = relu(h2)
        num = 24
        h2 = self.conv24(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn24(h2)
        h3 = self.conv25(h)  # branch 3.0
        h3 = self.bn25(h3)
        h3 = relu(h3)
        num = 26
        h3 = self.conv26(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn26(h3)
        h4 = self.maxpool7(h)  # branch 4.0
        h4 = self.conv27(h4)  # branch 4.1
        h4 = self.bn27(h4)

        if aux:
            aux1 = self.avgpool1(h)
            aux1 = self.conv28(aux1)
            aux1 = self.bn28(aux1)
            aux1 = relu(aux1)
            _shape = aux1.shape
            aux1 = aux1.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
            aux1 = self.fc1(aux1)
            aux1 = relu(aux1)
            aux1 = self.dropout1(aux1)
            aux1 = self.fc2(aux1)
        else:
            aux1 = None

        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 4c
        h1 = self.conv29(h)  # branch 1
        h1 = self.bn29(h1)
        h2 = self.conv30(h)  # branch 2.0
        h2 = self.bn30(h2)
        h2 = relu(h2)
        num = 31
        h2 = self.conv31(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn31(h2)
        h3 = self.conv32(h)  # branch 3.0
        h3 = self.bn32(h3)
        h3 = relu(h3)
        num = 33
        h3 = self.conv33(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn33(h3)
        h4 = self.maxpool8(h)  # branch 4.0
        h4 = self.conv34(h4)  # branch 4.1
        h4 = self.bn34(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 4d
        h1 = self.conv35(h)  # branch 1
        h1 = self.bn35(h1)
        h2 = self.conv36(h)  # branch 2.0
        h2 = self.bn36(h2)
        h2 = relu(h2)
        num = 37
        h2 = self.conv37(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn37(h2)
        h3 = self.conv38(h)  # branch 3.0
        h3 = self.bn38(h3)
        h3 = relu(h3)
        num = 39
        h3 = self.conv39(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn39(h3)
        h4 = self.maxpool9(h)  # branch 4.0
        h4 = self.conv40(h4)  # branch 4.1
        h4 = self.bn40(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 4e
        h1 = self.conv41(h)  # branch 1
        h1 = self.bn41(h1)
        h2 = self.conv42(h)  # branch 2.0
        h2 = self.bn42(h2)
        h2 = relu(h2)
        num = 43
        h2 = self.conv43(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn43(h2)
        h3 = self.conv44(h)  # branch 3.0
        h3 = self.bn44(h3)
        h3 = relu(h3)
        num = 45
        h3 = self.conv45(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn45(h3)
        h4 = self.maxpool10(h)  # branch 4.0
        h4 = self.conv46(h4)  # branch 4.1
        h4 = self.bn46(h4)

        if aux:
            aux2 = self.avgpool2(h)
            aux2 = self.conv47(aux2)
            aux2 = self.bn47(aux2)
            aux2 = relu(aux2)
            _shape = aux2.shape
            aux2 = aux2.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
            aux2 = self.fc3(aux2)
            aux2 = relu(aux2)
            aux2 = self.dropout2(aux2)
            aux2 = self.fc4(aux2)
        else:
            aux2 = None

        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)
        h = self.maxpool11(h)

        # inception 5a
        h1 = self.conv48(h)  # branch 1
        h1 = self.bn48(h1)
        h2 = self.conv49(h)  # branch 2.0
        h2 = self.bn49(h2)
        h2 = relu(h2)
        num = 50
        h2 = self.conv50(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn50(h2)
        h3 = self.conv51(h)  # branch 3.0
        h3 = self.bn51(h3)
        h3 = relu(h3)
        num = 52
        h3 = self.conv52(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn52(h3)
        h4 = self.maxpool12(h)  # branch 4.0
        h4 = self.conv53(h4)  # branch 4.1
        h4 = self.bn53(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)

        # inception 5b
        h1 = self.conv54(h)  # branch 1
        h1 = self.bn54(h1)
        h2 = self.conv55(h)  # branch 2.0
        h2 = self.bn55(h2)
        h2 = relu(h2)
        num = 56
        h2 = self.conv56(h2, device)  # branch 2.1
        if stop_at == num:
            return h2
        h2 = self.bn56(h2)
        h3 = self.conv57(h)  # branch 3.0
        h3 = self.bn57(h3)
        h3 = relu(h3)
        num = 58
        h3 = self.conv58(h3, device)  # branch 3.1
        if stop_at == num:
            return h3
        h3 = self.bn58(h3)
        h4 = self.maxpool13(h)  # branch 4.0
        h4 = self.conv59(h4)  # branch 4.1
        h4 = self.bn59(h4)
        h = torch.cat((h1, h2, h3, h4), dim=1)
        h = relu(h)  # torch.Size([1, 1024, 3, 3, 6])

        h = self.avgpool3(h)
        h = self.dropout3(h)
        num = 60
        y = self.conv60(h, device)
        if stop_at == num:
            return y
        _shape = y.shape
        y = y.view(_shape[0], _shape[1])
        return aux1, aux2, y

# [1, 3, 6, 8, 12, 14, 18, 20, 24, 26, 31, 33, 37, 39, 43, 45, 50, 52, 56, 58, 60]