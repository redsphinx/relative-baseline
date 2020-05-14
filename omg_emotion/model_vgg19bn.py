import torch
from torch.nn.functional import relu
from torch.nn import MaxPool3d, AdaptiveAvgPool3d, Conv3d, BatchNorm3d, Linear, Dropout

from relative_baseline.omg_emotion.convttn3d import ConvTTN3d


class VGG19BN_Explicit_3T (torch.nn.Module):
    def __init__(self, pv):
        super(VGG19BN_Explicit_3T , self).__init__()

        self.conv1 = ConvTTN3d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn1 = BatchNorm3d(64)
        self.conv2 = ConvTTN3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn2 = BatchNorm3d(64)
        self.maxpool1 = MaxPool3d(kernel_size=2, padding=0, stride=2)

        self.conv3 = ConvTTN3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn3 = BatchNorm3d(128)
        self.conv4 = ConvTTN3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn4 = BatchNorm3d(128)
        self.maxpool2 = MaxPool3d(kernel_size=2, padding=0, stride=2)

        self.conv5 = ConvTTN3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn5 = BatchNorm3d(256)
        self.conv6 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn6 = BatchNorm3d(256)
        self.conv7 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn7 = BatchNorm3d(256)
        self.conv8 = ConvTTN3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn8 = BatchNorm3d(256)
        self.maxpool3 = MaxPool3d(kernel_size=2, padding=0, stride=2)

        self.conv9 = ConvTTN3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn9 = BatchNorm3d(512)
        self.conv10 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn10 = BatchNorm3d(512)
        self.conv11 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn11 = BatchNorm3d(512)
        self.conv12 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn12 = BatchNorm3d(512)
        self.maxpool4 = MaxPool3d(kernel_size=2, padding=0, stride=2)

        self.conv13 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn13 = BatchNorm3d(512)
        self.conv14 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn14 = BatchNorm3d(512)
        self.conv15 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn15 = BatchNorm3d(512)
        self.conv16 = ConvTTN3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, project_variable=pv, bias=True)
        self.bn16 = BatchNorm3d(512)
        self.maxpool5 = MaxPool3d(kernel_size=2, padding=0, stride=2)

        self.avgpool = AdaptiveAvgPool3d(output_size=(7, 7))
        self.fc1 = Linear(25088, 4096)
        self.dropout1 = Dropout(p=0.5)
        self.fc2 = Linear(4096, 4096)
        self.dropout2 = Dropout(p=0.5)
        self.fc3 = Linear(4096, 27)


    def forward(self, x, device, stop_at=None):

        h = self.conv1(x, device)
        if stop_at == 1:
            return h
        h = self.bn1(h)
        h = relu(h)
        h = self.conv2(h, device)
        if stop_at == 2:
            return h
        h = self.bn2(h)
        h = relu(h)
        h = self.maxpool1(h)

        h = self.conv3(h, device)
        if stop_at == 3:
            return h
        h = self.bn3(h)
        h = relu(h)
        h = self.conv4(h, device)
        if stop_at == 4:
            return h
        h = self.bn4(h)
        h = relu(h)
        h = self.maxpool2(h)

        h = self.conv5(h, device)
        if stop_at == 5:
            return h
        h = self.bn5(h)
        h = relu(h)
        h = self.conv6(h, device)
        if stop_at == 6:
            return h
        h = self.bn6(h)
        h = relu(h)
        h = self.conv7(h, device)
        if stop_at == 7:
            return h
        h = self.bn7(h)
        h = relu(h)
        h = self.conv8(h, device)
        if stop_at == 8:
            return h
        h = self.bn8(h)
        h = relu(h)
        h = self.maxpool3(h)

        h = self.conv9(h, device)
        if stop_at == 9:
            return h
        h = self.bn9(h)
        h = relu(h)
        h = self.conv10(h, device)
        if stop_at == 10:
            return h
        h = self.bn10(h)
        h = relu(h)
        h = self.conv11(h, device)
        if stop_at == 11:
            return h
        h = self.bn11(h)
        h = relu(h)
        h = self.conv12(h, device)
        if stop_at == 12:
            return h
        h = self.bn12(h)
        h = relu(h)
        h = self.maxpool4(h)

        h = self.conv13(h, device)
        if stop_at == 13:
            return h
        h = self.bn13(h)
        h = relu(h)
        h = self.conv14(h, device)
        if stop_at == 14:
            return h
        h = self.bn14(h)
        h = relu(h)
        h = self.conv15(h, device)
        if stop_at == 15:
            return h
        h = self.bn15(h)
        h = relu(h)
        h = self.conv16(h, device)
        if stop_at == 16:
            return h
        h = self.bn16(h)
        h = relu(h)
        h = self.maxpool4(h)

        h = self.avgpool(h)
        _shape = h.shape
        h = h.view(-1, _shape[1] * _shape[2] * _shape[3] * _shape[4])
        h = self.fc1(h)
        h = self.dropout1(h)
        h = self.fc2(h)
        h = self.dropout2(h)
        y = self.fc3(h)

        return y