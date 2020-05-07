import numpy as np

from relative_baseline.omg_emotion.convttn3d import ConvTTN3d

import torch
from torch.nn.functional import relu
from torch.nn import MaxPool3d, MaxPool3d, AdaptiveAvgPool3d, AdaptiveAvgPool3d, Conv3d, Conv3d, BatchNorm3d, BatchNorm3d


class Googlenet3TConv_explicit(torch.nn.Module):
    def __init__(self):
        super(Googlenet3TConv_explicit, self).__init__()

