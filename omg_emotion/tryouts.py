from torchvision.models import resnet18
from torch.optim.adam import Adam
import torch
from torch import nn

# https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch


model = resnet18(pretrained=True)

model.fc = nn.Linear(in_features=512, out_features=7, bias=True)

for i, param in enumerate(model.named_parameters()):

