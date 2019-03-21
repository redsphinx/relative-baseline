from torchvision.models import resnet18
from torch.optim.adam import Adam
import torch
from torch import nn

# https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch


batch_size = 5
nb_classes = 2
in_features = 10

model = nn.Linear(in_features, nb_classes)
criterion = nn.CrossEntropyLoss()

x = torch.randn(batch_size, in_features)
target = torch.empty(batch_size, dtype=torch.long).random_(nb_classes)

output = model(x)
loss = criterion(output, target)
loss.backward()