from relative_baseline.omg_emotion.settings import ProjectVariable
from torchvision.models import resnet18
from torch.optim.adam import Adam
import torch
from torch import nn


def prepare_model(project_variable, model):
    # resnet
    if project_variable.model_number == 0:
        model.fc = nn.Linear(in_features=512, out_features=project_variable.label_size, bias=True)

    return model


def get_model(project_variable):
    if project_variable.model_number == 0:
        model = resnet18(pretrained=True)
        model = prepare_model(project_variable, model)
    else:
        model = None

    return model


def get_optimizer(project_variable, model):
    # project_variable = ProjectVariable()

    if project_variable.optimizer[0] == 'adam':
        optimizer = Adam(
            [
                {'params': model.conv1.parameters(), 'lr': project_variable.learning_rate/7},
                {'params': model.bn1.parameters(), 'lr': project_variable.learning_rate/6},
                {'params': model.layer1.parameters(), 'lr': project_variable.learning_rate/5},
                {'params': model.layer2.parameters(), 'lr': project_variable.learning_rate/4},
                {'params': model.layer3.parameters(), 'lr': project_variable.learning_rate/3},
                {'params': model.layer4.parameters(), 'lr': project_variable.learning_rate/2},
                {'params': model.fc.parameters(), 'lr': project_variable.learning_rate}
            ],
            lr=project_variable.learning_rate
        )
    else:
        optimizer = None

    return optimizer


def get_device(project_variable):
    if project_variable.device is None:
        _dev = 'cpu'
    elif type(project_variable.device) is int:
        _dev = 'cuda:%d' % project_variable.device
    else:
        _dev = None

    device = torch.device(_dev)

    return device
