from relative_baseline.omg_emotion.settings import ProjectVariable
from torchvision.models import resnet18
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
import torch
from torch import nn
from relative_baseline.omg_emotion import project_paths as PP
import os
from relative_baseline.omg_emotion import models as M


def prepare_model(project_variable, model):
    # resnet
    if project_variable.model_number == 0:
        model.fc = nn.Linear(in_features=512, out_features=project_variable.label_size, bias=True)

    return model


def get_model(project_variable):
    # project_variable = ProjectVariable()

    if project_variable.model_number == 0:
        model = resnet18(pretrained=project_variable.pretrain_resnet18_weights)
        model = prepare_model(project_variable, model)
        if project_variable.load_model is not None:
            ex, mo, ep = project_variable.load_model
            path = os.path.join(PP.models, 'experiment_%d_model_%d' % (ex, mo), 'epoch_%d' % ep)
            model.load_state_dict(torch.load(path))
            # TODO: https://cs230-stanford.github.io/pytorch-getting-started.html#training-vs-evaluation
            # model.eval()

            print('experiment_%d model_%d epoch_%d loaded' % (ex, mo, ep))
    elif project_variable.model_number == 1:
        model = M.LeNet5_2d()
    elif project_variable.model_number == 2:
        model = M.LeNet5_3d()
    elif project_variable.model_number == 3:
        # TODO: implement transfer from 2D here

        model = M.LeNet5_TTN3d(project_variable)
        model.conv1.weight.requires_grad = False
        model.conv2.weight.requires_grad = False

        # if not initializing from theta, only update s r x y; do not update theta with backprop
        # if project_variable.theta_init is None:
        #     model.conv1.theta.requires_grad = False
        #     model.conv2.theta.requires_grad = False

    else:
        print('Error: model with number %d not supported' % project_variable.model_number)
        model = None

    return model


def get_optimizer(project_variable, model):
    # project_variable = ProjectVariable()

    if project_variable.optimizer == 'adam':
        if project_variable.model_number == 0:

            optimizer = Adam(
                [
                    {'params': model.conv1.parameters(), 'lr': project_variable.learning_rate/64},
                    {'params': model.bn1.parameters(), 'lr': project_variable.learning_rate/32},
                    {'params': model.layer1.parameters(), 'lr': project_variable.learning_rate/16},
                    {'params': model.layer2.parameters(), 'lr': project_variable.learning_rate/8},
                    {'params': model.layer3.parameters(), 'lr': project_variable.learning_rate/4},
                    {'params': model.layer4.parameters(), 'lr': project_variable.learning_rate/2},
                    {'params': model.fc.parameters(), 'lr': project_variable.learning_rate}
                ],
                lr=project_variable.learning_rate
            )
        else:
            optimizer = Adam(model.parameters(), lr=project_variable.learning_rate)

    elif project_variable.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=project_variable.learning_rate, momentum=project_variable.momentum)

    else:
        print('Error: optimizer %s not supported' % project_variable.optimizer)
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

