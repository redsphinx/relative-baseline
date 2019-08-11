from relative_baseline.omg_emotion.settings import ProjectVariable
from torchvision.models import resnet18
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
import torch
from torch import nn
from relative_baseline.omg_emotion import project_paths as PP
import os
from relative_baseline.omg_emotion import models as M

# TODO
# bzip2: https://pymotw.com/2/bz2/


def prepare_model(project_variable, model):
    # resnet
    if project_variable.model_number == 0:
        model.fc = nn.Linear(in_features=512, out_features=project_variable.label_size, bias=True)

    return model


def get_model(project_variable):
    # project_variable = ProjectVariable()
    if project_variable.load_model is not None:
        if len(project_variable.load_model) == 3:
            ex, mo, ep = project_variable.load_model
            path = os.path.join(PP.models, 'experiment_%d_model_%d' % (ex, mo), 'epoch_%d' % ep)
        else:
            ex, mo, ep, run = project_variable.load_model
            path = os.path.join(PP.models, 'experiment_%d_model_%d_run_%d' % (ex, mo, run), 'epoch_%d' % ep)

        if not os.path.exists(path):
            print("ERROR: saved model path '%s' does not exist" % path)
            return None
    else:
        path, ex, mo, ep = None, None, None, None

    if project_variable.model_number == 0:
        model = resnet18(pretrained=project_variable.pretrain_resnet18_weights)
        model = prepare_model(project_variable, model)
        # TODO: FIX WEIGHT LOADING
        if project_variable.load_model is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            # https://cs230-stanford.github.io/pytorch-getting-started.html#training-vs-evaluation
            # model.eval()
            print('experiment_%d model_%d epoch_%d loaded' % (ex, mo, ep))

    elif project_variable.model_number == 1:
        model = M.LeNet5_2d()
        # TODO: FIX WEIGHT LOADING
        if project_variable.load_model is not None:
            if mo == 1:
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                print('experiment_%d model_%d epoch_%d loaded' % (ex, mo, ep))
            else:
                print('ERROR: loading weights from model_number=%d not supported for model_number=%d'
                      % (mo, project_variable.model_number))


    elif project_variable.model_number == 2:
        model = M.LeNet5_3d(project_variable)
        # TODO: FIX WEIGHT LOADING
        if project_variable.load_model is not None:
            if mo == 1:
                pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
                model.conv1.weight = torch.nn.Parameter(pretrained_dict['conv1.weight'].unsqueeze(2).repeat(1, 1, project_variable.k_shape[0], 1, 1))
                model.conv1.bias = torch.nn.Parameter(pretrained_dict['conv1.bias'])
                model.conv2.weight = torch.nn.Parameter(pretrained_dict['conv2.weight'].unsqueeze(2).repeat(1, 1, project_variable.k_shape[0], 1, 1))
                model.conv2.bias = torch.nn.Parameter(pretrained_dict['conv2.bias'])
                print('experiment_%d model_%d epoch_%d loaded' % (ex, mo, ep))
            elif mo == 2:
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                print('experiment_%d model_%d epoch_%d loaded' % (ex, mo, ep))
            else:
                print('ERROR: loading weights from model_number=%d not supported for model_number=%d'
                      % (mo, project_variable.model_number))

    elif project_variable.model_number == 3:
        model = M.LeNet5_TTN3d(project_variable)
        model.conv1.weight.requires_grad = False
        model.conv2.weight.requires_grad = False
        # TODO: FIX WEIGHT LOADING
        if project_variable.load_model is not None:
            if mo == 1:
                pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
                model.conv1.first_weight = torch.nn.Parameter(pretrained_dict['conv1.weight'].unsqueeze(2))
                model.conv1.bias = torch.nn.Parameter(pretrained_dict['conv1.bias'])
                model.conv2.first_weight = torch.nn.Parameter(pretrained_dict['conv2.weight'].unsqueeze(2))
                model.conv2.bias = torch.nn.Parameter(pretrained_dict['conv2.bias'])
                print('experiment_%d model_%d epoch_%d loaded' % (ex, mo, ep))
            elif mo == 3:
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                print('experiment_%d model_%d epoch_%d loaded' % (ex, mo, ep))
            else:
                print('ERROR: loading weights from model_number=%d not supported for model_number=%d'
                      % (mo, project_variable.model_number))
    elif project_variable.model_number == 4:
        model = M.Sota_3d([project_variable.load_num_frames, 60, 60])
    elif project_variable.model_number == 5:
        model = M.C3D([project_variable.load_num_frames, 60, 60], project_variable)
    else:
        print('ERROR: model_number=%d not supported' % project_variable.model_number)
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

