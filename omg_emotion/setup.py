from torchvision.models import resnet18, googlenet
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
import torch
from torch import nn
from relative_baseline.omg_emotion import project_paths as PP
import os
from relative_baseline.omg_emotion import models as M
from relative_baseline.omg_emotion import factorized_convolution as C3D
from relative_baseline.omg_emotion.evolution.model_maker import ModularConv
from relative_baseline.omg_emotion.model_resnet18 import ResNet18, ResNet18Explicit, ResNet18Explicit3DConv, ResNet18Explicit3DConvReduced
from relative_baseline.omg_emotion.model_googlenet import Googlenet3TConv_explicit

def prepare_model(project_variable, model):
    # resnet
    if project_variable.model_number == 0:
        model.fc = nn.Linear(in_features=512, out_features=project_variable.label_size, bias=True)

    return model


def get_model(project_variable):
    # project_variable = ProjectVariable()
    if type(project_variable.load_model) == bool:
        print('loading weights from resnet18')
    elif project_variable.load_model is not None:
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
        if project_variable.load_model is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            # https://cs230-stanford.github.io/pytorch-getting-started.html#training-vs-evaluation
            # model.eval()
            print('experiment_%d model_%d epoch_%d loaded' % (ex, mo, ep))

    elif project_variable.model_number == 1:
        model = M.LeNet5_2d()
        if project_variable.load_model is not None:
            if mo == 1:
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                print('experiment_%d model_%d epoch_%d loaded' % (ex, mo, ep))
            else:
                print('ERROR: loading weights from model_number=%d not supported for model_number=%d'
                      % (mo, project_variable.model_number))


    elif project_variable.model_number == 2:
        model = M.LeNet5_3d(project_variable)
        if project_variable.load_model is not None:
            if mo == 1:
                pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
                model.conv1.weight = torch.nn.Parameter(pretrained_dict['conv1.weight'].unsqueeze(2).repeat(1, 1, project_variable.k_shape[0], 1, 1))
                model.conv1.bias = torch.nn.Parameter(pretrained_dict['conv1.bias'])
                model.conv2.weight = torch.nn.Parameter(pretrained_dict['conv2.weight'].unsqueeze(2).repeat(1, 1, project_variable.k_shape[0], 1, 1))
                model.conv2.bias = torch.nn.Parameter(pretrained_dict['conv2.bias'])
                # model.fc1.weight = torch.nn.Parameter(pretrained_dict['fc1.weight'].repeat(1, 5))
                model.fc1.bias = torch.nn.Parameter(pretrained_dict['fc1.bias'])
                model.fc2.weight = torch.nn.Parameter(pretrained_dict['fc2.weight'])
                model.fc2.bias = torch.nn.Parameter(pretrained_dict['fc2.bias'])
                model.fc3.weight = torch.nn.Parameter(pretrained_dict['fc3.weight'])
                model.fc3.bias = torch.nn.Parameter(pretrained_dict['fc3.bias'])
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
        # mo is for transfer learning
        if project_variable.load_model is not None:
            if mo == 1: # transfer weights from 2d conv
                pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
                model.conv1.first_weight = torch.nn.Parameter(pretrained_dict['conv1.weight'].unsqueeze(2))
                model.conv1.bias = torch.nn.Parameter(pretrained_dict['conv1.bias'])
                model.conv2.first_weight = torch.nn.Parameter(pretrained_dict['conv2.weight'].unsqueeze(2))
                model.conv2.bias = torch.nn.Parameter(pretrained_dict['conv2.bias'])
                # model.fc1.weight = torch.nn.Parameter(pretrained_dict['fc1.weight'].repeat(1, 5))
                model.fc1.bias = torch.nn.Parameter(pretrained_dict['fc1.bias'])
                model.fc2.weight = torch.nn.Parameter(pretrained_dict['fc2.weight'])
                model.fc2.bias = torch.nn.Parameter(pretrained_dict['fc2.bias'])
                model.fc3.weight = torch.nn.Parameter(pretrained_dict['fc3.weight'])
                model.fc3.bias = torch.nn.Parameter(pretrained_dict['fc3.bias'])
                print('experiment_%d model_%d epoch_%d loaded' % (ex, mo, ep))
            elif mo == 3: # transfer weights from 3d ttn conv
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                print('experiment_%d model_%d epoch_%d loaded' % (ex, mo, ep))
            else:
                print('ERROR: loading weights from model_number=%d not supported for model_number=%d'
                      % (mo, project_variable.model_number))
    elif project_variable.model_number == 4:
        model = M.Sota_3d([project_variable.load_num_frames, 60, 60])
    elif project_variable.model_number == 5:
        model = M.C3D([project_variable.load_num_frames, 60, 60], project_variable)
    elif project_variable.model_number == 6:
        model = M.C3DTTN([project_variable.load_num_frames, 60, 60], project_variable)
        model.conv1.weight.requires_grad = False
        model.conv2.weight.requires_grad = False
        model.conv3.weight.requires_grad = False
        model.conv4.weight.requires_grad = False
    elif project_variable.model_number in [71, 72, 73, 75, 76, 77]:
        if project_variable.model_number == 71:
            model = M.C3TTN1([project_variable.load_num_frames, 60, 60], project_variable)
        elif project_variable.model_number == 72:
            model = M.C3TTN2([project_variable.load_num_frames, 60, 60], project_variable)
        elif project_variable.model_number == 73:
            model = M.C3TTN3([project_variable.load_num_frames, 60, 60], project_variable)
        elif project_variable.model_number == 75:
            model = M.C3TTN5([project_variable.load_num_frames, 60, 60], project_variable)
        elif project_variable.model_number == 76:
            model = M.C3TTN6([project_variable.load_num_frames, 60, 60], project_variable)
        elif project_variable.model_number == 77:
            model = M.C3TTN7([project_variable.load_num_frames, 60, 60], project_variable)
        model.conv1.weight.requires_grad = False
        model.conv2.weight.requires_grad = False
        model.conv3.weight.requires_grad = False
        model.conv4.weight.requires_grad = False
    elif project_variable.model_number == 74:
        model = M.C3TTN4([project_variable.load_num_frames, 60, 60], project_variable)
        model.conv1.weight.requires_grad = False
        model.conv2.weight.requires_grad = False
        model.conv3.weight.requires_grad = False
    elif project_variable.model_number == 8:
        model = M.C3DTTN_1L([project_variable.load_num_frames, 60, 60], project_variable)
        model.conv1.weight.requires_grad = False
    elif project_variable.model_number == 9:
        model = M.C3D_1L([project_variable.load_num_frames, 60, 60], project_variable)
        # model = M.C3D_1L([project_variable.load_num_frames, 28, 28], project_variable)
    elif project_variable.model_number == 10:
        model = C3D.LeNet5_TTN3d()
        model.conv1.weight.requires_grad = False
        model.conv2.weight.requires_grad = False
    elif project_variable.model_number == 11:
        model = M.LeNet5_TTN3d_xD([project_variable.load_num_frames, 96, 96], project_variable)
        model.conv1.weight.requires_grad = False
        model.conv2.weight.requires_grad = False

        if project_variable.load_model is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    elif project_variable.model_number == 12:
        model = M.LeNet5_3d_xD(project_variable)
        if project_variable.load_model is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    elif project_variable.model_number == 14:
        model = M.Experimental_TTN3d_xD(project_variable)
        if project_variable.load_model is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    elif project_variable.model_number == 15:
        model = M.Experimental15_TTN3d_xD(project_variable)
        if project_variable.load_model is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    elif project_variable.model_number == 16:
        model = ModularConv(project_variable)
        if project_variable.load_model is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    elif project_variable.model_number == 17:
        model = M.Model17(project_variable)
        if project_variable.load_model is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    elif project_variable.model_number == 18:
        model = M.Model18(project_variable)
        if project_variable.load_model is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    elif project_variable.model_number == 20:
        # model = ResNet18(project_variable)
        model = ResNet18Explicit(project_variable)
        if type(project_variable.load_model) != bool and not project_variable.load_model is None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        elif project_variable.load_model:
            # load resnet18 from pytorch
            tmp_resnet18 = resnet18(pretrained=True)
            # copy the weights
            # Umut version
            # model.conv1_relu.conv.first_weight = torch.nn.Parameter(tmp_resnet18.conv1.weight.unsqueeze(2))
            # model.res2a_relu.res_branch2a.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[0].conv1.weight.unsqueeze(2))
            # model.res2a_relu.res_branch2b.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[0].conv2.weight.unsqueeze(2))
            # model.res2b_relu.res_branch2a.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[1].conv1.weight.unsqueeze(2))
            # model.res2b_relu.res_branch2b.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[1].conv2.weight.unsqueeze(2))
            # model.res3a_relu.res_branch1.weight = torch.nn.Parameter(tmp_resnet18.layer2[0].downsample[0].weight.unsqueeze(2))
            # model.res3a_relu.res_branch2a.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[0].conv1.weight.unsqueeze(2))
            # model.res3a_relu.res_branch2b.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[0].conv2.weight.unsqueeze(2))
            # model.res3b_relu.res_branch2a.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[1].conv1.weight.unsqueeze(2))
            # model.res3b_relu.res_branch2b.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[1].conv2.weight.unsqueeze(2))
            # model.res4a_relu.res_branch1.weight = torch.nn.Parameter(tmp_resnet18.layer3[0].downsample[0].weight.unsqueeze(2))
            # model.res4a_relu.res_branch2a.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[0].conv1.weight.unsqueeze(2))
            # model.res4a_relu.res_branch2b.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[0].conv2.weight.unsqueeze(2))
            # model.res4b_relu.res_branch2a.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[1].conv1.weight.unsqueeze(2))
            # model.res4b_relu.res_branch2b.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[1].conv2.weight.unsqueeze(2))
            # model.res5a_relu.res_branch1.weight = torch.nn.Parameter(tmp_resnet18.layer4[0].downsample[0].weight.unsqueeze(2))
            # model.res5a_relu.res_branch2a.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[0].conv1.weight.unsqueeze(2))
            # model.res5a_relu.res_branch2b.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[0].conv2.weight.unsqueeze(2))
            # model.res5b_relu.res_branch2a.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[1].conv1.weight.unsqueeze(2))
            # model.res5b_relu.res_branch2b.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[1].conv2.weight.unsqueeze(2))

            # explicit version
            divide = True
            if divide:
                div1 = 7
                div2 = 3
            else:
                div1 = 1
                div2 = 1
            model.conv1.first_weight = torch.nn.Parameter(tmp_resnet18.conv1.weight.unsqueeze(2) / div1)
            model.conv2.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[0].conv1.weight.unsqueeze(2) / div2)
            model.conv3.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[0].conv2.weight.unsqueeze(2) / div2)
            model.conv4.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[1].conv1.weight.unsqueeze(2) / div2)
            model.conv5.first_weight = torch.nn.Parameter(tmp_resnet18.layer1[1].conv2.weight.unsqueeze(2) / div2)
            model.conv6.weight = torch.nn.Parameter(tmp_resnet18.layer2[0].downsample[0].weight.unsqueeze(2) / div2)
            model.conv7.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[0].conv1.weight.unsqueeze(2) / div2)
            model.conv8.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[0].conv2.weight.unsqueeze(2) / div2)
            model.conv9.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[1].conv1.weight.unsqueeze(2) / div2)
            model.conv10.first_weight = torch.nn.Parameter(tmp_resnet18.layer2[1].conv2.weight.unsqueeze(2) / div2)
            model.conv11.weight = torch.nn.Parameter(tmp_resnet18.layer3[0].downsample[0].weight.unsqueeze(2) / div2)
            model.conv12.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[0].conv1.weight.unsqueeze(2) / div2)
            model.conv13.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[0].conv2.weight.unsqueeze(2) / div2)
            model.conv14.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[1].conv1.weight.unsqueeze(2) / div2)
            model.conv15.first_weight = torch.nn.Parameter(tmp_resnet18.layer3[1].conv2.weight.unsqueeze(2) / div2)
            model.conv16.weight = torch.nn.Parameter(tmp_resnet18.layer4[0].downsample[0].weight.unsqueeze(2) / div2)
            model.conv17.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[0].conv1.weight.unsqueeze(2) / div2)
            model.conv18.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[0].conv2.weight.unsqueeze(2) / div2)
            model.conv19.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[1].conv1.weight.unsqueeze(2) / div2)
            model.conv20.first_weight = torch.nn.Parameter(tmp_resnet18.layer4[1].conv2.weight.unsqueeze(2) / div2)

        # set weights of 3D conv to not require grad
        # Umut version
        # model.conv1_relu.conv.weight.requires_grad = False
        # model.res2a_relu.res_branch2a.weight.requires_grad = False
        # model.res2a_relu.res_branch2b.weight.requires_grad = False
        # model.res2b_relu.res_branch2a.weight.requires_grad = False
        # model.res2b_relu.res_branch2b.weight.requires_grad = False
        # model.res3a_relu.res_branch2a.weight.requires_grad = False
        # model.res3a_relu.res_branch2b.weight.requires_grad = False
        # model.res3b_relu.res_branch2a.weight.requires_grad = False
        # model.res3b_relu.res_branch2b.weight.requires_grad = False
        # model.res4a_relu.res_branch2a.weight.requires_grad = False
        # model.res4a_relu.res_branch2b.weight.requires_grad = False
        # model.res4b_relu.res_branch2a.weight.requires_grad = False
        # model.res4b_relu.res_branch2b.weight.requires_grad = False
        # model.res5a_relu.res_branch2a.weight.requires_grad = False
        # model.res5a_relu.res_branch2b.weight.requires_grad = False
        # model.res5b_relu.res_branch2a.weight.requires_grad = False
        # model.res5b_relu.res_branch2b.weight.requires_grad = False

        # explicit version
        model.conv1.weight.requires_grad = False
        model.conv2.weight.requires_grad = False
        model.conv3.weight.requires_grad = False
        model.conv4.weight.requires_grad = False
        model.conv5.weight.requires_grad = False
        model.conv7.weight.requires_grad = False
        model.conv8.weight.requires_grad = False
        model.conv9.weight.requires_grad = False
        model.conv10.weight.requires_grad = False
        model.conv12.weight.requires_grad = False
        model.conv13.weight.requires_grad = False
        model.conv14.weight.requires_grad = False
        model.conv15.weight.requires_grad = False
        model.conv17.weight.requires_grad = False
        model.conv18.weight.requires_grad = False
        model.conv19.weight.requires_grad = False
        model.conv20.weight.requires_grad = False

    elif project_variable.model_number == 21:
        model = ResNet18Explicit3DConv()
        if project_variable.load_model is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    elif project_variable.model_number == 22:
        model = ResNet18Explicit3DConvReduced()
        if type(project_variable.load_model) != bool and not project_variable.load_model is None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    elif project_variable.model_number == 23:
        tmp_googlenet = googlenet(pretrained=True)
        model = Googlenet3TConv_explicit()
        if type(project_variable.load_model) != bool and not project_variable.load_model is None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        elif project_variable.load_model:
            # load googlenet from pytorch
            tmp_googlenet = googlenet(pretrained=True)
            print('asdf')

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
            if project_variable.theta_learning_rate is not None:
                if project_variable.theta_init is None:  # using SRXY to make theta
                    if project_variable.k0_theta_learning_rate and not project_variable.bias_theta_learning_rate:
                        # SRXY + k0
                        optimizer = Adam(
                            [
                                {'params': model.conv1.first_weight, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv1.bias, 'lr': project_variable.learning_rate},
                                {'params': model.conv1.scale, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv1.rotate, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv1.translate_x, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv1.translate_y, 'lr': project_variable.theta_learning_rate},

                                {'params': model.conv2.first_weight, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv2.bias, 'lr': project_variable.learning_rate},
                                {'params': model.conv2.scale, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv2.rotate, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv2.translate_x, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv2.translate_y, 'lr': project_variable.theta_learning_rate},

                                {'params': model.fc1.parameters(), 'lr': project_variable.learning_rate},
                                {'params': model.fc2.parameters(), 'lr': project_variable.learning_rate},
                                {'params': model.fc3.parameters(), 'lr': project_variable.learning_rate},
                            ],
                            lr=project_variable.learning_rate
                        )
                    elif project_variable.k0_theta_learning_rate and project_variable.bias_theta_learning_rate:
                        # SRXY + k0 + bias
                        optimizer = Adam(
                            [
                                {'params': model.conv1.first_weight, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv1.bias, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv1.scale, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv1.rotate, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv1.translate_x, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv1.translate_y, 'lr': project_variable.theta_learning_rate},

                                {'params': model.conv2.first_weight, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv2.bias, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv2.scale, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv2.rotate, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv2.translate_x, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv2.translate_y, 'lr': project_variable.theta_learning_rate},

                                {'params': model.fc1.parameters(), 'lr': project_variable.learning_rate},
                                {'params': model.fc2.parameters(), 'lr': project_variable.learning_rate},
                                {'params': model.fc3.parameters(), 'lr': project_variable.learning_rate},
                            ],
                            lr=project_variable.learning_rate
                        )
                    else:
                        # only SRXY
                        optimizer = Adam(
                            [
                                {'params': model.conv1.first_weight, 'lr': project_variable.learning_rate},
                                {'params': model.conv1.bias, 'lr': project_variable.learning_rate},
                                {'params': model.conv1.scale, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv1.rotate, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv1.translate_x, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv1.translate_y, 'lr': project_variable.theta_learning_rate},

                                {'params': model.conv2.first_weight, 'lr': project_variable.learning_rate},
                                {'params': model.conv2.bias, 'lr': project_variable.learning_rate},
                                {'params': model.conv2.scale, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv2.rotate, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv2.translate_x, 'lr': project_variable.theta_learning_rate},
                                {'params': model.conv2.translate_y, 'lr': project_variable.theta_learning_rate},

                                {'params': model.fc1.parameters(), 'lr': project_variable.learning_rate},
                                {'params': model.fc2.parameters(), 'lr': project_variable.learning_rate},
                                {'params': model.fc3.parameters(), 'lr': project_variable.learning_rate},
                            ],
                            lr=project_variable.learning_rate
                        )

                else:
                    optimizer = Adam(
                        [
                            # Use theta learning rate for theta parameters
                            # Use normal learning rate for other things

                            {'params': model.conv1.first_weight, 'lr': project_variable.learning_rate},
                            {'params': model.conv1.bias, 'lr': project_variable.learning_rate},
                            {'params': model.conv1.theta, 'lr': project_variable.theta_learning_rate},

                            {'params': model.conv2.first_weight, 'lr': project_variable.learning_rate},
                            {'params': model.conv2.bias, 'lr': project_variable.learning_rate},
                            {'params': model.conv2.theta, 'lr': project_variable.theta_learning_rate},

                            {'params': model.fc1.parameters(), 'lr': project_variable.learning_rate},
                            {'params': model.fc2.parameters(), 'lr': project_variable.learning_rate},
                            {'params': model.fc3.parameters(), 'lr': project_variable.learning_rate},
                        ],
                        lr=project_variable.learning_rate
                    )
            else:
                optimizer = Adam(model.parameters(), lr=project_variable.learning_rate)

    elif project_variable.optimizer == 'sgd':
        if project_variable.theta_learning_rate is not None:
            if project_variable.theta_init is None: # using SRXY to make theta
                if project_variable.k0_theta_learning_rate and not project_variable.bias_theta_learning_rate:
                    # SRXY + k0
                    optimizer = SGD(
                        [
                            {'params': model.conv1.first_weight, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv1.bias, 'lr': project_variable.learning_rate},
                            {'params': model.conv1.scale, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv1.rotate, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv1.translate_x, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv1.translate_y, 'lr': project_variable.theta_learning_rate},

                            {'params': model.conv2.first_weight, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv2.bias, 'lr': project_variable.learning_rate},
                            {'params': model.conv2.scale, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv2.rotate, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv2.translate_x, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv2.translate_y, 'lr': project_variable.theta_learning_rate},

                            {'params': model.fc1.parameters(), 'lr': project_variable.learning_rate},
                            {'params': model.fc2.parameters(), 'lr': project_variable.learning_rate},
                            {'params': model.fc3.parameters(), 'lr': project_variable.learning_rate},
                        ],
                        lr=project_variable.learning_rate
                    )
                elif project_variable.k0_theta_learning_rate and project_variable.bias_theta_learning_rate:
                    # SRXY + k0 + bias
                    optimizer = SGD(
                        [
                            {'params': model.conv1.first_weight, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv1.bias, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv1.scale, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv1.rotate, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv1.translate_x, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv1.translate_y, 'lr': project_variable.theta_learning_rate},

                            {'params': model.conv2.first_weight, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv2.bias, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv2.scale, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv2.rotate, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv2.translate_x, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv2.translate_y, 'lr': project_variable.theta_learning_rate},

                            {'params': model.fc1.parameters(), 'lr': project_variable.learning_rate},
                            {'params': model.fc2.parameters(), 'lr': project_variable.learning_rate},
                            {'params': model.fc3.parameters(), 'lr': project_variable.learning_rate},
                        ],
                        lr=project_variable.learning_rate
                    )
                else:
                    # only SRXY
                    optimizer = SGD(
                        [
                            {'params': model.conv1.first_weight, 'lr': project_variable.learning_rate},
                            {'params': model.conv1.bias, 'lr': project_variable.learning_rate},
                            {'params': model.conv1.scale, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv1.rotate, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv1.translate_x, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv1.translate_y, 'lr': project_variable.theta_learning_rate},

                            {'params': model.conv2.first_weight, 'lr': project_variable.learning_rate},
                            {'params': model.conv2.bias, 'lr': project_variable.learning_rate},
                            {'params': model.conv2.scale, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv2.rotate, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv2.translate_x, 'lr': project_variable.theta_learning_rate},
                            {'params': model.conv2.translate_y, 'lr': project_variable.theta_learning_rate},

                            {'params': model.fc1.parameters(), 'lr': project_variable.learning_rate},
                            {'params': model.fc2.parameters(), 'lr': project_variable.learning_rate},
                            {'params': model.fc3.parameters(), 'lr': project_variable.learning_rate},
                        ],
                        lr=project_variable.learning_rate
                    )

            else:
                optimizer = SGD(
                    [
                        # Use theta learning rate for theta parameters
                        # Use normal learning rate for other things

                        {'params': model.conv1.first_weight, 'lr': project_variable.learning_rate},
                        {'params': model.conv1.bias, 'lr': project_variable.learning_rate},
                        {'params': model.conv1.theta, 'lr': project_variable.theta_learning_rate},

                        {'params': model.conv2.first_weight, 'lr': project_variable.learning_rate},
                        {'params': model.conv2.bias, 'lr': project_variable.learning_rate},
                        {'params': model.conv2.theta, 'lr': project_variable.theta_learning_rate},

                        {'params': model.fc1.parameters(), 'lr': project_variable.learning_rate},
                        {'params': model.fc2.parameters(), 'lr': project_variable.learning_rate},
                        {'params': model.fc3.parameters(), 'lr': project_variable.learning_rate},
                    ],
                    lr=project_variable.learning_rate
                )

        else:
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


