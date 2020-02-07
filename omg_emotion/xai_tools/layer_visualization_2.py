import os
import numpy as np
from PIL import Image

import torch
from torch.optim import Adam
from torch.autograd import Variable

from relative_baseline.omg_emotion.xai_tools.misc_functions import preprocess_image, recreate_image, save_clip
from relative_baseline.omg_emotion.models import deconv_3DTTN

def run_erhan2009(my_model, device, epoch):
    # based on "Visualizing Higher-Layer Features of a Deep Network" by Erhan et al. 2009

    processed_image = torch.rand((1, 1, 50, 28, 28), requires_grad=True, device=device)
    optimizer = Adam([processed_image], lr=0.01, weight_decay=0)
    mini_epochs = 20

    for i in range(1, mini_epochs+1):
        optimizer.zero_grad()

        x = processed_image
        x = my_model.conv1(x, device)
        conv_output = x[0, 0]

        loss = -torch.mean(conv_output)
        # print('Iteration: %s, Loss: %0.2f' % (str(i), float(loss.data.cpu())))
        # Backward
        loss.backward()
        # Update image
        optimizer.step()

        if i == mini_epochs:
            save_location = '/home/gabras/deployed/relative_baseline/omg_emotion/images/erhan2009'
            save_clip(conv_output, save_location, epoch)


def run_zeiler2014(project_variable, input, my_model, device, epoch, which_conv):
    # based on "Visualizing and Understanding Convolutional Networks" by Zeiler et al. 2014

    assert(project_variable.model_number == 11)

    def get_deconv_model():
        model = deconv_3DTTN(which_conv)
        # copy the weights from trained model
        model.deconv1.weight = my_model.conv1.weight.data
        model.deconv1.bias = my_model.conv1.bias.data
        
        if which_conv == 'conv2':
            model.deconv2.weight = my_model.conv2.weight.data
            model.deconv2.bias = my_model.conv2.bias.data

        return model

    deconv_model = get_deconv_model()
    
    # for the unpooling switches
    switches = []

    # pass input through my_model until the point of interest
    x1 = my_model.conv1(input, device)
    x2, _s = my_model.max_pool_1(x1)
    switches.append(_s)
    x3 = torch.nn.functional.relu(x2)

    if which_conv == 'conv2':
        x4 = my_model.conv2(x3, device)
        x5, _s = my_model.max_pool_2(x4)
        switches.append(_s)
        x6 = torch.nn.functional.relu(x5)

    # set the irrelevant activations to zero
    # assume that it means: ignore the higher up layers

    # pass the activations as input to the deconv_model
    if which_conv == 'conv1':
        reconstruction = deconv_model(x3, switches[0])
    elif which_conv == 'conv2':
        reconstruction = deconv_model(x6, switches[1])
    else:
        reconstruction = None

    # save reconstruction
    save_location = '/home/gabras/deployed/relative_baseline/omg_emotion/images/zeiler2014'
    if not os.path.exists(save_location):
        os.mkdir(save_location)

    # automatic number assignment
    existing = os.listdir(save_location)
    if len(existing) != 0:
        existing = [int(i.split('_')[-1]) for i in existing]
        existing.sort()
        number = max(existing) + 1
    else:
        number = 0

    folder = '%s_epoch_%d_n%d' % (which_conv, epoch, number)
    save_path = os.path.join(save_location, folder)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for f in range(reconstruction.shape[0]):
        name = 'frame_%d.png' % f
        ultimate_path = os.path.join(save_path, name)

        im = Image.fromarray(reconstruction[f], mode='L')
        im.save(ultimate_path)
