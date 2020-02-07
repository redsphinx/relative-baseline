import os
import numpy as np
from PIL import Image

import torch
from torch.optim import Adam
from torch.autograd import Variable

from relative_baseline.omg_emotion.xai_tools.misc_functions import preprocess_image, recreate_image, save_clip


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
        # TODO
        return None

    # TODO: construct the deconv model based on model number and which_conv
    deconv_model = get_deconv_model()

    # pass input through my_model until the point of interest
    x1 = my_model.conv1(input, device)
    x2 = my_model.max_pool_1(x1)
    x3 = torch.nn.functional.relu(x2)

    if which_conv == 'conv2':
        x4 = my_model.conv2(x3, device)
        x5 = my_model.max_pool_2(x4)
        x6 = torch.nn.functional.relu(x5)

    # TODO: set the irrelevant activations to zero

    # pass the activations as input to the deconv_model
    if which_conv == 'conv1':
        reconstruction = deconv_model(x3)
    elif which_conv == 'conv2':
        reconstruction = deconv_model(x6)
    else:
        reconstruction = None

    # save reconstruction
    save_location = '/home/gabras/deployed/relative_baseline/omg_emotion/images/zeiler2014'
    if not os.path.exists(save_location):
        os.mkdir(save_location)
    name = '%s_epoch_%d.png' % (which_conv, epoch)
    save_path = os.path.join(save_location, name)
    im = Image.fromarray(reconstruction, mode='L')

    im.save(save_path)
