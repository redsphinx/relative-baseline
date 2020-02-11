import os
import numpy as np
from PIL import Image

import torch
from torch.optim import Adam
from torch.autograd import Variable

from relative_baseline.omg_emotion.xai_tools.misc_functions import preprocess_image, recreate_image, save_clip
from relative_baseline.omg_emotion.models import deconv_3DTTN
import relative_baseline.omg_emotion.project_paths as PP


def save_image(image, location, epoch_number):

    assert(isinstance(image, torch.Tensor)), "Image type not torch.Tensor: '%s' " % str(type(image))

    image = np.array(image.data.cpu(), dtype=np.uint8)
    assert(image.shape[1] == 1), "Image contains more than 1 channel: '%s'" % str()

    if image.shape[0] > 1:
        folder = os.path.join(location, 'epoch_%d' % epoch_number)
        if not os.path.exists(folder):
            os.mkdir(folder)

        for i in range(image.shape[0]):
            im = Image.fromarray(image[i], mode='L')
            name = 'frame_%d.png' % i

            save_path = os.path.join(folder, name)
            im.save(save_path)

    else:
        im = Image.fromarray(image[0, 0], mode='L')
        name = 'image_epoch_%d.png' % epoch_number
        save_path = os.path.join(location, name)
        im.save(save_path)


def run_erhan2009(project_variable, my_model, device):
    # based on "Visualizing Higher-Layer Features of a Deep Network" by Erhan et al. 2009
    all_outputs = []

    for l in range(len(project_variable.which_layers)):
        channels = []

        for c in range(len(project_variable.which_channels[l])):
            which_layer = project_variable.which_layers[l]
            which_channel = project_variable.which_channels[l][c]

            random_image_1 = torch.rand((1, 1, 50, 28, 28), requires_grad=True, device=device)
            a = np.random.randint(low=0, high=255, size=(1, 1, 50, 28, 28))
            a = a * 1.
            b = torch.Tensor(a).cuda(device)
            random_image = torch.nn.Parameter(b, requires_grad=True)


            # TODO: scale values accordingly
            # TODO: subtract mean and divide by std of avg image in training set

            optimizer = Adam([random_image], lr=0.01, weight_decay=0)
            mini_epochs = 50
            conv_output = None

            for i in range(1, mini_epochs+1):
                optimizer.zero_grad()
                x = random_image
                my_model.eval() # prevent gradients being computed for my_model

                if which_layer == 'conv1':
                    x = my_model.conv1(x, device)
                elif which_layer == 'conv2':
                    x = my_model.conv1(x, device)
                    x = my_model.max_pool_1(x)
                    x = torch.nn.functional.relu(x)
                    x = my_model.conv2(x, device)

                my_model.train()

                conv_output = x[0, which_channel]
                loss = -torch.mean(conv_output)
                loss.backward()
                optimizer.step()

                # print('erhan2009 loss mini-epoch %d: %f' % (i, loss.data.cpu()))

            # conv_output = np.array(conv_output.data.cpu(), dtype=np.uint8)
            # channels.append(conv_output)
            random_image = np.array(random_image.data.cpu(), dtype=np.uint8)
            channels.append(random_image)

        all_outputs.append(channels)

                ## use this for debugging
                # if i == mini_epochs:
                #     save_location = PP.erhan2009
                #     save_clip(conv_output, save_location, epoch)

    # return for tensorboard plotting
    return all_outputs


def run_zeiler2014(project_variable, input, my_model, device, epoch, which_conv, which_channel):
    # based on "Visualizing and Understanding Convolutional Networks" by Zeiler et al. 2014

    # which_channels contain the numbers of the channels that need to be visualized

    def get_deconv_model():
        model = deconv_3DTTN(which_conv)
        model.cuda(device)

        # copy the weights from trained model
        w1 = my_model.conv1.weight
        # w1 = torch.nn.Parameter(w1.permute(1, 0, 4, 3, 2))
        model.deconv1.weight = w1

        if which_conv == 'conv2':
            w2 = my_model.conv2.weight
            # w2 = torch.nn.Parameter(w2.permute(1, 0, 4, 3, 2))
            model.deconv2.weight = w2

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
    if which_conv == 'conv1':
        assert(0 < len(which_channel) < project_variable.num_out_channels[0]+1)
        for i in range(x3.shape[1]):
            if i not in which_channel:
                x3[0, i] = torch.nn.Parameter(torch.zeros(x3[0, i].shape))

    elif which_conv == 'conv2':
        assert (0 < len(which_channel) < project_variable.num_out_channels[1]+1)
        for i in range(x6.shape[1]):
            if i not in which_channel:
                x6[0, i] = torch.nn.Parameter(torch.zeros(x6[0, i].shape))

    # pass the activations as input to the deconv_model
    if which_conv == 'conv1':
        reconstruction = deconv_model(x3, switches)
    elif which_conv == 'conv2':
        reconstruction = deconv_model(x6, switches)
    else:
        reconstruction = None

    save_indices = [0, 9, 19, 29, 39, 49]

    if epoch in save_indices:
        # save reconstruction
        save_location = PP.zeiler2014
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

        folder = '%s_epoch_%d_n_%d' % (which_conv, epoch, number)
        save_path = os.path.join(save_location, folder)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for f in range(reconstruction.shape[2]):
            name = 'frame_%d.png' % f
            ultimate_path = os.path.join(save_path, name)

            im_as_arr = np.array(reconstruction[0][0][f].cpu().data, dtype=np.uint8)

            im = Image.fromarray(im_as_arr, mode='L')
            im.save(ultimate_path)

        path_input = os.path.join(save_path, 'og_image')
        if not os.path.exists(path_input):
            os.mkdir(path_input)

        for f in range(input.shape[2]):
            name = 'frame_%d.png' % f
            path = os.path.join(path_input, name)
            input_as_arr = np.array(input[0][0][f].cpu().data, dtype=np.uint8)
            input_img = Image.fromarray(input_as_arr, mode='L')

            input_img.save(path)

