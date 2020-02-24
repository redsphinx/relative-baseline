import os
import numpy as np
from PIL import Image
import cv2 as cv

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn import functional as F

from relative_baseline.omg_emotion.xai_tools.misc_functions import preprocess_image, recreate_image, save_clip
from relative_baseline.omg_emotion.models import deconv_3DTTN, deconv_3D
import relative_baseline.omg_emotion.project_paths as PP
import relative_baseline.omg_emotion.data_loading as DL
from relative_baseline.omg_emotion import utils as U
from relative_baseline.omg_emotion import visualization as VZ


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


def make_affine_matrix(sc, ro, tx, ty, use_opencv=False):
    if use_opencv:
        matrix = np.zeros((2, 3))

        sc = float(sc.data.cpu())
        ro = float(ro.data.cpu())
        tx = float(tx.data.cpu())
        ty = float(ty.data.cpu())

        matrix[0, 0] = sc * np.cos(ro)
        matrix[0, 1] = -sc * np.sin(ro)
        matrix[0, 2] = tx * sc * np.cos(ro) - ty * sc * np.sin(ro)
        matrix[1, 0] = sc * np.sin(ro)
        matrix[1, 1] = sc * np.cos(ro)
        matrix[1, 2] = tx * sc * np.sin(ro) + ty * sc * np.cos(ro)

    else:
    
        matrix = torch.zeros((1, 2, 3))

        matrix[0, 0, 0] = sc * torch.cos(ro)
        matrix[0, 0, 1] = -sc * torch.sin(ro)
        matrix[0, 0, 2] = tx * sc * torch.cos(ro) - ty * sc * torch.sin(ro)
        matrix[0, 1, 0] = sc * torch.sin(ro)
        matrix[0, 1, 1] = sc * torch.cos(ro)
        matrix[0, 1, 2] = tx * sc * torch.sin(ro) + ty * sc * torch.cos(ro)

    return matrix


def create_next_frame(s, r, x, y, data, device, use_opencv=False):

    affine_matrix = make_affine_matrix(s, r, x, y, use_opencv)

    if use_opencv:
        # data = np.array(data.data.cpu(), dtype=np.uint8)
        # data = cv.cvtColor(data, cv.COLOR_RGB2GRAY)
        affine_matrix = VZ.make_pacman_frame(data, affine_matrix)
        return affine_matrix

    else:

        h_, w_ = data.shape
        affine_matrix = F.affine_grid(theta=affine_matrix, size=[1, 1, h_, w_])
        affine_matrix = affine_matrix.cuda(device)
        data = data.unsqueeze(0).unsqueeze(0)
        affine_matrix = F.grid_sample(data, affine_matrix)
        return affine_matrix[0][0]


def binarize(data):

    data = torch.clamp(input=data, min=0, max=255, out=None)

    max = torch.ones(data.shape) * 255
    min = torch.zeros(data.shape)

    torch.where(data > 0, max, min)

    return data


def normalize(data, use_opencv=False):
    z = 255
    y = 0
    if use_opencv:
        a = float(data.max())
        b = float(data.min())
    else:
        a = float(data.max().cpu())
        b = float(data.min().cpu())

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            c = data[i, j]
            data[i, j] = (c - a) * (z - y) / (b - a) + y

    return data


def run_erhan2009(project_variable, my_model, device):
    # based on "Visualizing Higher-Layer Features of a Deep Network" by Erhan et al. 2009
    all_outputs = []

    if project_variable.dataset == 'dhg':
        base_image = np.random.randint(low=250, high=255, size=(1, 1, 50, 28, 28))
    elif project_variable.dataset == 'mov_mnist':
        base_image = np.random.randint(low=250, high=255, size=(1, 1, 30, 28, 28))

    base_image = base_image * 1.
    # subtract mean and divide by std of avg image in training set
    if project_variable.dataset == 'dhg':
        mean, std = DL.get_mean_std_train_dhg()
    elif project_variable.dataset == 'mov_mnist':
        mean, std = DL.get_mean_std_train_mov_mnist()
    else:
        mean, std = None, None

    std[std == 0.] = 1.

    base_image = base_image - mean
    base_image = base_image / std

    # set negative values to zero
    base_image = base_image.clip(min=0)

    for l in range(len(project_variable.which_layers)):
        channels = []

        for c in range(len(project_variable.which_channels[l])):
            which_layer = project_variable.which_layers[l]
            which_channel = project_variable.which_channels[l][c]

            b = torch.Tensor(base_image).cuda(device)
            random_image = torch.nn.Parameter(b, requires_grad=True)

            optimizer = Adam([random_image], lr=0.05, weight_decay=0)
            mini_epochs = 50

            for i in range(1, mini_epochs+1):
                optimizer.zero_grad()
                x = random_image
                my_model.eval() # prevent gradients being computed for my_model
                
                if project_variable.model_number == 11:
                    if which_layer == 'conv1':
                        x = my_model.conv1(x, device)
                    elif which_layer == 'conv2':
                        x = my_model.conv1(x, device)
                        if 'zeiler2014' in project_variable.which_methods:
                            x, _ = my_model.max_pool_1(x)
                        else:
                            x = my_model.max_pool_1(x)
                        x = torch.nn.functional.relu(x)
                        x = my_model.conv2(x, device)
                else:
                    if which_layer == 'conv1':
                        x = my_model.conv1(x, )
                    elif which_layer == 'conv2':
                        x = my_model.conv1(x, )
                        if 'zeiler2014' in project_variable.which_methods:
                            x, _ = my_model.max_pool_1(x)
                        else:
                            x = my_model.max_pool_1(x)
                        x = torch.nn.functional.relu(x)
                        x = my_model.conv2(x, )

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

    # return for tensorboard plotting
    return all_outputs


def run_zeiler2014(project_variable, data_point, my_model, device):
    # based on "Visualizing and Understanding Convolutional Networks" by Zeiler et al. 2014

    # which_channels contain the numbers of the channels that need to be visualized individually

    # TODO: data_point has to be clip that maximizes the ONLY nonzero value allowed
    
    the_data = data_point
    all_outputs = []
    
    def get_deconv_model(which_layer):
        if project_variable.model_number == 11:
            model = deconv_3DTTN(which_layer)
        else:
            model = deconv_3D(which_layer)
        model.cuda(device)

        # copy the weights from trained model
        w1 = my_model.conv1.weight
        # w1 = torch.nn.Parameter(w1.permute(1, 0, 2, 4, 3))
        model.deconv1.weight = w1

        if which_layer == 'conv2':
            w2 = my_model.conv2.weight
            # w2 = torch.nn.Parameter(w2.permute(1, 0, 2, 4, 3))
            model.deconv2.weight = w2

        return model

    for l in range(len(project_variable.which_layers)):
        channels = []
        channels.append(np.array(the_data.data.cpu(), dtype=np.uint8))
        
        for c in range(len(project_variable.which_channels[l])):
            which_layer = project_variable.which_layers[l]
            which_channel = project_variable.which_channels[l][c]
            
            deconv_model = get_deconv_model(which_layer)
            # for the unpooling switches
            switches = []

            # pass the_data through my_model until the point of interest
            my_model.eval()
            
            if project_variable.model_number == 11:
                x1 = my_model.conv1(the_data, device)
                x2, _s = my_model.max_pool_1(x1)
                switches.append(_s)
                x3 = torch.nn.functional.relu(x2)
    
                if which_layer == 'conv2':
                    x4 = my_model.conv2(x3, device)
                    x5, _s = my_model.max_pool_2(x4)
                    switches.append(_s)
                    x6 = torch.nn.functional.relu(x5)
            else:
                x1 = my_model.conv1(the_data, )
                x2, _s = my_model.max_pool_1(x1)
                switches.append(_s)
                x3 = torch.nn.functional.relu(x2)

                if which_layer == 'conv2':
                    x4 = my_model.conv2(x3, )
                    x5, _s = my_model.max_pool_2(x4)
                    switches.append(_s)
                    x6 = torch.nn.functional.relu(x5)

            my_model.train()

            # set the irrelevant activations to zero
            if which_layer == 'conv1':

                # remove the bias
                bias = my_model.conv1.bias
                n_ = bias.shape[0]
                d_, h_, w_ = x3[0, 0].shape
                bias = bias.repeat_interleave(d_ * h_ * w_, axis=0)
                bias = bias.reshape((n_, d_, h_, w_))
                x3 = x3 - bias

                for i in range(x3.shape[1]):
                    if i != which_channel:
                        x3[0, i] = torch.nn.Parameter(torch.zeros(x3[0, i].shape))

                # set all non-max activations to zero, per time dimension
                for j in range(d_):
                    # get index of highest value and the value
                    highest_value = 0
                    ind_1, ind_2 = 0, 0
                    for m in range(h_):
                        for n in range(w_):
                            val = float(x3[0, which_channel, j, m, n].data.cpu())
                            if val > highest_value:
                                highest_value = val
                                ind_1 = m
                                ind_2 = n

                    # set everything to zero
                    x3[0, which_channel, j] = torch.nn.Parameter(torch.zeros(x3[0, which_channel, j].shape))

                    # set index with highest value
                    for m in range(h_):
                        for n in range(w_):
                            if ind_1 == m and ind_2 == n:
                                x3[0, which_channel, j, ind_1, ind_2] = highest_value

            elif which_layer == 'conv2':

                # remove the bias
                bias = my_model.conv2.bias
                n_ = bias.shape[0]
                d_, h_, w_ = x6[0, 0].shape
                bias = bias.repeat_interleave(d_ * h_ * w_, axis=0)
                bias = bias.reshape((n_, d_, h_, w_))
                x6 = x6 - bias

                for i in range(x6.shape[1]):
                    if i != which_channel:
                        x6[0, i] = torch.nn.Parameter(torch.zeros(x6[0, i].shape))

                # set all non-max activations to zero, per time dimension
                for j in range(x6[0, which_channel].shape[0]):
                    # get index of highest value and the value
                    highest_value = 0
                    ind_1, ind_2 = 0, 0
                    for m in range(h_):
                        for n in range(w_):
                            val = float(x6[0, which_channel, j, m, n].data.cpu())
                            if val > highest_value:
                                highest_value = val
                                ind_1 = m
                                ind_2 = n

                    # set everything to zero
                    x6[0, which_channel, j] = torch.nn.Parameter(torch.zeros(x6[0, which_channel, j].shape))

                    # set index with highest value
                    for m in range(h_):
                        for n in range(w_):
                            if ind_1 == m and ind_2 == n:
                                x6[0, which_channel, j, ind_1, ind_2] = highest_value
                                # break
            
            # pass the activations as input to the deconv_model
            if which_layer == 'conv1':
                reconstruction = deconv_model(x3, switches)
            elif which_layer == 'conv2':
                reconstruction = deconv_model(x6, switches)
            else:
                reconstruction = None

            reconstruction = torch.clamp(input=reconstruction, min=0, max=255, out=None)
            reconstruction = np.array(reconstruction.data.cpu(), dtype=np.uint8)
            channels.append(reconstruction)

        all_outputs.append(channels)
    return all_outputs


def our_gradient_method(project_variable, data_point, my_model, device, basic_mode=True, use_opencv=False):
    # if basic_mode is False, apply the temporal motion from the learned transformations
    all_outputs = []
    data = None
    trafo_per_filter = 4

    for l in range(len(project_variable.which_layers)):
        channels = []

        for c in range(len(project_variable.which_channels[l])):
            which_layer = project_variable.which_layers[l]
            which_channel = project_variable.which_channels[l][c]

            data = data_point
            data = torch.nn.Parameter(data, requires_grad=True)

            x1 = my_model.conv1(data, device)
            x2 = my_model.max_pool_1(x1)
            x3 = torch.nn.functional.relu(x2)

            if which_layer == 'conv2':
                x4 = my_model.conv2(x3, device)
                x5 = my_model.max_pool_2(x4)
                x6 = torch.nn.functional.relu(x5)

            if which_layer == 'conv1':
                _, ch, d, h, w = x3.shape
            else:
                _, ch, d, h, w = x6.shape

            highest_value = 0
            ind_1, ind_2 = 0, 0
            for m in range(h):
                for n in range(w):
                    if which_layer == 'conv1':
                        val = float(x3[0, which_channel, 0, m, n].data.cpu())
                        # val = float(x1[0, which_channel, 0, m, n].data.cpu())
                    else:
                        val = float(x6[0, which_channel, 0, m, n].data.cpu())
                        # val = float(x4[0, which_channel, 0, m, n].data.cpu())
                    if val > highest_value:
                        highest_value = val
                        ind_1 = m
                        ind_2 = n

            if which_layer == 'conv1':
                x3[0, which_channel, 0, ind_1, ind_2].backward()
                # x1[0, which_channel, 0, ind_1, ind_2].backward()
            else:
                x6[0, which_channel, 0, ind_1, ind_2].backward()
                # x4[0, which_channel, 0, ind_1, ind_2].backward()

            image_grad = data.grad

            final = image_grad[0, 0, 0] * data[0, 0, 0]

            if basic_mode:
                final = normalize(final)
                final = final.unsqueeze(0)
                final = np.array(final.data.cpu(), dtype=np.uint8)

                image_grad = normalize(image_grad[0, 0, 0])
                image_grad = image_grad.unsqueeze(0)
                image_grad = np.array(image_grad.data.cpu(), dtype=np.uint8)
                channels.append([image_grad, final])
            else:
                
                all_finals = []
                if use_opencv:
                    final = np.array(final.data.cpu(), dtype=np.uint8)

                all_finals.append(final)
                
                # get transformations
                
                if which_layer == 'conv1':

                    for trafo in range(trafo_per_filter):
                        s = my_model.conv1.scale[trafo, which_channel]
                        r = my_model.conv1.rotate[trafo, which_channel]
                        x = my_model.conv1.translate_x[trafo, which_channel]
                        y = my_model.conv1.translate_y[trafo, which_channel]
                        # apply them on the final image
                        next_final = create_next_frame(s, r, x, y, all_finals[trafo], device, use_opencv)
                        all_finals.append(next_final)
                    
                else:
                    # all_finals.append(final)
                    for trafo in range(trafo_per_filter):
                        s = my_model.conv2.scale[trafo, which_channel]
                        r = my_model.conv2.rotate[trafo, which_channel]
                        x = my_model.conv2.translate_x[trafo, which_channel]
                        y = my_model.conv2.translate_y[trafo, which_channel]
                        # apply them on the final image
                        next_final = create_next_frame(s, r, x, y, all_finals[trafo], device, use_opencv)
                        all_finals.append(next_final)

                channels.append(all_finals)

        all_outputs.append(channels)

    data = data[0, 0, 0]
    data = data.unsqueeze(0)
    data = np.array(data.data.cpu(), dtype=np.uint8)
    
    if not basic_mode:
        processed_outputs = []
        
        for l in range(len(project_variable.which_layers)):
            channels = []

            for c in range(len(project_variable.which_channels[l])):
                all_finals = []
                
                for t in range(trafo_per_filter+1):
                    if use_opencv:
                        processed_final = all_outputs[l][c][t]
                        processed_final = normalize(processed_final, use_opencv)
                        processed_final = np.expand_dims(processed_final, 0)
                        processed_final = np.array(processed_final, dtype=np.uint8)
                    else:
                        processed_final = all_outputs[l][c][t]
                        processed_final = normalize(processed_final)
                        processed_final = processed_final.unsqueeze(0)
                        processed_final = np.array(processed_final.data.cpu(), dtype=np.uint8)

                    all_finals.append(processed_final)

                channels.append(all_finals)

            processed_outputs.append(channels)

        all_outputs = processed_outputs

    return data, all_outputs


def our_gradient_method_regular_3d(project_variable, data_point, my_model):
    all_outputs = []
    the_data = []
    data = None

    for l in range(len(project_variable.which_layers)):
        channels = []

        for c in range(len(project_variable.which_channels[l])):
            which_layer = project_variable.which_layers[l]
            which_channel = project_variable.which_channels[l][c]

            kernel = []

            for k in range(project_variable.k_shape[0]):
            
                data = data_point
                data = torch.nn.Parameter(data, requires_grad=True)
                
                if l == 0 and c == 0:
                    d_ = data[0, 0, k]
                    d_ = d_.unsqueeze(0)
                    d_ = np.array(d_.data.cpu(), dtype=np.uint8)
                    the_data.append(d_)

                x1 = my_model.conv1(data)
                x2 = my_model.max_pool_1(x1)
                x3 = torch.nn.functional.relu(x2)
    
                if which_layer == 'conv2':
                    x4 = my_model.conv2(x3)
                    x5 = my_model.max_pool_2(x4)
                    x6 = torch.nn.functional.relu(x5)
    
                if which_layer == 'conv1':
                    _, ch, d, h, w = x3.shape
                else:
                    _, ch, d, h, w = x6.shape
    
                highest_value = 0
                ind_1, ind_2 = 0, 0
                for m in range(h):
                    for n in range(w):
                        if which_layer == 'conv1':
                            val = float(x3[0, which_channel, k, m, n].data.cpu())
                            # val = float(x1[0, which_channel, 0, m, n].data.cpu())
                        else:
                            val = float(x6[0, which_channel, k, m, n].data.cpu())
                            # val = float(x4[0, which_channel, 0, m, n].data.cpu())
                        if val > highest_value:
                            highest_value = val
                            ind_1 = m
                            ind_2 = n
    
                if which_layer == 'conv1':
                    x3[0, which_channel, k, ind_1, ind_2].backward()
                    # x1[0, which_channel, 0, ind_1, ind_2].backward()
                else:
                    x6[0, which_channel, k, ind_1, ind_2].backward()
                    # x4[0, which_channel, 0, ind_1, ind_2].backward()
    
                image_grad = data.grad
    
                final = image_grad[0, 0, k] * data[0, 0, k]
                final = normalize(final)
                final = final.unsqueeze(0)
                final = np.array(final.data.cpu(), dtype=np.uint8)
                
                kernel.append(final)
            
            channels.append(kernel)
        
        all_outputs.append(channels)

    return the_data, all_outputs


def gradient_method(project_variable, data_point, my_model, device, basic_mode=True, use_opencv=False):

    if project_variable.model_number == 11:
        return our_gradient_method(project_variable, data_point, my_model, device, basic_mode, use_opencv)
    else:
        return our_gradient_method_regular_3d(project_variable, data_point, my_model)

