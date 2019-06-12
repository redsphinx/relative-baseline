import numpy as np
from PIL import Image
import os

import torch
from torch.nn import functional as F

from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import data_loading as D


def make_theta_matrix(s, r, x, y, out_channels=1):
    assert type(s) == type(r) == type(x) == type(y)
    if type(s) == float:
        s = torch.Tensor(s)
        r = torch.Tensor(r)
        x = torch.Tensor(x)
        y = torch.Tensor(y)
    theta = torch.zeros((out_channels, 2, 3))
    theta[0][0][0] = s * torch.cos(r)
    theta[0][0][1] = -s * torch.sin(r)
    theta[0][0][2] = x * s * torch.cos(r) - y * s * torch.sin(r)
    theta[0][1][0] = s * torch.sin(r)
    theta[0][1][1] = -s * torch.cos(r)
    theta[0][1][2] = x * s * torch.sin(r) + y * s * torch.cos(r)
    return theta


def random_sampling():
    project_variable = ProjectVariable(debug_mode=True)
    project_variable.dataset = 'mnist'
    project_variable.train = False
    project_variable.val = True
    project_variable.test = False

    how_many = 200
    num_params = 4
    # num_params = 6

    # get  MNIST images
    splits, data, labels = D.load_mnist(project_variable)
    data = data[0].numpy()[0:how_many].reshape((how_many, 28, 28))

    transformed_mnist = np.zeros((data.shape[0], 28, 28))

    if num_params == 6:
        params = np.zeros((data.shape[0], 2, 3))
    else:
        params = np.zeros((data.shape[0], 4))

    for i in range(data.shape[0]):
        # make transformation grids
        out_channels = 1
        # 1: theta = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, 2, 3))))
        # 2: theta = torch.eye(3)[:2] + torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, 2, 3))), std=0.1)
        # 3: theta = torch.eye(3)[:2] + torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, 2, 3))), std=0.5)
        # 4: theta = torch.eye(3)[:2] + torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, 2, 3))), std=0.5)
        #    theta[0][0][2] = 0
        #    theta[0][1][2] = 0
        # 5:
        scale = torch.nn.init.normal(torch.nn.Parameter(torch.zeros(1)))
        rotate = torch.nn.init.normal(torch.nn.Parameter(torch.zeros(1)))
        translate_x = torch.nn.init.normal(torch.nn.Parameter(torch.zeros(1)))
        translate_y = torch.nn.init.normal(torch.nn.Parameter(torch.zeros(1)))
        theta = make_theta_matrix(scale, rotate, translate_x, translate_y)

        # theta = torch.zeros((out_channels, 2, 3))
        # theta[0][0][0] = scale * torch.cos(rotate)
        # theta[0][0][1] = -scale * torch.sin(rotate)
        # theta[0][0][2] = translate_x * scale * torch.cos(rotate) - translate_y * scale * torch.sin(rotate)
        # theta[0][1][0] = scale * torch.sin(rotate)
        # theta[0][1][1] = -scale * torch.cos(rotate)
        # theta[0][1][2] = translate_x * scale * torch.sin(rotate) + translate_y * scale * torch.cos(rotate)
        grid = F.affine_grid(theta, torch.Size([out_channels, 1, 28, 28]))

        # apply transformations to MNIST
        d = np.expand_dims(data[i], 0)
        d = np.expand_dims(d, 0)
        transformed_mnist[i] = F.grid_sample(torch.Tensor(d), grid).detach().numpy().reshape((28, 28))

        if num_params == 6:
            params[i] = theta.detach().numpy().reshape((2, 3))
        else:
            params[i] = float(scale), float(rotate), float(translate_x), float(translate_y)

    # save data & visualize
    save_location_images = '/scratch/users/gabras/data/convttn3d_project/sanity_check_affine/images'
    save_location_log = '/scratch/users/gabras/data/convttn3d_project/sanity_check_affine/log.txt'

    if os.path.exists(save_location_log):
        os.remove(save_location_log)

    for i in range(data.shape[0]):
        # make image, before and after
        the_mode = 'L'
        canvas = Image.new(mode=the_mode, size=(28 + 5 + 28, 28))
        im1 = Image.fromarray(data[i].astype(np.uint8), mode=the_mode)
        im2 = Image.fromarray(transformed_mnist[i].astype(np.uint8), mode=the_mode)
        canvas.paste(im1, (0, 0))
        canvas.paste(im2, (28 + 5, 0))
        canvas_path = os.path.join(save_location_images, '%d.jpg' % i)
        canvas.save(canvas_path)

        # write line
        if num_params == 6:
            p = params[i].flatten()
            line = '%d,%f,%f,%f,%f,%f,%f\n' % (i, p[0], p[1], p[2], p[3], p[4], p[5])
            with open(save_location_log, 'a') as my_file:
                my_file.write(line)
        else:
            p = params[i]
            line = '%d,%f,%f,%f,%f\n' % (i, p[0], p[1], p[2], p[3])
            with open(save_location_log, 'a') as my_file:
                my_file.write(line)



def make_transformations(out_channels=1):
    scale = np.arange(-1, 1, 0.2)
    rotation = np.arange(-1, 1, 0.2)
    translate_x = np.arange(-1, 1, 0.2)
    translate_y = np.arange(-1, 1, 0.2)

    # scaling only
    for i in range(10):
        theta = make_theta_matrix(scale[i], rotation[i], translate_x[i], translate_y[i])
        grid = F.affine_grid(theta, torch.Size([out_channels, 1, 28, 28]))



