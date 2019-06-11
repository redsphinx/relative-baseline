import numpy as np
from PIL import Image
import os

import torch
from torch.nn import functional as F

from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import data_loading as D

project_variable = ProjectVariable(debug_mode=True)
project_variable.dataset = 'mnist'
project_variable.train = False
project_variable.val = True
project_variable.test = False

how_many = 200

# get  MNIST images
splits, data, labels = D.load_mnist(project_variable)
data = data[0].numpy()[0:how_many].reshape((how_many, 28, 28))

transformed_mnist = np.zeros((data.shape[0], 28, 28))
params = np.zeros((data.shape[0], 2, 3))

for i in range(data.shape[0]):
    # make transformation grids
    out_channels = 1
    # 1: theta = torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, 2, 3))))
    # 2: theta = torch.eye(3)[:2] + torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, 2, 3))), std=0.1)
    theta = torch.eye(3)[:2] + torch.nn.init.normal(torch.nn.Parameter(torch.zeros((out_channels, 2, 3))), std=0.5)
    theta[0][0][2] = 0
    theta[0][1][2] = 0
    grid = F.affine_grid(theta, torch.Size([out_channels, 1, 28, 28]))

    # apply transformations to MNIST
    d = np.expand_dims(data[i], 0)
    d = np.expand_dims(d, 0)
    transformed_mnist[i] = F.grid_sample(torch.Tensor(d), grid).detach().numpy().reshape((28, 28))

    params[i] = theta.detach().numpy().reshape((2, 3))

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
    p = params[i].flatten()
    line = '%d,%f,%f,%f,%f,%f,%f\n' % (i, p[0], p[1], p[2], p[3], p[4], p[5])
    with open(save_location_log, 'a') as my_file:
        my_file.write(line)
