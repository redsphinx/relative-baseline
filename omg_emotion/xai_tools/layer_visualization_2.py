import os
import numpy as np
from PIL import Image

import torch
from torch.optim import Adam
from torch.autograd import Variable

from relative_baseline.omg_emotion.xai_tools.misc_functions import preprocess_image, recreate_image, save_clip


def run(my_model, device, epoch):

    processed_image = torch.rand((1, 1, 50, 28, 28), requires_grad=True, device=device)



    # random_image = np.uint8(np.random.uniform(150, 180, (1, 1, 50, 28, 28)))
    #
    # processed_image = torch.from_numpy(random_image).float()
    # processed_image = Variable(processed_image, requires_grad=True)
    # processed_image = processed_image.cuda(device)
    # processed_image = preprocess_image(random_image, False)

    # Define optimizer for the image
    optimizer = Adam([processed_image], lr=0.01, weight_decay=0)

    mini_epochs = 20

    for i in range(1, mini_epochs+1):
        optimizer.zero_grad()

        # pass random image through layer
        x = processed_image
        x = my_model.conv1(x, device)

        # get output from 1st filter in layer
        conv_output = x[0, 0]

        loss = -torch.mean(conv_output)
        # print('Iteration: %s, Loss: %0.2f' % (str(i), float(loss.data.cpu())))
        # Backward
        loss.backward()
        # Update image
        optimizer.step()

        if i == mini_epochs:
            save_location = '/home/gabras/deployed/relative_baseline/omg_emotion/images/layer_visualization'
            save_clip(conv_output, save_location, epoch)



