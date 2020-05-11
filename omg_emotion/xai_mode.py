import numpy as np
from relative_baseline.omg_emotion import saving
import torch
from tqdm import tqdm
from relative_baseline.omg_emotion import utils as U
from relative_baseline.omg_emotion import data_loading as DL
from relative_baseline.omg_emotion import tensorboard_manager as TM
from relative_baseline.omg_emotion.xai_tools import layer_visualization as layer_vis
from relative_baseline.omg_emotion.visualization import plot_srxy, save_kernels

# temporary for debugging
# from .settings import ProjectVariable


def run(project_variable, my_model, device):

    assert project_variable.use_dali

    the_iterator = DL.get_jester_iter(None, project_variable)

    # loss_epoch, accuracy_epoch, confusion_epoch, nice_div, steps, full_labels, full_data = \
    #     U.initialize(project_variable, all_data)

    # if project_variable.use_dali:
    #     the_iterator = DL.get_jester_iter('val', project_variable)
    #     steps = 0

    all_predictions = []
    all_labels = []


    for i, data_and_labels in tqdm(enumerate(the_iterator)):

        prediction = None

        data = data_and_labels[0]['data']
        labels = data_and_labels[0]['labels']

        # transpose data
        data = data.permute(0, 4, 1, 2, 3)
        og_data = data.clone()
        # convert to floattensor
        data = data.type(torch.float32)
        data = data / 255
        data[:, 0, :, :, :] = (data[:, 0, :, :, :] - 0.485) / 0.229
        data[:, 1, :, :, :] = (data[:, 1, :, :, :] - 0.456) / 0.224
        data[:, 2, :, :, :] = (data[:, 2, :, :, :] - 0.406) / 0.225

        labels = labels.type(torch.long)
        labels = labels.flatten()
        labels = labels - 1

        my_model.eval()
        if project_variable.model_number == 20:
            prediction = my_model(data, device)
        elif project_variable.model_number == 23:
            aux1, aux2, prediction = my_model(data, device, None, False)

        my_model.zero_grad()

        prediction = np.array(prediction[0].data.cpu()).argmax()
        labels = int(labels.data.cpu())
        all_predictions.append(prediction)
        all_labels.append(labels)

        tmp, kernel_vis, srxy_params = layer_vis.visualize_resnet18(project_variable, og_data, data, my_model, device,
                                                                    kernel_visualizations=True, srxy_plots=True)
        del tmp

        info = (project_variable.model_number, project_variable.load_model)


        # TODO: modify and implement defs below
        # save kernel_vis as gifs
        save_kernels(kernel_vis, og_data, info)
        # save srxy params as graphs
        # plot_srxy(srxy_params, 'all', 2)


