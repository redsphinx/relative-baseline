import numpy as np
from relative_baseline.omg_emotion import saving
import torch
from tqdm import tqdm
from relative_baseline.omg_emotion import utils as U
from relative_baseline.omg_emotion import data_loading as DL
from relative_baseline.omg_emotion import visualization as VZ
from relative_baseline.omg_emotion import tensorboard_manager as TM

# temporary for debugging
from .settings import ProjectVariable


def run(project_variable, all_data, my_model, my_optimizer, device):
    # all_data = np.array with the train datasplit depending
    # all_data = [data, labels] shape = (n, 2)

    loss_epoch, accuracy_epoch, confusion_epoch, nice_div, steps, full_labels, full_data = \
        U.initialize(project_variable, all_data)

    for ts in tqdm(range(steps)):
        data, labels = DL.prepare_data(project_variable, full_data, full_labels, device, ts, steps, nice_div)

        my_optimizer.zero_grad()

        if project_variable.model_number in [3, 6, 71, 72, 73, 74, 75, 76, 77, 8]:
            predictions = my_model(data, device)
        else:
            predictions = my_model(data)
        loss = U.calculate_loss(project_variable, predictions, labels)
        # THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=383 error=11 : invalid argument
        loss.backward()

        my_optimizer.step()

        accuracy = U.calculate_accuracy(predictions, labels)
        confusion_epoch = U.confusion_matrix(confusion_epoch, predictions, labels)

        loss_epoch.append(float(loss))
        accuracy_epoch.append(float(accuracy))
        
    # save data
    loss = float(np.mean(loss_epoch))
    accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size + nice_div)
    confusion_flatten = U.flatten_confusion(confusion_epoch)

    # if project_variable.model_number in [2, 3, 71, 72, 73, 74, 75, 76, 77, 8]:
    #     TM.add_kernels(project_variable, my_model)
    #
    # if project_variable.model_number in [1, 2, 3, 71, 72, 73, 74, 75, 76, 77, 8]:
    #     TM.add_histograms(project_variable, my_model)
    #
    # if project_variable.model_number in [3, 71, 72, 73, 74, 75, 76, 77, 8]:
    #     TM.add_scalars(project_variable, my_model)

    if project_variable.save_data:
        saving.update_logs(project_variable, 'train', [loss, accuracy, confusion_flatten])

    print('epoch %d train, %s: %f, accuracy: %f ' % (project_variable.current_epoch,
                                                     project_variable.loss_function,
                                                     loss, accuracy))

    # save model
    if project_variable.save_model:
        if project_variable.current_epoch == project_variable.end_epoch - 1:
            saving.save_model(project_variable, my_model)

    # add things to writer
    TM.add_standard_info(project_variable, 'train', (loss, accuracy, confusion_epoch))

    # project_variable.writer.add_scalar('loss/train', loss, project_variable.current_epoch)
    # project_variable.writer.add_scalar('accuracy/train', accuracy, project_variable.current_epoch)
    # fig = VZ.plot_confusion_matrix(confusion_epoch, project_variable.dataset)
    # project_variable.writer.add_figure(tag='confusion/train', figure=fig, global_step=project_variable.current_epoch)
    #
