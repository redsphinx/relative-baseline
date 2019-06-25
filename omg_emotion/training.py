import numpy as np
from relative_baseline.omg_emotion import saving
import torch
from tqdm import tqdm
from relative_baseline.omg_emotion import utils as U
from relative_baseline.omg_emotion import data_loading as DL
from relative_baseline.omg_emotion import visualization as VZ

# temporary for debugging
from .settings import ProjectVariable


def run(project_variable, all_data, my_model, my_optimizer, device):
    # all_data = np.array with the train datasplit depending
    # all_data = [data, labels] shape = (n, 2)

    loss_epoch, accuracy_epoch, confusion_epoch, nice_div, steps, full_labels, full_data = \
        U.initialize(project_variable, all_data)

    for ts in tqdm(range(steps)):
        # get part of data
        # data, labels = all_data[ts*project_variable.batch_size:(1+ts)*project_variable.batch_size][:]

        data, labels = DL.prepare_data(project_variable, full_data, full_labels, device, ts, steps, nice_div)

        my_optimizer.zero_grad()
        predictions = my_model(data)
        loss = U.calculate_loss(project_variable, predictions, labels)
        # TODO: fix THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=383 error=11 : invalid argument
        # retrain_graph=True because RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.
        # loss.backward()
        # try:
        #     loss.backward()
        # except RuntimeError:
        #     loss.backward(retain_graph=True)

        # This seems to solve the RuntimeError
        # loss.backward(retain_graph=True)
        loss.backward()

        # if project_variable.model_number == 3 and project_variable.current_epoch == 0 and ts == 0:
        #     print('retain_graph is True')
        #     loss.backward(retain_graph=True, create_graph=True)
        # else:
        #     loss.backward()

        my_optimizer.step()
        my_model.update()

        accuracy = U.calculate_accuracy(predictions, labels)
        confusion_epoch = U.confusion_matrix(confusion_epoch, predictions, labels)

        loss_epoch.append(float(loss))
        accuracy_epoch.append(float(accuracy))

    # save data
    loss = float(np.mean(loss_epoch))
    accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size + nice_div)
    confusion_flatten = U.flatten_confusion(confusion_epoch)

    # accuracy = float(np.mean(accuracy_epoch))

    if project_variable.save_data:
        saving.update_logs(project_variable, 'train', [loss, accuracy, confusion_flatten])

    print('epoch %d train, %s: %f, accuracy: %f ' % (project_variable.current_epoch,
                                                     project_variable.loss_function,
                                                     loss, accuracy))

    # save model
    if project_variable.save_model:
        saving.save_model(project_variable, my_model)

    # add things to writer
    project_variable.writer.add_scalars('metrics/train', {"loss": loss,
                                                        "accuracy": accuracy},
                                        project_variable.current_epoch)

    fig = VZ.plot_confusion_matrix(confusion_epoch)
    project_variable.writer.add_figure(tag='confusion/train', figure=fig, global_step=project_variable.current_epoch)