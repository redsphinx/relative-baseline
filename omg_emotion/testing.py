import numpy as np
from relative_baseline.omg_emotion import saving
import torch
from tqdm import tqdm
from relative_baseline.omg_emotion import utils as U
from relative_baseline.omg_emotion import data_loading as DL
from relative_baseline.omg_emotion import tensorboard_manager as TM


def run(project_variable, all_data, my_model, device):
    # all_data = np.array with the train datasplit depending
    # all_data = [data, labels] shape = (n, 2)
    # device is string

    loss_epoch, accuracy_epoch, confusion_epoch, nice_div, steps, full_labels, full_data = \
        U.initialize(project_variable, all_data)

    if project_variable.use_dali:
        steps = 0

        for i, data_and_labels in enumerate(all_data):

            data = data_and_labels[0]['data']
            labels = data_and_labels[0]['labels']

            # transpose data
            data = data.permute(0, 4, 1, 2, 3)
            # convert to floattensor
            data = data.type(torch.float32)
            labels = labels.type(torch.long)
            labels = labels.flatten()
            labels = labels - 1

            my_model.eval()
            with torch.no_grad():
                # my_optimizer.zero_grad()
                # predictions = my_model.forward(data)
                if project_variable.model_number in [3, 6, 71, 72, 73, 74, 75, 76, 77, 8, 10, 11, 14, 15, 17, 18, 19, 20]:
                    predictions = my_model(data, device)
                elif project_variable.model_number in [16]:
                    predictions = my_model(data, device, project_variable.genome)
                else:
                    predictions = my_model(data)
                # print(predictions)
                loss = U.calculate_loss(project_variable, predictions, labels)
                # print('loss raw: %s' % str(loss))
                loss = loss.detach()
                # loss.backward()
            my_model.train()
            steps = steps + 1
            
    else:
        for ts in tqdm(range(steps)):

            # get part of data
            data, labels = DL.prepare_data(project_variable, full_data, full_labels, device, ts, steps, nice_div)

            my_model.eval()
            with torch.no_grad():
                # my_optimizer.zero_grad()
                # predictions = my_model.forward(data)
                if project_variable.model_number in [3, 6, 71, 72, 73, 74, 75, 76, 77, 8, 10, 11, 14, 15, 16]:
                    predictions = my_model(data, device)
                else:
                    predictions = my_model(data)
                # print(predictions)
                loss = U.calculate_loss(project_variable, predictions, labels)
                # print('loss raw: %s' % str(loss))
                loss = loss.detach()
                # loss.backward()
            my_model.train()

            accuracy = U.calculate_accuracy(predictions, labels)
            confusion_epoch = U.confusion_matrix(confusion_epoch, predictions, labels)

            loss_epoch.append(float(loss))
            accuracy_epoch.append(float(accuracy))

    # save data
    # print('loss epoch: ', loss_epoch)
    loss = float(np.mean(loss_epoch))
    if project_variable.use_dali:
        accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size)
    else:
        accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size + nice_div)

    confusion_flatten = U.flatten_confusion(confusion_epoch)

    if project_variable.save_data:
        saving.update_logs(project_variable, 'test', [loss, accuracy, confusion_flatten])

    print('epoch %d test, %s: %f, accuracy: %f ' % (project_variable.current_epoch,
                                                   project_variable.loss_function,
                                                   loss, accuracy))

    # project_variable.writer.add_scalar('loss/test', loss, project_variable.current_epoch)
    # project_variable.writer.add_scalar('accuracy/test', accuracy, project_variable.current_epoch)
    # fig = VZ.plot_confusion_matrix(confusion_epoch, project_variable.dataset)
    # project_variable.writer.add_figure(tag='confusion/test', figure=fig, global_step=project_variable.current_epoch)
    TM.add_standard_info(project_variable, 'test', (loss, accuracy, confusion_epoch))

    if project_variable.use_dali:
            # reset so that we evaluate on the same data
            all_data.reset()

    if project_variable.inference_in_batches[0]:
        return accuracy
