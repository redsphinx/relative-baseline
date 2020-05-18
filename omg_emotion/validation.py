import numpy as np
from relative_baseline.omg_emotion import saving
import torch
from tqdm import tqdm
from relative_baseline.omg_emotion import utils as U
from relative_baseline.omg_emotion import data_loading as DL
from relative_baseline.omg_emotion import tensorboard_manager as TM

# temporary for debugging
# from .settings import ProjectVariable

# for debugging
# project_variable = ProjectVariable()
# from tensorboardX import SummaryWriter
# project_variable.writer = SummaryWriter()

def run(project_variable, all_data, my_model, device):
    # all_data = np.array with the train datasplit depending
    # all_data = [data, labels] shape = (n, 2)
    # device is string

    loss_epoch, accuracy_epoch, confusion_epoch, nice_div, steps, full_labels, full_data = \
        U.initialize(project_variable, all_data)

    if project_variable.use_dali:
        if project_variable.dataset == 'jester':
            the_iterator = DL.get_jester_iter('val', project_variable)
        elif project_variable.dataset == 'ucf101':
            the_iterator = DL.get_ucf101_iter('val', project_variable)
        else:
            the_iterator = None
        steps = 0

        for i, data_and_labels in tqdm(enumerate(the_iterator)):

            data = data_and_labels[0]['data']
            labels = data_and_labels[0]['labels']

            # transpose data
            data = data.permute(0, 4, 1, 2, 3)
            # convert to floattensor
            data = data.type(torch.float32)
            data = data / 255
            data[:, 0, :, :, :] = (data[:, 0, :, :, :] - 0.485) / 0.229
            data[:, 1, :, :, :] = (data[:, 1, :, :, :] - 0.456) / 0.224
            data[:, 2, :, :, :] = (data[:, 2, :, :, :] - 0.406) / 0.225

            labels = labels.type(torch.long)
            labels = labels.flatten()
            if project_variable.dataset == 'jester':
                labels = labels - 1

            my_model.eval()
            with torch.no_grad():
                # my_optimizer.zero_grad()
                # predictions = my_model.forward(data)
                if project_variable.model_number in [3, 6, 71, 72, 73, 74, 75, 76, 77, 8, 10, 11, 14, 15, 17, 18, 19, 20, 23, 24]:
                    if project_variable.model_number in [23]:
                        aux1, aux2, predictions = my_model(data, device, None, False)
                        assert aux1 is None and aux2 is None
                    else:
                        predictions = my_model(data, device)

                elif project_variable.model_number in [25]:
                        aux1, aux2, predictions = my_model(data, None, False)
                        assert aux1 is None and aux2 is None

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

            accuracy = U.calculate_accuracy(predictions, labels)
            confusion_epoch = U.confusion_matrix(confusion_epoch, predictions, labels)

            loss_epoch.append(float(loss))
            accuracy_epoch.append(float(accuracy))

            steps = steps + 1

    else:

        for ts in tqdm(range(steps)):

            # get part of data
            data, labels = DL.prepare_data(project_variable, full_data, full_labels, device, ts, steps, nice_div)

            my_model.eval()
            with torch.no_grad():
                # my_optimizer.zero_grad()
                # predictions = my_model.forward(data)
                if project_variable.model_number in [3, 6, 71, 72, 73, 74, 75, 76, 77, 8, 10, 11, 14, 15, 16, 20, 23]:
                    if project_variable.model_number == 23:
                        aux1, aux2, predictions = my_model(data, device, None, False)
                        assert aux1 is None and aux2 is None
                    else:
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
        saving.update_logs(project_variable, 'val', [loss, accuracy, confusion_flatten])

    print('epoch %d val, %s: %f, accuracy: %f ' % (project_variable.current_epoch,
                                                   project_variable.loss_function,
                                                   loss, accuracy))

    TM.add_standard_info(project_variable, 'val', (loss, accuracy, confusion_epoch))

    # save_epochs = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    # save_epochs = [2]
    # save_epochs = np.arange(project_variable.end_epoch)
    save_epochs = [9 + (i * 10) for i in range(project_variable.end_epoch // 10)]

    # which_datapoint = 15
    which_datapoint = 1

    if project_variable.inference_only_mode:

        TM.add_xai(project_variable, my_model, device, data_point=data[which_datapoint].unsqueeze(0))
        TM.add_histograms_srxy(project_variable, my_model)
        TM.add_text_srxy_per_channel(project_variable, my_model)
    else:
        if project_variable.do_xai and project_variable.current_epoch in save_epochs:

            TM.add_xai(project_variable, my_model, device, data_point=data[which_datapoint].unsqueeze(0))

    # if project_variable.use_dali:
    #         # reset so that we evaluate on the same data
    #         all_data.reset()

    # project_variable.writer.add_scalar('loss/val', loss, project_variable.current_epoch)
    # project_variable.writer.add_scalar('accuracy/val', accuracy, project_variable.current_epoch)
    # fig = VZ.plot_confusion_matrix(confusion_epoch, project_variable.dataset)
    # project_variable.writer.add_figure(tag='confusion/val', figure=fig, global_step=project_variable.current_epoch)
    if project_variable.early_stopping:
        return accuracy, loss
    else:
        return accuracy

