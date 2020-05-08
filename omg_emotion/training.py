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
    # if project_variable.use_dali, all_data = train iterator
    # all_data = np.array with the train datasplit depending
    # all_data = [data, labels] shape = (n, 2)

    loss_epoch, accuracy_epoch, confusion_epoch, nice_div, steps, full_labels, full_data = \
        U.initialize(project_variable, all_data)

    if project_variable.use_clr:
        clr_scheduler = torch.optim.lr_scheduler.CyclicLR(my_optimizer,
                                                          base_lr=project_variable.learning_rate/10,
                                                          max_lr=project_variable.learning_rate*10,
                                                          step_size_up=steps/2)

    data_for_vis = None

    if project_variable.use_dali:
        # https://towardsdatascience.com/nvidia-dali-speeding-up-pytorch-876c80182440
        # to reduce GPU memory usage
        the_iterator = DL.get_jester_iter('train', project_variable)

        steps = 0

        for i, data_and_labels in enumerate(the_iterator):

            print('\n'
                  '\n'
                  '\n'
                  'STEP %d'
                  '\n'
                  '\n'
                  '\n' % steps)
            data = data_and_labels[0]['data']
            labels = data_and_labels[0]['labels']

            # transpose data
            data = data.permute(0, 4, 1, 2, 3)
            # convert to floattensor
            data = data.type(torch.float32)

            # data shape: b, c, d, h, w
            data = data / 255
            data[:, 0, :, :, :] = (data[:, 0, :, :, :] - 0.485) / 0.229
            data[:, 1, :, :, :] = (data[:, 1, :, :, :] - 0.456) / 0.224
            data[:, 2, :, :, :] = (data[:, 2, :, :, :] - 0.406) / 0.225

            # data = (data/255 - project_variable.imnet_mean) / project_variable.imnet_stds
            labels = labels.type(torch.long)
            labels = labels.flatten()
            labels = labels - 1

            my_optimizer.zero_grad()

            if project_variable.model_number in [3, 6, 71, 72, 73, 74, 75, 76, 77, 8, 10, 11, 14, 15, 17, 18, 19, 20, 23]:
                predictions = my_model(data, device)
                if project_variable.model_number == 23:
                    aux1, aux2, predictions = my_model(data, device)
                    # https://discuss.pytorch.org/t/why-auxiliary-logits-set-to-false-in-train-mode/40705
                    # TODO: implement this in utils.py
                    # loss1 = criterion(outputs, target)
                    # loss2 = criterion(aux1, target)
                    # loss3 = criterion(aux2, target)
                    # loss = loss1 + 0.3 * (loss2 + loss3)
            elif project_variable.model_number in [16]:
                predictions = my_model(data, device, project_variable.genome)
            else:
                predictions = my_model(data)

            loss = U.calculate_loss(project_variable, predictions, labels)
            # THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=383 error=11 : invalid argument
            loss.backward()

            my_optimizer.step()

            if project_variable.use_clr:
                clr_scheduler.step()
                # print('CLR LR: ', clr_scheduler.get_lr())

            # my_optimizer.step()

            accuracy = U.calculate_accuracy(predictions, labels)
            confusion_epoch = U.confusion_matrix(confusion_epoch, predictions, labels)

            loss_epoch.append(float(loss))
            accuracy_epoch.append(float(accuracy))

            steps = steps + 1

    else:
        assert steps is not None
        for ts in tqdm(range(steps)):
            data, labels = DL.prepare_data(project_variable, full_data, full_labels, device, ts, steps, nice_div)

            my_optimizer.zero_grad()

            if project_variable.model_number in [3, 6, 71, 72, 73, 74, 75, 76, 77, 8, 10, 11, 14, 15, 16, 20, 23]:
                predictions = my_model(data, device)
            else:
                predictions = my_model(data)

            loss = U.calculate_loss(project_variable, predictions, labels)
            # THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=383 error=11 : invalid argument
            loss.backward()

            my_optimizer.step()

            if project_variable.use_clr:
                clr_scheduler.step()
                # print('CLR LR: ', clr_scheduler.get_lr())

            # my_optimizer.step()

            accuracy = U.calculate_accuracy(predictions, labels)
            confusion_epoch = U.confusion_matrix(confusion_epoch, predictions, labels)

            loss_epoch.append(float(loss))
            accuracy_epoch.append(float(accuracy))

    # save data
    loss = float(np.mean(loss_epoch))


    if project_variable.use_dali:
        accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size)
    else:
        accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size + nice_div)

    confusion_flatten = U.flatten_confusion(confusion_epoch)

    # TM.add_kernels(project_variable, my_model)

    if project_variable.model_number in [1, 2, 3, 71, 72, 73, 74, 75, 76, 77, 8]:
        TM.add_histograms(project_variable, my_model)

    if project_variable.model_number in [3, 71, 72, 73, 74, 75, 76, 77, 8]:
        TM.add_scalars(project_variable, my_model)

    if project_variable.save_data:
        saving.update_logs(project_variable, 'train', [loss, accuracy, confusion_flatten])

    print('epoch %d train, %s: %f, accuracy: %f ' % (project_variable.current_epoch,
                                                     project_variable.loss_function,
                                                     loss, accuracy))

    # save model
    if project_variable.save_model:
        if project_variable.stop_at_collapse or project_variable.early_stopping:
            # if model collapses, we don't want to save it. collapse usually happens pretty early in the training.
            # if we're stopping early we want to be able to choose the best model, so we need all the models
            # the best model will be kept at the end of the experiment, the other models will be deleted
            saving.save_model(project_variable, my_model)

        else:
            if project_variable.current_epoch == project_variable.end_epoch - 1:
                saving.save_model(project_variable, my_model)

    # add things to writer
    TM.add_standard_info(project_variable, 'train', (loss, accuracy, confusion_epoch))


    # TM.add_temporal_visualizations(project_variable, my_model)


    # project_variable.writer.add_scalar('loss/train', loss, project_variable.current_epoch)
    # project_variable.writer.add_scalar('accuracy/train', accuracy, project_variable.current_epoch)
    # fig = VZ.plot_confusion_matrix(confusion_epoch, project_variable.dataset)
    # project_variable.writer.add_figure(tag='confusion/train', figure=fig, global_step=project_variable.current_epoch)
    if project_variable.nas or project_variable.stop_at_collapse:
        return accuracy, U.has_collapsed(confusion_epoch)
    else:
        return accuracy
