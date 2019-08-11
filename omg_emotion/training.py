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

        if project_variable.model_number == 3:
            predictions = my_model(data, device)
        else:
            predictions = my_model(data)
        loss = U.calculate_loss(project_variable, predictions, labels)
        # FIX: THCudaCheck FAIL file=/pytorch/aten/src/THC/THCGeneral.cpp line=383 error=11 : invalid argument
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

    # accuracy = float(np.mean(accuracy_epoch))

    # if conv3d model, add kernels from first layer to tensorboard
    if project_variable.model_number in [2, 3]:
        kernel = my_model.conv1.weight.data
        kernel = kernel.transpose(1, 2)

        for k in range(kernel.shape[0]):

            new_k = kernel[k].unsqueeze(0)

            project_variable.writer.add_video(tag='kernels/%d' % k, vid_tensor=new_k,
                                          global_step=project_variable.current_epoch, fps=2)
    elif project_variable.model_number == 1:
        kernel = my_model.conv1.weight.data

        for k in range(kernel.shape[0]):

            new_k = kernel[k]
            project_variable.writer.add_image(tag='kernels/%d' % k, img_tensor=new_k,
                                          global_step=project_variable.current_epoch)

        print('asdf')

    # plot learned s, r, x, y parameters
    if project_variable.theta_init is None and project_variable.model_number == 3:
        # data as histogram
        project_variable.writer.add_histogram('conv1.weight/scale', my_model.conv1.scale, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv1.weight/rotate', my_model.conv1.rotate, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv1.weight/translate_x', my_model.conv1.translate_x, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv1.weight/translate_y', my_model.conv1.translate_y, project_variable.current_epoch)

        project_variable.writer.add_histogram('conv1.bias', my_model.conv1.bias, project_variable.current_epoch)

        project_variable.writer.add_histogram('conv2.weight/scale', my_model.conv2.scale, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv2.weight/rotate', my_model.conv2.rotate, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv2.weight/translate_x', my_model.conv2.translate_x, project_variable.current_epoch)
        project_variable.writer.add_histogram('conv2.weight/translate_y', my_model.conv2.translate_y, project_variable.current_epoch)

        project_variable.writer.add_histogram('conv2.bias', my_model.conv2.bias, project_variable.current_epoch)

        s = my_model.conv1.scale.data
        r = my_model.conv1.rotate.data
        x = my_model.conv1.translate_x.data
        y = my_model.conv1.translate_y.data

        for i in range(s.shape[1]):
            project_variable.writer.add_scalars('k%d/scale' %i,
                                                {"t0": s[0, i], "t1": s[1, i], "t2": s[2, i], "t3": s[3, i]},
                                                project_variable.current_epoch)
            project_variable.writer.add_scalars('k%d/rotate' % i,
                                                {"t0": r[0, i], "t1": r[1, i], "t2": r[2, i], "t3": r[3, i]},
                                                project_variable.current_epoch)
            project_variable.writer.add_scalars('k%d/translate_x' % i,
                                                {"t0": x[0, i], "t1": x[1, i], "t2": x[2, i], "t3": x[3, i]},
                                                project_variable.current_epoch)
            project_variable.writer.add_scalars('k%d/translate_y' % i,
                                                {"t0": y[0, i], "t1": y[1, i], "t2": y[2, i], "t3": y[3, i]},
                                                project_variable.current_epoch)

    if project_variable.model_number == 3:
        project_variable.writer.add_histogram('first_weight/conv1', my_model.conv1.first_weight, project_variable.current_epoch)
        project_variable.writer.add_histogram('first_weight/conv2', my_model.conv2.first_weight, project_variable.current_epoch)

    if project_variable.model_number in [1, 2, 3]:
        project_variable.writer.add_histogram('fc1/weight', my_model.fc1.weight, project_variable.current_epoch)
        project_variable.writer.add_histogram('fc2/weight', my_model.fc2.weight, project_variable.current_epoch)
        project_variable.writer.add_histogram('fc3/weight', my_model.fc3.weight, project_variable.current_epoch)

        project_variable.writer.add_histogram('fc1/bias', my_model.fc1.bias, project_variable.current_epoch)
        project_variable.writer.add_histogram('fc2/bias', my_model.fc2.bias, project_variable.current_epoch)
        project_variable.writer.add_histogram('fc3/bias', my_model.fc3.bias, project_variable.current_epoch)
    

    if project_variable.save_data:
        saving.update_logs(project_variable, 'train', [loss, accuracy, confusion_flatten])

    print('epoch %d train, %s: %f, accuracy: %f ' % (project_variable.current_epoch,
                                                     project_variable.loss_function,
                                                     loss, accuracy))

    # save model
    # TODO: DEBUG
    if project_variable.save_model:
        if project_variable.current_epoch == project_variable.end_epoch:
            saving.save_model(project_variable, my_model)

    # add things to writer
    project_variable.writer.add_scalar('loss/train', loss, project_variable.current_epoch)
    project_variable.writer.add_scalar('accuracy/train', accuracy, project_variable.current_epoch)

    # project_variable.writer.add_scalars('some_new_shit', {'thing1': np.random.randint(5), 'thing2': np.random.randint(5)},
    #                                     project_variable.current_epoch)

    # project_variable.writer.add_scalar('some_shit', np.random.randint(5), project_variable.current_epoch)

    fig = VZ.plot_confusion_matrix(confusion_epoch, project_variable.dataset)
    project_variable.writer.add_figure(tag='confusion/train', figure=fig, global_step=project_variable.current_epoch)

