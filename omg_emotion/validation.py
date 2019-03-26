import numpy as np
from relative_baseline.omg_emotion import saving
import torch
from tqdm import tqdm
from relative_baseline.omg_emotion import utils as U
from relative_baseline.omg_emotion import data_loading as DL

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

    # project_variable = ProjectVariable()

    loss_epoch, accuracy_epoch, confusion_epoch, nice_div, steps, full_labels, full_data = \
        U.initialize(project_variable, all_data)

    # for ts in range(project_variable.train_steps):
    for ts in tqdm(range(steps)):

        # get part of data
        data, labels = DL.prepare_data(project_variable, full_data, full_labels, device, ts, steps, nice_div)

        my_model.eval()
        with torch.no_grad():
            # my_optimizer.zero_grad()
            # predictions = my_model.forward(data)
            predictions = my_model(data)
            # print(predictions)
            loss = U.calculate_loss(project_variable.loss_function, predictions, labels)
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
    accuracy = sum(accuracy_epoch) / (steps * project_variable.batch_size + nice_div)
    confusion_flatten = U.flatten_confusion(confusion_epoch)

    if project_variable.save_data:
        saving.update_logs(project_variable, 'val', [loss, accuracy, confusion_flatten])

    print('epoch %d val, %s: %f, accuracy: %f ' % (project_variable.current_epoch,
                                                   project_variable.loss_function,
                                                   loss, accuracy))

    # add things to writer
    project_variable.writer.add_scalar('metrics/loss', loss, project_variable.current_epoch)