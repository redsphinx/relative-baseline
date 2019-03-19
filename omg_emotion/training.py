import numpy as np
from . import saving
import torch
from torch.nn import CrossEntropyLoss

# temporary for debugging
from .settings import ProjectVariable


def calculate_loss(loss_name, input, target):
    if loss_name == 'cross_entropy':
        loss_function = CrossEntropyLoss
    else:
        loss_function = None

    loss = loss_function(input, target)
    return loss


def calculate_accuracy(input, target):
    # TODO
    assert len(input) == len(target)
    total = len(input)
    acc = 0

    def threshold(x):
        pass

    for i in range(total):
        if threshold(input[i]) == target[i]:
           acc += 1

    return acc


def run(project_variable, all_data, my_model, my_optimizer, device):
    # all_data = np.array with the train datasplit depending
    # all_data = [data, labels] shape = (n, 2)
    # device is string

    project_variable = ProjectVariable()

    loss_epoch = []

    for ts in range(project_variable.train_steps):

        # get part of data
        data, labels = all_data[ts*project_variable.batch_size:(1+ts)*project_variable.batch_size, :]

        # put data part on GPU
        data = torch.from_numpy(data).to(device)
        labels = torch.from_numpy(labels).to(device)

        # train
        with torch.device(device):
            my_optimizer.zero_grad()
            predictions = my_model(data)
            loss = calculate_loss(project_variable.loss_function, predictions, labels)
            loss.backward()
            my_optimizer.step()

        loss_epoch.append(loss)

    # save data
    loss = float(np.mean(loss_epoch))
    if project_variable.save_data:
        saving.update_logs(project_variable, 'train', [loss])

    print('epoch %d train %s: %f' % (project_variable.current_epoch, project_variable.loss_function, loss))

    # save model
    if project_variable.save_model:
        saving.save_model(project_variable, my_model)

    # save graphs
    # TODO