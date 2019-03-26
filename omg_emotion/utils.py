import numpy as np
from torch.nn import CrossEntropyLoss


def initialize(project_variable, all_data):
    loss_epoch = []
    accuracy_epoch = []
    confusion_epoch = np.zeros(shape=(project_variable.label_size, project_variable.label_size), dtype=int)
    nice_div = len(all_data[0]) % project_variable.batch_size
    if nice_div == 0:
        steps = len(all_data[0]) // project_variable.batch_size
    else:
        steps = len(all_data[0]) // project_variable.batch_size + 1

    full_data, full_labels = all_data

    if len(full_labels) == 1:
        full_labels = full_labels[0]

    return loss_epoch, accuracy_epoch, confusion_epoch, nice_div, steps, full_labels, full_data


def str_list_to_num_arr(input_list, to_type):
    assert to_type in [float, int]

    if to_type is float:
        input_list = np.array([float(input_list[i]) for i in range(len(input_list))])
    elif to_type is int:
        input_list = np.array([int(input_list[i]) for i in range(len(input_list))])

    return input_list


def calculate_loss(project_variable, input, target):
    loss_name = project_variable.loss_function
    if loss_name == 'cross_entropy':
        loss_function = CrossEntropyLoss(weight=project_variable.loss_weights)
    else:
        loss_function = None

    loss = loss_function(input, target)
    return loss


def calculate_accuracy(input, target):
    # accuracy of step
    acc = 0

    input = input.cpu()
    input = np.array(input.data)

    target = target.cpu()
    target = np.array(target.data)

    for i in range(len(input)):
        if input[i].argmax() == target[i]:
            acc += 1

    return acc


def confusion_matrix(matrix, input, target):
    # col = target, row = prediction
    input = input.cpu()
    input = np.array(input.data)

    target = target.cpu()
    target = np.array(target.data)

    for i in range(len(input)):
        matrix[target[i], input[i].argmax()] += 1

    return matrix


def flatten_confusion(matrix):
    s = ''
    matrix = matrix.flatten()
    for i in range(len(matrix)):
        s += '%d,' % matrix[i]

    s = s[:-1]

    return s
