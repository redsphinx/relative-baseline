import numpy as np
from torch.nn import CrossEntropyLoss
from torchviz import make_dot
import torch


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
    if project_variable.model_number == 0:
        if loss_name == 'cross_entropy':
            loss_function = CrossEntropyLoss(weight=project_variable.loss_weights)
        else:
            loss_function = None
    else:
        loss_function = CrossEntropyLoss()

    loss = loss_function(input, target)
    return loss


def calculate_accuracy(input, target):
    # accuracy of step
    predictions = []
    labels = []

    acc = 0

    input = input.cpu()
    input = np.array(input.data)

    target = target.cpu()
    target = np.array(target.data)

    for i in range(len(input)):
        predictions.append(input[i].argmax())
        labels.append(target[i])

        if input[i].argmax() == target[i]:
            acc += 1

    # print('predictions: %s\n'
    #       'labels:      %s' % (str(predictions), str(labels)))

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


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_architecture_as_dot(model, file_name, save_location):
    # file_name ends with .dot
    x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False)
    out = model(x)
    dot = make_dot(out)
    dot.save(file_name, save_location)

# https://pytorch.org/docs/master/onnx.html
# https://github.com/onnx/onnx-tensorflow
# https://github.com/ysh329/deep-learning-model-convertor
# https://github.com/albermax/innvestigate/blob/master/examples/notebooks/introduction.ipynb


