import tqdm
import os
import shutil
import numpy as np
from torch.nn import CrossEntropyLoss
from torchviz import make_dot
import torch

from relative_baseline.omg_emotion import project_paths as PP


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


def experiment_runs_statistics(experiment, model):

    test_acc, test_std, test_best = 0, 0, 0

    for i in ['train', 'val', 'test']:
        acc = []
        folder_path = os.path.join(PP.saving_data, i)
        name = 'experiment_%d_model_%d_run' % (experiment, model)
        runs = sum([name in j for j in os.listdir(folder_path)])

        for j in range(runs):
            file_path = os.path.join(folder_path, '%s_%d.txt' % (name, j))
            if os.path.exists(file_path):
                # file_path = os.path.join(folder_path, '%s_%d.txt' % (name, j) )
                data = np.genfromtxt(file_path, str, delimiter=',')
                if len(data.shape) == 1:
                    acc.append(float(data[1]))
                else:
                    acc.append(float(data[-1][1]))

        # print('%s   mean: %f    std: %f     runs: %d    best run: %d' % (i, np.mean(acc), np.std(acc), runs,
        #                                                                  acc.index(max(acc))))

        if i == 'test':
            test_acc = np.mean(acc)
            test_std = np.std(acc)
            test_best = acc.index(max(acc))

    return test_acc, test_std, test_best


def experiment_exists(experiment_number, model_number):
    num_runs = 0

    path = os.path.join(PP.saving_data, 'tensorboardX', 'experiment_%d_model_%d' % (experiment_number, model_number))
    if os.path.exists(path):
        num_runs = len(os.listdir(path)) - 1  # -1 to adjust for logging dirs number difference.

    return num_runs


# https://pytorch.org/docs/master/onnx.html
# https://github.com/onnx/onnx-tensorflow
# https://github.com/ysh329/deep-learning-model-convertor
# https://github.com/albermax/innvestigate/blob/master/examples/notebooks/introduction.ipynb

# for i in range(12, 16):
#     print(i)
#     experiment_runs_statistics(i, 5)
# experiment_runs_statistics(22, 5)


# ================================================================
# !! NOTE: be careful. this method DELETES stuff. use with care !!
# ================================================================
def delete_runs(project_variable, except_run):
    base_path = PP.models

    for i in range(project_variable.repeat_experiments):
        name = 'experiment_%d_model_%d_run_%d' % (project_variable.experiment_number,
                                                  project_variable.model_number, i)
        if i != except_run:
            to_be_del = os.path.join(base_path, name)
            shutil.rmtree(to_be_del)


# ================================================================
# !! NOTE: be careful. this method DELETES stuff. use with care !!
# ================================================================
# def remove_model_files():
#
#     def delete_existing_runs(experiment, model):
#         # find best run
#         _, _, best_run = experiment_runs_statistics(experiment, model)
#         num_runs = experiment_exists(experiment, model) + 1
#
#         base_path = PP.models
#
#         for i in range(num_runs):
#             name = 'experiment_%d_model_%d_run_%d' % (experiment, model, i)
#             to_be_del = os.path.join(base_path, name)
#             if os.path.exists(to_be_del):
#                 if i != best_run:
#                     print(to_be_del)
#                     ## shutil.rmtree(to_be_del)
#                 else:
#                     # delete all but last epoch
#                     num_epochs = len(os.listdir(to_be_del))
#                     for j in range(num_epochs):
#                         delete_file = os.path.join(to_be_del, 'epoch_%d' % j)
#                         if j != num_epochs-1:
#                             print(delete_file)
#                             ## os.remove(delete_file)
#
#     model_num = 5
#     experiments = list(np.arange(1, 44))
#     for expe in tqdm.tqdm(experiments):
#         delete_existing_runs(expe, model_num)

# ================================================================
# !! NOTE: be careful. this method DELETES stuff. use with care !!
# ================================================================
def remove_all_files(experiment, model):
    base_path = PP.saving_data
    folders = os.listdir(base_path)
    name = 'experiment_%d_model_%d' % (experiment, model)

    for i in folders:
        folder_path = os.path.join(base_path, i)
        files = os.listdir(folder_path)
        for j in files:
            if name in j:
                file_path = os.path.join(folder_path, j)
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                elif os.path.isfile(file_path):
                    os.remove(file_path)
