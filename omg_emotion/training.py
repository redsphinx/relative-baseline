import numpy as np
from relative_baseline.omg_emotion import saving
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# temporary for debugging
from .settings import ProjectVariable


def calculate_loss(loss_name, input, target):
    if loss_name == 'cross_entropy':
        loss_function = CrossEntropyLoss()
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


def run(project_variable, all_data, my_model, my_optimizer, device):
    # all_data = np.array with the train datasplit depending
    # all_data = [data, labels] shape = (n, 2)
    # device is string

    # project_variable = ProjectVariable()

    loss_epoch = []
    accuracy_epoch = []

    train_steps = len(all_data[0]) // project_variable.batch_size
    # print('train steps: %d' % train_steps)

    full_data, full_labels = all_data

    if len(full_labels) == 1:
        full_labels = full_labels[0]

    # for ts in range(project_variable.train_steps):
    for ts in tqdm(range(train_steps)):

        # get part of data
        # data, labels = all_data[ts*project_variable.batch_size:(1+ts)*project_variable.batch_size][:]
        data = full_data[ts*project_variable.batch_size:(ts+1)*project_variable.batch_size]
        labels = full_labels[ts * project_variable.batch_size:(ts + 1) * project_variable.batch_size]

        if project_variable.model_number == 0:
            # normalize image data
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            data = torch.from_numpy(data)
            for _b in range(project_variable.batch_size):
                data[_b] = normalize(data[_b])

            data = data.cuda(device)

            # onehot encode labels -- not needed

            # labels = np.expand_dims(labels, -1)
            labels = torch.from_numpy(labels)

            # labels_onehot = torch.Tensor(project_variable.batch_size, project_variable.label_size)
            # labels_onehot.zero_()
            # labels_onehot.scatter_(1, labels, 1)
            #
            # # crossentropyloss requires target to be of type long
            # labels = labels_onehot.long()
            labels = labels.long()
            # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542

            labels = labels.cuda(device)


        else:
            data = torch.from_numpy(data).cuda(device)
            labels = torch.from_numpy(labels).cuda(device)

        # train
        # with torch.device(device):
        my_optimizer.zero_grad()
        predictions = my_model(data)
        loss = calculate_loss(project_variable.loss_function, predictions, labels)
        loss.backward()
        my_optimizer.step()

        accuracy = calculate_accuracy(predictions, labels)

        loss_epoch.append(float(loss))
        accuracy_epoch.append(float(accuracy))

    # save data
    loss = float(np.mean(loss_epoch))
    accuracy = float(np.mean(accuracy_epoch))

    if project_variable.save_data:
        saving.update_logs(project_variable, 'train', [loss, accuracy])

    print('epoch %d train, %s: %f, accuracy: %f out of %d' % (project_variable.current_epoch,
                                                              project_variable.loss_function,
                                                              loss, accuracy, project_variable.batch_size))

    # save model
    if project_variable.save_model:
        saving.save_model(project_variable, my_model)

    # save graphs
    # TODO


