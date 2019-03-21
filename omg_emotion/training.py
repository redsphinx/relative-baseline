import numpy as np
from relative_baseline.omg_emotion import saving
import torch
from torch.nn import CrossEntropyLoss

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

    # project_variable = ProjectVariable()

    loss_epoch = []

    train_steps = len(all_data[0]) // project_variable.batch_size

    # for ts in range(project_variable.train_steps):
    for ts in range(train_steps):

        # get part of data
        data, labels = all_data[ts*project_variable.batch_size:(1+ts)*project_variable.batch_size][:]
        if len(labels) == 1:
            labels = labels[0]

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

        loss_epoch.append(float(loss))

    # save data
    # TODO: DEBUG FROM HERE
    loss = float(np.mean(loss_epoch))
    if project_variable.save_data:
        saving.update_logs(project_variable, 'train', [loss])

    print('epoch %d train %s: %f' % (project_variable.current_epoch, project_variable.loss_function, loss))

    # save model
    if project_variable.save_model:
        saving.save_model(project_variable, my_model)

    # save graphs
    # TODO


# TODO
# added different learning rates for different layers
