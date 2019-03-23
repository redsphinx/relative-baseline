import numpy as np
from relative_baseline.omg_emotion import saving
import torch
from tqdm import tqdm
from relative_baseline.omg_emotion import utils as U

# temporary for debugging
from .settings import ProjectVariable


def run(project_variable, all_data, my_model, device):
    # all_data = np.array with the train datasplit depending
    # all_data = [data, labels] shape = (n, 2)
    # device is string

    # project_variable = ProjectVariable()

    loss_epoch = []
    accuracy_epoch = []

    val_steps = len(all_data[0]) // project_variable.batch_size

    full_data, full_labels = all_data

    if len(full_labels) == 1:
        full_labels = full_labels[0]

    # for ts in range(project_variable.train_steps):
    for ts in tqdm(range(val_steps)):

        # get part of data
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
            labels = torch.from_numpy(labels)

            labels = labels.long()
            # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542

            labels = labels.cuda(device)
        else:
            data = torch.from_numpy(data).cuda(device)
            labels = torch.from_numpy(labels).cuda(device)

        # train
        # with torch.device(device):
        predictions = my_model(data)
        loss = U.calculate_loss(project_variable.loss_function, predictions, labels)

        accuracy = U.calculate_accuracy(predictions, labels)

        loss_epoch.append(float(loss))
        accuracy_epoch.append(float(accuracy))

    # save data
    loss = float(np.mean(loss_epoch))
    accuracy = float(np.mean(accuracy_epoch))

    if project_variable.save_data:
        saving.update_logs(project_variable, 'val', [loss, accuracy])

    print('epoch %d val, %s: %f, accuracy: %f out of %d' % (project_variable.current_epoch,
                                                            project_variable.loss_function,
                                                            loss, accuracy, project_variable.batch_size))
