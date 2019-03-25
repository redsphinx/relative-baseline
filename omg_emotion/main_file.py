from relative_baseline.omg_emotion import training
from relative_baseline.omg_emotion import validation
from relative_baseline.omg_emotion import testing
from relative_baseline.omg_emotion import setup
from relative_baseline.omg_emotion import data_loading as D
from relative_baseline.omg_emotion import project_paths as PP

import os
from tensorboardX import SummaryWriter

# temporary for debugging
# from .settings import ProjectVariable


def run(project_variable):
    # project_variable = ProjectVariable()

    # create writer for tensorboardX
    path = os.path.join(PP.writer_path, 'experiment_%d_model_%d' % (project_variable.experiment_number,
                                                                    project_variable.model_number))
    if not os.path.exists(path):
        os.mkdir(path)
    project_variable.writer = SummaryWriter(path)

    # load val and test data once
    project_variable.val = True
    project_variable.train = False
    project_variable.test = False

    data = D.load_data(project_variable)

    if project_variable.val:
        data_val = data[1][0]
        labels_val = data[2][0]
        print(labels_val)

    if project_variable.test:
        data_test = data[1][0]
        labels_test = data[2][0]

    # setup model, optimizer & device
    my_model = setup.get_model(project_variable)

    # print(my_model.fc.weight)

    device = setup.get_device(project_variable)

    if project_variable.device is not None:
        my_model.cuda(device)

    if project_variable.train:
        my_optimizer = setup.get_optimizer(project_variable, my_model)
    else:
        my_optimizer = None

    data = data_val, labels_val
    validation.run(project_variable, my_optimizer, data, my_model, device)

    # for e in range(project_variable.start_epoch+1, project_variable.end_epoch):
    #     project_variable.current_epoch = e
        # my_model.train()

        # get data
        # splits = ['train', 'val', 'test']
        # final_data = [[img0, img1,...],
        #               [img0, img1,...],
        #               [img0, img1,...]]
        # final_labels = [[arousal, valence, categories],
        #                 [arousal, valence, categories],
        #                 [arousal, valence, categories]]

        # project_variable.train = True
        # project_variable.val = False
        # project_variable.test = False

        # data = D.load_data(project_variable)
        # data_train = data[1][0]
        # labels_train = data[2][0]
        # labels is list because can be more than one type of labels

        # data = data_train, labels_train
        #
        # if project_variable.train:
        #     training.run(project_variable, data, my_model, my_optimizer, device)

        # project_variable.val = True
        # if project_variable.val:
        #     data = data_val, labels_val
        #     validation.run(project_variable, my_optimizer, data, my_model, device)

        # project_variable.test = True
        # if project_variable.test:
        #     data = data_test, labels_test
        #     testing.run(project_variable, my_optimizer, data, my_model, device)


def run_many_val(project_variable, data1, data2):
    # create writer for tensorboardX
    # setup model, optimizer & device
    my_model = setup.get_model(project_variable)
    # print(my_model.fc.weight)
    device = setup.get_device(project_variable)

    if project_variable.device is not None:
        my_model.cuda(device)

    if project_variable.train:
        my_optimizer = setup.get_optimizer(project_variable, my_model)
    else:
        my_optimizer = None

    print(data2)
    validation.run(project_variable, my_optimizer, (data1, data2), my_model, device)

    del my_model


def run_many_val_0(project_variable):
    # from .settings import ProjectVariable
    # project_variable = ProjectVariable()

    # create writer for tensorboardX
    path = os.path.join(PP.writer_path, 'experiment_%d_model_%d' % (project_variable.experiment_number,
                                                                    project_variable.model_number))
    if not os.path.exists(path):
        os.mkdir(path)
    project_variable.writer = SummaryWriter(path)

    # load val and test data once
    project_variable.val = True
    project_variable.train = False
    project_variable.test = False

    data = D.load_data(project_variable)

    if project_variable.val:
        data_val = data[1][0]
        labels_val = data[2][0]
    else:
        data_val = None
        labels_val = None

    device = setup.get_device(project_variable)

    ex, mo, ep = project_variable.load_model
    all_models = os.path.join(PP.models, 'experiment_%d_model_%d' % (ex, mo))
    models_to_load = len(os.listdir(all_models))

    prev_data_val = data_val
    prev_labels_val = labels_val

    for i in range(0, models_to_load):
        # data = data_val, labels_val

        project_variable.current_epoch = i

        project_variable.load_model = [ex, mo, i] # [experiment, model, epoch]

        # setup model, optimizer & device
        my_model = setup.get_model(project_variable)
        # my_model.eval()


        if project_variable.device is not None:
            my_model.cuda(device)

        # my_optimizer = setup.get_optimizer(project_variable, my_model)
        my_optimizer = None

        project_variable.val = True
        if project_variable.val:

            validation.run(project_variable, my_optimizer, (data_val, labels_val), my_model, device)


