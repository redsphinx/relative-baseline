from relative_baseline.omg_emotion import training
from relative_baseline.omg_emotion import validation
from relative_baseline.omg_emotion import testing
from relative_baseline.omg_emotion import setup
from relative_baseline.omg_emotion import data_loading as D

from . import training

# temporary for debugging
# from .settings import ProjectVariable
# project_variable = ProjectVariable()


def run(project_variable):
# def run(project_variable=project_variable):

    # load val and test data once
    # project_variable.val = True
    # project_variable.test = True
    # data = D.load_data(project_variable)

    # data_val = data[1][0]
    # data_test = data[1][1]
    #
    # labels_val = data[2][0]
    # labels_test = data[2][1]

    # setup model, optimizer & device
    my_model = setup.get_model(project_variable)
    device = setup.get_device(project_variable)

    if project_variable.device is not None:
        my_model.cuda(device)

    my_optimizer = setup.get_optimizer(project_variable, my_model)

    for e in range(project_variable.start_epoch+1, project_variable.end_epoch):
        project_variable.current_epoch = e

        # get data
        # splits = ['train', 'val', 'test']
        # final_data = [[img0, img1,...],
        #               [img0, img1,...],
        #               [img0, img1,...]]
        # final_labels = [[arousal, valence, categories],
        #                 [arousal, valence, categories],
        #                 [arousal, valence, categories]]

        project_variable.train = True
        project_variable.val = False
        project_variable.test = False

        data = D.load_data(project_variable)
        data_train = data[1][0]
        labels_train = data[2][0]
        # labels is list because can be more than one type of labels

        data = data_train, labels_train

        if project_variable.train:
            training.run(project_variable, data, my_model, my_optimizer, device)

        # TODO: implement
        # project_variable.val = True
        # project_variable.test = True
        #
        # if project_variable.val:
        #     pass
        #
        # if project_variable.test:
        #     pass

