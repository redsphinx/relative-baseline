from relative_baseline.omg_emotion import training
from relative_baseline.omg_emotion import validation
from relative_baseline.omg_emotion import testing
from relative_baseline.omg_emotion import setup
from relative_baseline.omg_emotion import data_loading as D



# temporary for debugging
# from .settings import ProjectVariable
# project_variable = ProjectVariable()

def run(project_variable):
# def run(project_variable=project_variable):

    # TODO: load val and test data once

    # setup model, optimizer & device
    my_model = setup.get_model(project_variable)
    my_optimizer = setup.get_optimizer(project_variable, my_model)
    device = setup.get_device(project_variable)

    # put model on GPU
    my_model.to(device)

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
        data = D.load_data(project_variable)
        # TODO: split the data nicely

        if project_variable.train:
            training.run(project_variable, data, my_model, my_optimizer, device)


        # if project_variable.val:
        #     pass
        #
        # if project_variable.test:
        #     pass

