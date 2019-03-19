from . import training
from . import validation
from . import test
from . import setup
from . import data_loading as D



# temporary for debugging
from .settings import ProjectVariable
project_variable = ProjectVariable()

# def run(project_variable):
def run(project_variable=project_variable):

    # get data
    data = D.load_data(project_variable)

    # setup model, optimizer & device
    my_model = setup.get_model(project_variable)
    my_optimizer = setup.get_optimizer(project_variable)
    device = setup.get_device(project_variable)

    # put model on GPU
    my_model.to(device)

    for e in range(project_variable.start_epoch+1, project_variable.end_epoch):
        project_variable.current_epoch = e

        if project_variable.train:
            training.run(project_variable, data, my_model, my_optimizer, device)


        # if project_variable.val:
        #     pass
        #
        # if project_variable.test:
        #     pass
