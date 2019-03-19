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

    for e in range(project_variable.epochs + 1):

        if project_variable.train:
            training.run(project_variable, data, my_model, my_optimizer, device)

            pass

        if project_variable.val:
            pass

        if project_variable.test:
            pass
