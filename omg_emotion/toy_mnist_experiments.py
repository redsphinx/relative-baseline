from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file
import numpy as np

project_variable = ProjectVariable(debug_mode=True)


def pilot():
    project_variable.device = 1
    project_variable.model_number = 1
    project_variable.experiment_number = 0

    project_variable.batch_size = 20

    # project_variable.save_model = False
    # project_variable.save_data = False

    main_file.run(project_variable)


def dummy_data():
    project_variable.device = 1
    project_variable.model_number = 2
    project_variable.experiment_number = 1
    project_variable.batch_size = 20
    project_variable.dataset = 'dummy'
    main_file.run(project_variable)


def conv3dttnpilot():
    project_variable.device = 1
    project_variable.model_number = 3
    project_variable.experiment_number = 1
    project_variable.batch_size = 20
    project_variable.dataset = 'dummy'
    main_file.run(project_variable)


def conv3dttn_mmnist_pilot():
    project_variable.device = 1
    project_variable.model_number = 3
    project_variable.experiment_number = 1
    project_variable.batch_size = 10
    project_variable.dataset = 'mov_mnist'
    main_file.run(project_variable)


# dummy_data()  # x = torch.Size([20, 1, 30, 28, 28])
# torch.Size([20, 16, 5, 5, 5]) before view
# torch.Size([100, 400]) after view
# torch.Size([100, 10])


# conv3dttnpilot()  # x = torch.Size([20, 1, 30, 28, 28])
# torch.Size([20, 16, 5, 5, 5])
# torch.Size([100, 400])
# torch.Size([100, 10])

conv3dttn_mmnist_pilot()
