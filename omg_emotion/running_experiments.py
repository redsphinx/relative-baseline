from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file
import numpy as np

project_variable = ProjectVariable(debug_mode=False)


def pilot():
    project_variable.device = 0
    project_variable.model_number = 0
    project_variable.experiment_number = 0

    project_variable.batch_size = 20

    # project_variable.save_model = False
    # project_variable.save_data = False

    main_file.run(project_variable)


def pilot_2():
    project_variable.device = 0
    project_variable.model_number = 0
    project_variable.experiment_number = 1
    project_variable.end_epoch = 1
    project_variable.load_model = [0, 0, 99]
    project_variable.save_data = False
    main_file.run(project_variable)
    # main_file.run_many_val(project_variable)


# pilot_2()

def e2():
    project_variable.device = 0
    project_variable.model_number = 0
    project_variable.experiment_number = 2
    project_variable.load_model = [2, 0, 15]
    project_variable.start_epoch = 15
    main_file.run(project_variable)


e2()
