from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


project_variable = ProjectVariable(debug_mode=False)


def pilot():
    project_variable.device = 0
    project_variable.model_number = 0
    project_variable.experiment_number = 0

    project_variable.batch_size = 20

    # project_variable.save_model = False
    # project_variable.save_data = False

    main_file.run(project_variable)


def e1():
    project_variable.device = 0
    project_variable.model_number = 0
    project_variable.experiment_number = 1

    project_variable.batch_size = 20
    project_variable.end_epoch = 1

    project_variable.load_model = [0, 0, 99]  # experiment, model, epoch

    main_file.run(project_variable)
