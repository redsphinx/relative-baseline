from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


def pilot():
    project_variable.device = 0
    project_variable.model_number = 2
    project_variable.experiment_number = 666
    project_variable.batch_size = 10
    project_variable.end_epoch = 20
    project_variable.dataset = 'kth_actions'
    project_variable.data_points = [12, 12, 12]
    project_variable.repeat_experiments = 1
    project_variable.same_training_data = False


    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=True)