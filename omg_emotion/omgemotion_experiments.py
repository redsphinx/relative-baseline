from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


def e1_C3D_omgemo():
    project_variable.model_number = 2
    project_variable.end_epoch = 10
    project_variable.dataset = 'omg_emotion'

    project_variable.data_points = [191, 192, 0]
    project_variable.label_size = 6

    project_variable.repeat_experiments = 1
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True

    project_variable.experiment_state = 'new'
    project_variable.sheet_number = 100

    project_variable.eval_on = 'val'

    project_variable.experiment_number = 1
    main_file.run(project_variable)



project_variable = ProjectVariable(debug_mode=True)