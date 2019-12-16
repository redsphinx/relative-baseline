from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


def e1_C3D_omgemo():
    project_variable.model_number = 11
    project_variable.end_epoch = 100
    project_variable.dataset = 'omg_emotion'
    project_variable.device = 2
    project_variable.loss_function = 'adam'
    project_variable.learning_rate = 5e-5
    # project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [12, 22]

    project_variable.data_points = [50*7, 3*7, 5*7]
    project_variable.label_size = 7
    project_variable.batch_size = 14
    project_variable.load_num_frames = 60 # 60
    project_variable.label_type = 'categories'

    project_variable.repeat_experiments = 1
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True

    project_variable.experiment_state = 'new'
    project_variable.sheet_number = 100

    project_variable.eval_on = 'test'

    project_variable.experiment_number = 1
    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=True)


e1_C3D_omgemo()