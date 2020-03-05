import numpy as np
from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


def set_init_1():
    project_variable.end_epoch = 100
    project_variable.dataset = 'jester'

    # total_dp = {'train': 118562, 'val': 7393, 'test': 7394}
    project_variable.num_in_channels = 3
    project_variable.data_points = [2 * 27,  1 * 27, 0 * 27]
    project_variable.label_size = 27
    project_variable.batch_size = 2 * 27
    project_variable.load_num_frames = 30
    project_variable.label_type = 'categories'

    project_variable.repeat_experiments = 1
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True

    project_variable.experiment_state = 'new'
    project_variable.eval_on = 'val'


def e0_3D_jester():
    set_init_1()
    project_variable.model_number = 12
    # project_variable.model_number = 11
    project_variable.experiment_number = 918274591283
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=True)


e0_3D_jester()