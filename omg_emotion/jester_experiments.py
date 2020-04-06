import numpy as np
from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


def set_init_1():
    project_variable.end_epoch = 100
    project_variable.dataset = 'jester'

    # if you want all the data: train: 150, val: 10, test: 10
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

    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'

    project_variable.experiment_state = 'new'
    project_variable.eval_on = 'val'


def e1_3D_jester():
    set_init_1()
    project_variable.model_number = 12
    project_variable.experiment_number = 1
    project_variable.sheet_number = 22
    project_variable.device = 0

    project_variable.data_points = [50 * 27, 5 * 27, 0 * 27]

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


def e_test_3D_jester():
    set_init_1()
    project_variable.model_number = 14
    project_variable.experiment_number = 1792792989823
    project_variable.sheet_number = 22
    project_variable.device = 0
    project_variable.end_epoch = 30
    project_variable.repeat_experiments = 1

    project_variable.data_points = [30 * 27, 5 * 27, 0 * 27]

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 5e-4
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16, 32]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    project_variable.do_xai = False
    project_variable.which_methods = ['gradient_method']
    project_variable.which_layers = ['conv1', 'conv2', 'conv3']
    project_variable.which_channels = [np.arange(2), np.arange(2), np.arange(2)]

    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=True)


# e1_3D_jester()
e_test_3D_jester()
