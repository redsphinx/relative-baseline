import numpy as np
from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


def set_init_1():
    project_variable.end_epoch = 100
    project_variable.dataset = 'jester'

    # if you want all the data: train: 4200, val: 250, test: 250
    # total_dp = {'train': 118562, 'val': 7393, 'test': 7394}
    project_variable.num_in_channels = 3
    project_variable.data_points = [2 * 27,  1 * 27, 0 * 27]
    project_variable.label_size = 27
    project_variable.batch_size = 5 * 27
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
    project_variable.num_out_channels = [32, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    project_variable.do_xai = False
    project_variable.which_methods = ['gradient_method']
    project_variable.which_layers = ['conv1', 'conv2', 'conv3']
    project_variable.which_channels = [np.arange(2), np.arange(2), np.arange(2)]

    main_file.run(project_variable)


def e3_conv3DTTN_jester():
    set_init_1()
    project_variable.model_number = 16
    project_variable.experiment_number = 3
    project_variable.sheet_number = 22
    project_variable.device = 0
    project_variable.end_epoch = 200
    project_variable.repeat_experiments = 5
    project_variable.batch_size = 5 * 27  # 9021MiB on lovelace

    # if you want all the data: train: 4200, val: 250, test: 250
    project_variable.data_points = [300 * 27, 50 * 27, 0 * 27]

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.0003
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [16, 32, 64, 128, 256]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    project_variable.do_xai = False
    project_variable.which_methods = ['gradient_method']
    project_variable.which_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    project_variable.which_channels = [np.arange(10), np.arange(10), np.arange(10), np.arange(10), np.arange(10)]

    main_file.run(project_variable)


def e4_conv3DTTN_jester():
    set_init_1()
    project_variable.model_number = 16
    project_variable.experiment_number = 4
    project_variable.sheet_number = 22
    project_variable.device = 0
    project_variable.end_epoch = 20
    project_variable.repeat_experiments = 3
    project_variable.batch_size = 2 * 27

    # if you want all the data: train: 4200, val: 250, test: 250
    # project_variable.data_points = [300 * 27, 50 * 27, 0 * 27]
    project_variable.data_points = [30 * 27, 5 * 27, 0 * 27]

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.0003
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [16, 32, 64, 128, 256]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)


def etes_conv3DTTN_jester():
    set_init_1()
    project_variable.model_number = 11
    project_variable.experiment_number = 12234232323232
    project_variable.sheet_number = 22
    project_variable.device = 0
    project_variable.end_epoch = 3
    project_variable.repeat_experiments = 1
    project_variable.batch_size = 5*27 # 10 * 27

    project_variable.use_dali = True
    project_variable.dali_workers = 8
    # for now, use 'all' for val, since idk how to reset the iterator
    project_variable.dali_iterator_size = [5*27, 'all', 0]

    # project_variable.stop_at_collapse = True
    # project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.0003
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]
    # project_variable.num_out_channels = [16, 32, 64, 128, 256]

    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=True)

etes_conv3DTTN_jester()