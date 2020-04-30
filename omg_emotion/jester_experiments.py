import numpy as np
from multiprocessing import Pool

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


def same_settings(pv):
    pv.nas = True

    pv.end_epoch = 1
    pv.dataset = 'jester'

    # total_dp = {'train': 118562, 'val': 7393, 'test': 7394}
    pv.num_in_channels = 3
    pv.label_size = 27
    pv.batch_size = 5 * 27
    pv.load_num_frames = 30
    pv.label_type = 'categories'

    pv.repeat_experiments = 1
    pv.save_only_best_run = True
    pv.same_training_data = True
    pv.randomize_training_data = True
    pv.balance_training_data = True

    pv.theta_init = None
    pv.srxy_init = 'eye'
    pv.weight_transform = 'seq'

    pv.experiment_state = 'new'
    pv.eval_on = 'val'

    pv.model_number = 11
    pv.sheet_number = 22

    pv.use_dali = True
    pv.dali_workers = 8
    # for now, use 'all' for val, since idk how to reset the iterator
    pv.dali_iterator_size = [5 * 27, 10 * 27, 0]

    # pv.stop_at_collapse = True
    # pv.early_stopping = True

    pv.optimizer = 'adam'
    pv.learning_rate = 0.0003
    pv.use_adaptive_lr = True
    # pv.num_out_channels = [6, 16]

    return pv

def parallel_experiment():
    pv1 = ProjectVariable(debug_mode=True)
    p1 = same_settings(pv1)
    pv1.experiment_number = 111111111111111111
    pv1.num_out_channels = [6, 16]
    pv1.device = 0

    pv2 = ProjectVariable(debug_mode=True)
    p2 = same_settings(pv2)
    pv2.experiment_number = 222222222222222
    pv2.num_out_channels = [12, 22]
    pv2.device = 1

    pv3 = ProjectVariable(debug_mode=True)
    p3 = same_settings(pv3)
    pv3.experiment_number = 333333333333333
    pv3.num_out_channels = [8, 18]
    pv3.device = 2

    pool = Pool(processes=3)
    # pool.apply_async(main_file.run)
    results = pool.map(main_file.run, [p1, p2, p3])

    pool.join()
    pool.close()

    # write the results to some file



def e6_conv3DTTN_jester():
    set_init_1()
    project_variable.model_number = 17
    project_variable.experiment_number = 6
    project_variable.sheet_number = 22
    project_variable.device = 0
    project_variable.end_epoch = 30
    project_variable.repeat_experiments = 3
    project_variable.batch_size = 5 * 27

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [10, 14, 32, 32, 50]

    main_file.run(project_variable)


def e7_conv3DTTN_jester():
    set_init_1()
    project_variable.model_number = 18
    project_variable.experiment_number = 7
    project_variable.sheet_number = 22
    project_variable.device = 2
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 3
    project_variable.batch_size = 5 * 27

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [10, 20, 32, 32, 38, 44]

    main_file.run(project_variable)


def e8_conv3DTTN_jester():
    set_init_1()
    project_variable.model_number = 17
    project_variable.experiment_number = 8
    project_variable.sheet_number = 22
    project_variable.device = 2
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 3
    project_variable.batch_size = 5 * 27

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [10, 20, 32, 32, 38, 44, 44]

    main_file.run(project_variable)


def eRESNET_jester():
    set_init_1()
    project_variable.model_number = 20
    project_variable.experiment_number = 8
    project_variable.sheet_number = 22
    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 3
    project_variable.batch_size = 1 * 27

    project_variable.data_points = [1 * 27, 1 * 27, 0 * 27]

    # project_variable.use_dali = True
    # project_variable.dali_workers = 32
    # project_variable.dali_iterator_size = ['all', 'all', 0]
    # project_variable.nas = False

    # project_variable.stop_at_collapse = True
    # project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    main_file.run(project_variable)


def e9_conv3DTTN_jester():
    set_init_1()
    project_variable.model_number = 20
    project_variable.experiment_number = 9
    project_variable.sheet_number = 22
    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.repeat_experiments = 3
    project_variable.batch_size = 5 * 27

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    main_file.run(project_variable)

# project_variable = ProjectVariable(debug_mode=False)
project_variable = ProjectVariable(debug_mode=False)

# e6_conv3DTTN_jester()
# e7_conv3DTTN_jester()
# e8_conv3DTTN_jester()
# eRESNET_jester()
e9_conv3DTTN_jester()