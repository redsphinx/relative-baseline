from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


def set_init_1():
    project_variable.model_number = 12
    project_variable.end_epoch = 100
    project_variable.dataset = 'dhg'

    project_variable.num_in_channels = 1
    project_variable.data_points = [140 * 14,  20 * 14, 40 * 14]
    project_variable.label_size = 14
    project_variable.batch_size = 2 * 14
    project_variable.load_num_frames = 50  # 50
    project_variable.label_type = 'categories'

    project_variable.repeat_experiments = 10
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True

    project_variable.experiment_state = 'new'
    project_variable.eval_on = 'val'


def e6_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 6
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e7_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 7
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e8_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 8
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-5
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


# -----------------------------------------------------------------------------------------------

def e9_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 9
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 5e-7
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e10_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 10
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 1e-8
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e11_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 11
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 5e-8
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

# -----------------------------------------------------------------------------------------------

def e12_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 12
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 5e-7
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e13_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 13
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 1e-8
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e14_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 14
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 5e-8
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


def e15_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 15
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 1e-7
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


def e16_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 16
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 1e-7
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e17_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 17
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.randomize_training_data = False
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


def e18_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 18
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.data_points = [1 * 14, 1 * 14, 1 * 14]

    project_variable.randomize_training_data = False
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

# ---------------------------------------------------------------------------------------------

def set_init_2():
    project_variable.model_number = 11
    project_variable.end_epoch = 100
    project_variable.dataset = 'dhg'

    project_variable.num_in_channels = 1
    project_variable.data_points = [140 * 14,  20 * 14, 40 * 14]
    project_variable.label_size = 14
    project_variable.batch_size = 2 * 14
    project_variable.load_num_frames = 50  # 50
    project_variable.label_type = 'categories'

    project_variable.repeat_experiments = 10
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True

    project_variable.experiment_state = 'new'
    project_variable.eval_on = 'val'

    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'


def e19_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 19
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e20_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 20
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e21_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 21
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-5
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

########

def e22_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 22
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e23_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 23
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e24_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 24
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-5
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

# ---------------------------------------------------------------------------------------------------
#                                       NO ADAPTIVE LR
# ---------------------------------------------------------------------------------------------------
def e25_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 25
    project_variable.sheet_number = 21
    project_variable.device = 1

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 5e-6
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e26_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 26
    project_variable.sheet_number = 21
    project_variable.device = 1

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 1e-7
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e27_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 27
    project_variable.sheet_number = 21
    project_variable.device = 1

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 5e-7
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e28_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 28
    project_variable.sheet_number = 21
    project_variable.device = 1

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 1e-8
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e29_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 29
    project_variable.sheet_number = 21
    project_variable.device = 1

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 5e-8
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e30_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 30
    project_variable.sheet_number = 21
    project_variable.device = 1

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 1e-9
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

# ---------------------------------------------------------------------------------------------------
#                                       WITH ADAPTIVE LR
# ---------------------------------------------------------------------------------------------------
def e31_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 31
    project_variable.sheet_number = 21
    project_variable.device = 1

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 5e-6
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


def e32_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 32
    project_variable.sheet_number = 21
    project_variable.device = 1

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 1e-7
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


def e33_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 33
    project_variable.sheet_number = 21
    project_variable.device = 1

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 5e-7
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


def e34_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 34
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 1e-8
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


def e35_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 35
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 5e-8
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


def e36_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 36
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'sgd'
    project_variable.learning_rate = 1e-9
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

# --------------------------------------------------------------------------------------------------------------
#                       BIGGER MODELS
# --------------------------------------------------------------------------------------------------------------

def e37_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 37
    project_variable.sheet_number = 21
    project_variable.device = 0

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [8, 18]

    main_file.run(project_variable)


def e38_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 38
    project_variable.sheet_number = 21
    project_variable.device = 0

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [12, 22]

    main_file.run(project_variable)



def e39_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 39
    project_variable.sheet_number = 21
    project_variable.device = 0

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [8, 18]

    main_file.run(project_variable)


def e40_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 40
    project_variable.sheet_number = 21
    project_variable.device = 1

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [12, 22]

    main_file.run(project_variable)

# --------

def e41_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 41
    project_variable.sheet_number = 21
    project_variable.device = 1

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [8, 18]

    main_file.run(project_variable)

def e42_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 42
    project_variable.sheet_number = 21
    project_variable.device = 1

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [12, 22]

    main_file.run(project_variable)

def e43_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 43
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [8, 18]

    main_file.run(project_variable)

def e44_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 44
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [12, 22]

    main_file.run(project_variable)

# -------- RERUNNING with adjustment transformation and k0_groups

def e45_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 45
    project_variable.sheet_number = 21
    project_variable.device = 0

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [8, 18]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e46_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 46
    project_variable.sheet_number = 21
    project_variable.device = 0

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [12, 22]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e47_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 47
    project_variable.sheet_number = 21
    project_variable.device = 0

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [8, 18]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e48_3D_dhg():
    set_init_2()
    project_variable.experiment_number = 48
    project_variable.sheet_number = 21
    project_variable.device = 0

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [12, 22]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# --------------------------------------------------------------------------------------------------------------
#                       LESS DATA
# --------------------------------------------------------------------------------------------------------------

def set_init_3():
    # project_variable.model_number = 11
    project_variable.end_epoch = 100
    project_variable.dataset = 'dhg'

    project_variable.num_in_channels = 1
    # project_variable.data_points = [140 * 14,  20 * 14, 40 * 14]
    project_variable.label_size = 14
    # project_variable.batch_size = 2 * 14
    project_variable.load_num_frames = 50  # 50
    project_variable.label_type = 'categories'

    project_variable.repeat_experiments = 10
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True

    project_variable.experiment_state = 'new'
    project_variable.eval_on = 'val'

    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'



project_variable = ProjectVariable(debug_mode=False)


# e45_3D_dhg()
# e46_3D_dhg()
# e47_3D_dhg()
e48_3D_dhg()
