import cProfile
from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


def pilot():
    project_variable.device = 0
    project_variable.model_number = 3
    project_variable.experiment_number = 666
    project_variable.batch_size = 10
    project_variable.end_epoch = 2
    project_variable.dataset = 'kth_actions'
    # project_variable.dataset = 'mov_mnist'
    project_variable.data_points = [12, 12, 12]
    # project_variable.data_points = [10, 10, 10]
    project_variable.repeat_experiments = 1
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.label_size = 6
    # project_variable.label_size = 10

    main_file.run(project_variable)


def pilot_2():
    project_variable.device = 0
    project_variable.model_number = 5
    project_variable.experiment_number = 666
    project_variable.batch_size = 16
    # project_variable.batch_size = 6
    project_variable.end_epoch = 10
    project_variable.dataset = 'kth_actions'
    project_variable.data_points = [191, 192, 216]
    # project_variable.data_points = [6, 6, 6]
    project_variable.repeat_experiments = 1
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.label_size = 6
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.0001

    main_file.run(project_variable)


#####################################################################################################################
#                   LONG EXPERIMENT MODEL 5 FINDING DECENT PARAMETERS 1 - 27
#####################################################################################################################
def set_init_1():
    project_variable.model_number = 5 # use C3D_experiment for these experiments.
    project_variable.batch_size = 16
    # project_variable.batch_size = 6
    project_variable.end_epoch = 50
    project_variable.dataset = 'kth_actions'
    project_variable.data_points = [191, 192, 216]
    # project_variable.data_points = [6, 6, 6]
    project_variable.repeat_experiments = 10
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.label_size = 6
    project_variable.optimizer = 'adam'
    project_variable.k_shape = (3, 3, 3)
    project_variable.experiment_state = 'new'
    project_variable.sheet_number = 10

# --------------------------------------------------------
#                   load_num_frames = 30
# --------------------------------------------------------

def e1_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 1
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)

def e2_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 2
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)
    
def e3_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 3
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)
# --
def e4_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 4
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [8, 16, 32, 64]
    main_file.run(project_variable)


def e5_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 5
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [8, 16, 32, 64]
    main_file.run(project_variable)


def e6_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 6
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [8, 16, 32, 64]
    main_file.run(project_variable)
# --
def e7_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 7
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [16, 32, 64, 128]
    main_file.run(project_variable)


def e8_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 8
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [16, 32, 64, 128]
    main_file.run(project_variable)


def e9_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 9
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [16, 32, 64, 128]
    main_file.run(project_variable)


# --------------------------------------------------------
#                   load_num_frames = 50
# --------------------------------------------------------

def e10_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 10
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)


def e11_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 11
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)


def e12_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 12
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)


# --
def e13_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 13
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [8, 16, 32, 64]
    main_file.run(project_variable)


def e14_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 14
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [8, 16, 32, 64]
    main_file.run(project_variable)


def e15_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 15
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [8, 16, 32, 64]
    main_file.run(project_variable)


# --
def e16_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 16
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [16, 32, 64, 128]
    main_file.run(project_variable)


def e17_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 17
    project_variable.device = 2
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [16, 32, 64, 128]
    main_file.run(project_variable)


def e18_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 18
    project_variable.device = 2
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [16, 32, 64, 128]
    main_file.run(project_variable)


# --------------------------------------------------------
#                   load_num_frames = 100
# --------------------------------------------------------

def e19_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 19
    project_variable.device = 2
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)


def e20_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 20
    project_variable.device = 2
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)


def e21_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 21
    project_variable.device = 1
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)


# --
def e22_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 22
    project_variable.device = 2
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [8, 16, 32, 64]
    main_file.run(project_variable)


def e23_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 23
    project_variable.device = 0
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [8, 16, 32, 64]
    main_file.run(project_variable)


def e24_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 24
    project_variable.device = 0
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [8, 16, 32, 64]
    main_file.run(project_variable)


# --
def e25_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 25
    project_variable.device = 1
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [16, 32, 64, 128]
    main_file.run(project_variable)


def e26_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 26
    project_variable.device = 0
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [16, 32, 64, 128]
    main_file.run(project_variable)


def e27_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 27
    project_variable.device = 2
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [16, 32, 64, 128]
    main_file.run(project_variable)
#####################################################################################################################
#                                         EXPERIMENTS WITH K_SHAPE 28 - 31
#####################################################################################################################
def set_init_2():
    set_init_1()
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.0001
    project_variable.sheet_number = 11

# --------------------------------------------------------
#              out_channels [8, 16, 32, 64]
# --------------------------------------------------------

def e28_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 28
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 5
    main_file.run(project_variable)

def e29_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 29
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 7
    main_file.run(project_variable)

# --------------------------------------------------------
#              out_channels [16, 32, 64, 128]
# --------------------------------------------------------

def e30_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 30
    project_variable.device = 2
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    main_file.run(project_variable)

def e31_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 31
    project_variable.device = 2
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 7
    main_file.run(project_variable)
#####################################################################################################################
#                                         BATCHNORM 32 - 49
#####################################################################################################################
def set_init_2():
    set_init_1()
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.0001
    project_variable.sheet_number = 11


# --------------------------------------------------------
#              out_channels [8, 16, 32, 64]
# --------------------------------------------------------

def e32_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 32
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 3
    project_variable.do_batchnorm = [True, True, True, True, True]
    main_file.run(project_variable)

def e33_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 33
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 3
    project_variable.do_batchnorm = [True, True, False, True, True]
    main_file.run(project_variable)

def e34_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 34
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 3
    project_variable.do_batchnorm = [True, False, False, False, False]
    main_file.run(project_variable)

def e35_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 35
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 3
    project_variable.do_batchnorm = [False, True, False, False, False]
    main_file.run(project_variable)

def e36_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 36
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 3
    project_variable.do_batchnorm = [False, False, False, True, False]
    main_file.run(project_variable)

def e37_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 37
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 3
    project_variable.do_batchnorm = [False, False, False, False, True]
    main_file.run(project_variable)


# --------------------------------------------------------
#              out_channels [16, 32, 64, 128]
# --------------------------------------------------------

def e38_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 38
    project_variable.device = 1
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    project_variable.do_batchnorm = [True, True, True, True, True]
    main_file.run(project_variable)


def e39_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 39
    project_variable.device = 1
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    project_variable.do_batchnorm = [True, True, False, True, True]
    main_file.run(project_variable)


def e40_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 40
    project_variable.device = 1
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    project_variable.do_batchnorm = [True, False, False, False, False]
    main_file.run(project_variable)


def e41_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 41
    project_variable.device = 1
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    project_variable.do_batchnorm = [False, True, False, False, False]
    main_file.run(project_variable)


def e42_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 42
    project_variable.device = 1
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    project_variable.do_batchnorm = [False, False, False, True, False]
    main_file.run(project_variable)


def e43_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 43
    project_variable.device = 1
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    project_variable.do_batchnorm = [False, False, False, False, True]
    main_file.run(project_variable)

# --------------------------------------------------------
#              out_channels [8, 16, 32, 64]
# --------------------------------------------------------
def e44_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 44
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 3
    project_variable.do_batchnorm = [False, False, True, False, False]
    main_file.run(project_variable)

def e45_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 45
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 3
    project_variable.do_batchnorm = [True, True, False, False, False]
    main_file.run(project_variable)

def e46_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 46
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 3
    project_variable.do_batchnorm = [True, True, True, False, False]
    main_file.run(project_variable)
# --------------------------------------------------------
#              out_channels [16, 32, 64, 128]
# --------------------------------------------------------
def e47_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 47
    project_variable.device = 2
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    project_variable.do_batchnorm = [False, False, True, False, False]
    main_file.run(project_variable)

def e48_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 48
    project_variable.device = 2
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    project_variable.do_batchnorm = [True, True, False, False, False]
    main_file.run(project_variable)

def e49_C3D_kth():
    set_init_2()
    project_variable.experiment_number = 49
    project_variable.device = 2
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    project_variable.do_batchnorm = [True, True, True, False, False]
    main_file.run(project_variable)


def pilot_3():
    project_variable.model_number = 5
    project_variable.batch_size = 16
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.label_size = 6

    # project_variable.conv1_k_t = 3
    # project_variable.k_shape = (3, 3, 3)
    # project_variable.do_batchnorm = [True, False, False, False, False]

    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.dataset = 'kth_actions'
    project_variable.data_points = [191, 192, 216]
    project_variable.optimizer = 'adam'
    project_variable.experiment_number = 666
    project_variable.device = 0
    main_file.run(project_variable)


def pilot_4():
    # project_variable.model_number = 6
    project_variable.model_number = 5
    project_variable.batch_size = 16
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [20, 40, 80, 160]
    # project_variable.num_out_channels = [10, 18, 34, 66]
    project_variable.label_size = 6

    # project_variable.conv1_k_t = 3
    # project_variable.k_shape = (3, 3, 3)
    # project_variable.do_batchnorm = [True, False, False, False, False]

    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.srxy_smoothness = None
    project_variable.weight_transform = 'seq'
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.dataset = 'kth_actions'
    project_variable.data_points = [191, 192, 216]
    # project_variable.data_points = [12, 12, 12]
    project_variable.end_epoch = 20
    project_variable.optimizer = 'adam'
    project_variable.experiment_number = 666
    project_variable.device = 0
    main_file.run(project_variable)

#####################################################################################################################
#                       LONG EXPERIMENT; MODELS 5, 6; DATA_POINTS; OUT_CHANNELS; 50 - 105
#####################################################################################################################
def set_init_3():
    # architecture
    project_variable.learning_rate = 0.0001
    project_variable.label_size = 6
    project_variable.optimizer = 'adam'
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.srxy_smoothness = None
    project_variable.weight_transform = 'seq'
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    # data
    project_variable.batch_size = 16
    project_variable.load_num_frames = 30
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.dataset = 'kth_actions'
    # training
    project_variable.end_epoch = 50
    project_variable.repeat_experiments = 30
    # misc
    project_variable.sheet_number = 12
# --------------------------------------------------------
#          MODEL 5; OUT_CHANNELS [8, 16, 32, 64]
# --------------------------------------------------------
def e50_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 50
    project_variable.model_number = 5
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e51_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 51
    project_variable.model_number = 5
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e52_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 52
    project_variable.model_number = 5
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e53_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 53
    project_variable.model_number = 5
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 5; OUT_CHANNELS [10, 20, 40, 80]
# --------------------------------------------------------
def e54_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 54
    project_variable.model_number = 5
    project_variable.num_out_channels = [10, 20, 40, 80]
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e55_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 55
    project_variable.model_number = 5
    project_variable.num_out_channels = [10, 20, 40, 80]
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e56_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 56
    project_variable.model_number = 5
    project_variable.num_out_channels = [10, 20, 40, 80]
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e57_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 57
    project_variable.model_number = 5
    project_variable.num_out_channels = [10, 20, 40, 80]
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 5; OUT_CHANNELS [12, 24, 48, 96]
# --------------------------------------------------------
def e58_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 58
    project_variable.model_number = 5
    project_variable.num_out_channels = [12, 24, 48, 96]
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e59_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 59
    project_variable.model_number = 5
    project_variable.num_out_channels = [12, 24, 48, 96]
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)

def e60_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 60
    project_variable.model_number = 5
    project_variable.num_out_channels = [12, 24, 48, 96]
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)

def e61_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 61
    project_variable.model_number = 5
    project_variable.num_out_channels = [12, 24, 48, 96]
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 5; OUT_CHANNELS [14, 28, 56, 112]
# --------------------------------------------------------
def e62_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 62
    project_variable.model_number = 5
    project_variable.num_out_channels = [14, 28, 56, 112]
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)

def e63_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 63
    project_variable.model_number = 5
    project_variable.num_out_channels = [14, 28, 56, 112]
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)

def e64_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 64
    project_variable.model_number = 5
    project_variable.num_out_channels = [14, 28, 56, 112]
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)

def e65_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 65
    project_variable.model_number = 5
    project_variable.num_out_channels = [14, 28, 56, 112]
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 5; OUT_CHANNELS [16, 32, 64, 128]
# --------------------------------------------------------
def e66_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 66
    project_variable.model_number = 5
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)

def e67_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 67
    project_variable.model_number = 5
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)

def e68_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 68
    project_variable.model_number = 5
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)

def e69_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 69
    project_variable.model_number = 5
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 5; OUT_CHANNELS [18, 36, 72, 144]
# --------------------------------------------------------
def e70_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 70
    project_variable.model_number = 5
    project_variable.num_out_channels = [18, 36, 72, 144]
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)

def e71_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 71
    project_variable.model_number = 5
    project_variable.num_out_channels = [18, 36, 72, 144]
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)

def e72_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 72
    project_variable.model_number = 5
    project_variable.num_out_channels = [18, 36, 72, 144]
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)

def e73_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 73
    project_variable.model_number = 5
    project_variable.num_out_channels = [18, 36, 72, 144]
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 5; OUT_CHANNELS [20, 40, 80, 160]
# --------------------------------------------------------
def e74_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 74
    project_variable.model_number = 5
    project_variable.num_out_channels = [20, 40, 80, 160]
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)

def e75_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 75
    project_variable.model_number = 5
    project_variable.num_out_channels = [20, 40, 80, 160]
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e76_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 76
    project_variable.model_number = 5
    project_variable.num_out_channels = [20, 40, 80, 160]
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e77_C3D_kth():
    set_init_3()
    project_variable.experiment_number = 77
    project_variable.model_number = 5
    project_variable.num_out_channels = [20, 40, 80, 160]
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 5; OUT_CHANNELS [8, 16, 32, 64]
# --------------------------------------------------------

def e78_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 78
    project_variable.model_number = 6
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)

def e79_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 79
    project_variable.model_number = 6
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e80_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 80
    project_variable.model_number = 6
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e81_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 81
    project_variable.model_number = 6
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 6; OUT_CHANNELS [10, 20, 40, 80]
# --------------------------------------------------------

def e82_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 82
    project_variable.model_number = 6
    project_variable.num_out_channels = [10, 20, 40, 80]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)

def e83_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 83
    project_variable.model_number = 6
    project_variable.num_out_channels = [10, 20, 40, 80]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)

def e84_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 84
    project_variable.model_number = 6
    project_variable.num_out_channels = [10, 20, 40, 80]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)

def e85_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 85
    project_variable.model_number = 6
    project_variable.num_out_channels = [10, 20, 40, 80]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 6; OUT_CHANNELS [12, 24, 48, 96]
# --------------------------------------------------------
def e86_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 86
    project_variable.model_number = 6
    project_variable.num_out_channels = [12, 24, 48, 96]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e87_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 87
    project_variable.model_number = 6
    project_variable.num_out_channels = [12, 24, 48, 96]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)

def e88_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 88
    project_variable.model_number = 6
    project_variable.num_out_channels = [12, 24, 48, 96]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)

def e89_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 89
    project_variable.model_number = 6
    project_variable.num_out_channels = [12, 24, 48, 96]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 6; OUT_CHANNELS [14, 28, 56, 112]
# --------------------------------------------------------

def e90_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 90
    project_variable.model_number = 6
    project_variable.num_out_channels = [14, 28, 56, 112]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)

def e91_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 91
    project_variable.model_number = 6
    project_variable.num_out_channels = [14, 28, 56, 112]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)

def e92_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 92
    project_variable.model_number = 6
    project_variable.num_out_channels = [14, 28, 56, 112]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e93_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 93
    project_variable.model_number = 6
    project_variable.num_out_channels = [14, 28, 56, 112]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 6; OUT_CHANNELS [16, 32, 64, 128]
# --------------------------------------------------------
def e94_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 94
    project_variable.model_number = 6
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e95_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 95
    project_variable.model_number = 6
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e96_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 96
    project_variable.model_number = 6
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)

def e97_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 97
    project_variable.model_number = 6
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 6; OUT_CHANNELS [18, 36, 72, 144]
# --------------------------------------------------------
def e98_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 98
    project_variable.model_number = 6
    project_variable.num_out_channels = [18, 36, 72, 144]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 2
    main_file.run(project_variable)

def e99_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_number = 99
    project_variable.model_number = 6
    project_variable.num_out_channels = [18, 36, 72, 144]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e100_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 100
    project_variable.model_number = 6
    project_variable.num_out_channels = [18, 36, 72, 144]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e101_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 101
    project_variable.model_number = 6
    project_variable.num_out_channels = [18, 36, 72, 144]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)
# --------------------------------------------------------
#          MODEL 6; OUT_CHANNELS [20, 40, 80, 160]
# --------------------------------------------------------
def e102_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 102
    project_variable.model_number = 6
    project_variable.num_out_channels = [20, 40, 80, 160]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [6, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e103_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 103
    project_variable.model_number = 6
    project_variable.num_out_channels = [20, 40, 80, 160]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [66, 192, 216]
    project_variable.device = 0
    main_file.run(project_variable)

def e104_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 104
    project_variable.model_number = 6
    project_variable.num_out_channels = [20, 40, 80, 160]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [126, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)

def e105_C3DTTN_kth():
    set_init_3()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 105
    project_variable.model_number = 6
    project_variable.num_out_channels = [20, 40, 80, 160]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [191, 192, 216]
    project_variable.device = 1
    main_file.run(project_variable)


def bottleneck():
    set_init_3()
    project_variable.experiment_state = 'crashed'
    project_variable.end_epoch = 5
    project_variable.repeat_experiments = 1
    project_variable.sheet_number = None

    # project_variable.theta_init = 'eye'
    project_variable.experiment_number = 666
    project_variable.model_number = 6
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.data_points = [60, 60, 60]
    project_variable.device = 0
    main_file.run(project_variable)


#####################################################################################################################
#                   LONG EXPERIMENT MODEL 6 FINDING DECENT PARAMETERS 106 - 132
#####################################################################################################################
def set_init_4():
    project_variable.model_number = 6 # experiment version
    project_variable.batch_size = 16
    project_variable.end_epoch = 70
    project_variable.dataset = 'kth_actions'
    project_variable.data_points = [191, 192, 216]
    project_variable.repeat_experiments = 10
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.label_size = 6
    project_variable.optimizer = 'adam'
    project_variable.k_shape = (3, 3, 3)
    project_variable.experiment_state = 'new'
    project_variable.sheet_number = 10


# --------------------------------------------------------
#                   load_num_frames = 30
# --------------------------------------------------------

def e106_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 106
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [4, 8, 16, 32]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e107_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 107
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [4, 8, 16, 32]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e108_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 108
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [4, 8, 16, 32]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


# --
def e109_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 109
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e110_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 110
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e111_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 111
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


# --
def e112_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 112
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e113_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 113
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e114_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 114
    project_variable.device = 0
    project_variable.load_num_frames = 30
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


# --------------------------------------------------------
#                   load_num_frames = 50
# --------------------------------------------------------

def e115_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 115
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [4, 8, 16, 32]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e116_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 116
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [4, 8, 16, 32]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e117_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 117
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [4, 8, 16, 32]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


# --
def e118_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 118
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e119_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 119
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e120_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 120
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


# --
def e121_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 121
    project_variable.device = 1
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e122_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 122
    project_variable.device = 2
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e123_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 123
    project_variable.device = 2
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


# --------------------------------------------------------
#                   load_num_frames = 100
# --------------------------------------------------------

def e124_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 124
    project_variable.device = 2
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [4, 8, 16, 32]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e125_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 125
    project_variable.device = 2
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [4, 8, 16, 32]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e126_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 126
    project_variable.device = 2
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [4, 8, 16, 32]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


# --
def e127_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 127
    project_variable.device = 0
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e128_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 128
    project_variable.device = 0
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e129_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 129
    project_variable.device = 0
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


# --
def e130_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 130
    project_variable.device = 1
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e131_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 131
    project_variable.device = 0
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e132_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 132
    project_variable.device = 1
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)
#####################################################################################################################
#          133-174  LONG EXPERIMENT MODEL 7.x NO MAX-POOL IN TEMPORAL DIMENSION. THETA_INIT NOT NONE
#####################################################################################################################
def set_init_5():
    project_variable.batch_size = 20
    project_variable.end_epoch = 70
    project_variable.dataset = 'kth_actions'
    project_variable.data_points = [191, 192, 216]
    # project_variable.data_points = [6, 6, 6]
    project_variable.repeat_experiments = 10
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.label_size = 6
    project_variable.optimizer = 'adam'
    project_variable.experiment_state = 'new'
    project_variable.sheet_number = 13
    project_variable.max_pool_temporal = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.do_batchnorm = [False, False, False, False, False, False]
# --------------------------------------------------------
#     model_number=71;load_num_frames=30;k_shape[0]=3
# --------------------------------------------------------
def e133_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number =133
    project_variable.device = 0

    project_variable.model_number = 71
    project_variable.load_num_frames = 30
    project_variable.k_shape = (3, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e134_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 134
    project_variable.device = 0

    project_variable.model_number = 71
    project_variable.load_num_frames = 30
    project_variable.k_shape = (3, 3, 3)

    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e135_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 135
    project_variable.device = 0

    project_variable.model_number = 71
    project_variable.load_num_frames = 30
    project_variable.k_shape = (3, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e136_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 136
    project_variable.device = 0

    project_variable.model_number = 71
    project_variable.load_num_frames = 30
    project_variable.k_shape = (3, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e137_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 137
    project_variable.device = 0

    project_variable.model_number = 71
    project_variable.load_num_frames = 30
    project_variable.k_shape = (3, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e138_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 138
    project_variable.device = 0

    project_variable.model_number = 71
    project_variable.load_num_frames = 30
    project_variable.k_shape = (3, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

# --------------------------------------------------------
#     model_number=72;load_num_frames=30;k_shape[0]=5
# --------------------------------------------------------
def e139_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 139
    project_variable.device = 0

    project_variable.model_number = 72
    project_variable.load_num_frames = 30
    project_variable.k_shape = (5, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e140_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 140
    project_variable.device = 0

    project_variable.model_number = 72
    project_variable.load_num_frames = 30
    project_variable.k_shape = (5, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e141_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 141
    project_variable.device = 1

    project_variable.model_number = 72
    project_variable.load_num_frames = 30
    project_variable.k_shape = (5, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e142_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 142
    project_variable.device = 1

    project_variable.model_number = 72
    project_variable.load_num_frames = 30
    project_variable.k_shape = (5, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e143_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 143
    project_variable.device = 1

    project_variable.model_number = 72
    project_variable.load_num_frames = 30
    project_variable.k_shape = (5, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e144_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 144
    project_variable.device = 1

    project_variable.model_number = 72
    project_variable.load_num_frames = 30
    project_variable.k_shape = (5, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

# --------------------------------------------------------
#     model_number=73;load_num_frames=30;k_shape[0]=7
# --------------------------------------------------------
def e145_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 145
    project_variable.device = 1

    project_variable.model_number = 73
    project_variable.load_num_frames = 30
    project_variable.k_shape = (7, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e146_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 146
    project_variable.device = 1

    project_variable.model_number = 73
    project_variable.load_num_frames = 30
    project_variable.k_shape = (7, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e147_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 147
    project_variable.device = 1

    project_variable.model_number = 73
    project_variable.load_num_frames = 30
    project_variable.k_shape = (7, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e148_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 148
    project_variable.device = 1

    project_variable.model_number = 73
    project_variable.load_num_frames = 30
    project_variable.k_shape = (7, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e149_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 149
    project_variable.device = 2

    project_variable.model_number = 73
    project_variable.load_num_frames = 30
    project_variable.k_shape = (7, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e150_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 150
    project_variable.device = 2

    project_variable.model_number = 73
    project_variable.load_num_frames = 30
    project_variable.k_shape = (7, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)
# --------------------------------------------------------
#     model_number=74;load_num_frames=30;k_shape[0]=9
# --------------------------------------------------------

def e151_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 151
    project_variable.device = 2

    project_variable.model_number = 74
    project_variable.load_num_frames = 30
    project_variable.k_shape = (9, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e152_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 152
    project_variable.device = 2

    project_variable.model_number = 74
    project_variable.load_num_frames = 30
    project_variable.k_shape = (9, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e153_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 153
    project_variable.device = 2

    project_variable.model_number = 74
    project_variable.load_num_frames = 30
    project_variable.k_shape = (9, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e154_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 154
    project_variable.device = 2

    project_variable.model_number = 74
    project_variable.load_num_frames = 30
    project_variable.k_shape = (9, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e155_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 155
    project_variable.device = 2

    project_variable.model_number = 74
    project_variable.load_num_frames = 30
    project_variable.k_shape = (9, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e156_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 156
    project_variable.device = 2

    project_variable.model_number = 74
    project_variable.load_num_frames = 30
    project_variable.k_shape = (9, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)
# --------------------------------------------------------
#    model_number=75;load_num_frames=100;k_shape[0]=7
# --------------------------------------------------------

def e157_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 157
    project_variable.device = 1

    project_variable.model_number = 75
    project_variable.load_num_frames = 100
    project_variable.k_shape = (7, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e158_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 158
    project_variable.device = 1

    project_variable.model_number = 75
    project_variable.load_num_frames = 100
    project_variable.k_shape = (7, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e159_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 159
    project_variable.device = 0

    project_variable.model_number = 75
    project_variable.load_num_frames = 100
    project_variable.k_shape = (7, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e160_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 160
    project_variable.device = 0

    project_variable.model_number = 75
    project_variable.load_num_frames = 100
    project_variable.k_shape = (7, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e161_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 161
    project_variable.device = 0

    project_variable.model_number = 75
    project_variable.load_num_frames = 100
    project_variable.k_shape = (7, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e162_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 162
    project_variable.device = 0

    project_variable.model_number = 75
    project_variable.load_num_frames = 100
    project_variable.k_shape = (7, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)
# --------------------------------------------------------
#    model_number=76;load_num_frames=100;k_shape[0]=9
# --------------------------------------------------------

def e163_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 163
    project_variable.device = 0

    project_variable.model_number = 76
    project_variable.load_num_frames = 100
    project_variable.k_shape = (9, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e164_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 164
    project_variable.device = 1

    project_variable.model_number = 76
    project_variable.load_num_frames = 100
    project_variable.k_shape = (9, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e165_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 165
    project_variable.device = 1

    project_variable.model_number = 76
    project_variable.load_num_frames = 100
    project_variable.k_shape = (9, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e166_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 166
    project_variable.device = 1

    project_variable.model_number = 76
    project_variable.load_num_frames = 100
    project_variable.k_shape = (9, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e167_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 167
    project_variable.device = 2

    project_variable.model_number = 76
    project_variable.load_num_frames = 100
    project_variable.k_shape = (9, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e168_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 168
    project_variable.device = 2

    project_variable.model_number = 76
    project_variable.load_num_frames = 100
    project_variable.k_shape = (9, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)
# --------------------------------------------------------
#   model_number=77;load_num_frames=100;k_shape[0]=11
# --------------------------------------------------------
def e169_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 169
    project_variable.device = 2

    project_variable.model_number = 77
    project_variable.load_num_frames = 100
    project_variable.k_shape = (11, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e170_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 170
    project_variable.device = 2

    project_variable.model_number = 77
    project_variable.load_num_frames = 100
    project_variable.k_shape = (11, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e171_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 171
    project_variable.device = 2

    project_variable.model_number = 77
    project_variable.load_num_frames = 100
    project_variable.k_shape = (11, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e172_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 172
    project_variable.device = 0

    project_variable.model_number = 77
    project_variable.load_num_frames = 100
    project_variable.k_shape = (11, 3, 3)
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e173_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 173
    project_variable.device = 0

    project_variable.model_number = 77
    project_variable.load_num_frames = 100
    project_variable.k_shape = (11, 3, 3)
    project_variable.theta_init = 'eye-like'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e174_C3DTTN_kth():
    set_init_5()
    project_variable.experiment_number = 174
    project_variable.device = 0

    project_variable.model_number = 77
    project_variable.load_num_frames = 100
    project_variable.k_shape = (11, 3, 3)
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)


#####################################################################################################################
#                       175-       back to beginnings: experiments with 1 layer Conv TTN
#####################################################################################################################
def set_init_6():
    project_variable.model_number = 8
    project_variable.end_epoch = 150
    project_variable.dataset = 'kth_actions'
    project_variable.data_points = [191, 192, 216]
    project_variable.repeat_experiments = 10
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.label_size = 6
    project_variable.optimizer = 'sgd'
    project_variable.experiment_state = 'new'
    project_variable.sheet_number = 14
    project_variable.model_number = 8
    project_variable.weight_transform = 'seq'
    project_variable.theta_init = 'eye'
    project_variable.srxy_init = 'eye'
# --------------------------------------------------------
#  find good LR, batchsize, out_channels, k_shape 175-280
# --------------------------------------------------------
def finding_good_values():
    project_variable.k0_init = 'kaiming-normal'
    project_variable.load_num_frames = 30
    
# --------------------------------------------------------
#                   LR = 0.000001
# --------------------------------------------------------
#------ 1
def e175_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 175
    project_variable.device = 0

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e176_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 176
    project_variable.device = 0

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e177_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 177
    project_variable.device = 0

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 2
def e178_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 178
    project_variable.device = 0

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e179_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 179
    project_variable.device = 0

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e180_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 180
    project_variable.device = 1

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 3
def e181_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 181
    project_variable.device = 1

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e182_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 182
    project_variable.device = 1

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e183_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 183
    project_variable.device = 1

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
#------ 1
def e184_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 184
    project_variable.device = 1

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e185_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 185
    project_variable.device = 1

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e186_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 186
    project_variable.device = 1

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 2
def e187_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 187
    project_variable.device = 2

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e188_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 188
    project_variable.device = 2

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e189_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 189
    project_variable.device = 2

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 3
def e190_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 190
    project_variable.device = 2

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e191_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 191
    project_variable.device = 0

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e192_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 192
    project_variable.device = 0

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
#------ 1
def e193_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 193
    project_variable.device = 0

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e194_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 194
    project_variable.device = 0

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e195_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 195
    project_variable.device = 0

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 2
def e196_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 196
    project_variable.device = 0

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e197_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 197
    project_variable.device = 1

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e198_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 198
    project_variable.device = 1

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 3
def e199_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 199
    project_variable.device = 1

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e200_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 200
    project_variable.device = 1

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e201_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 201
    project_variable.device = 1

    project_variable.learning_rate = 0.000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# --------------------------------------------------------
#                   LR = 0.0000001
# --------------------------------------------------------
#------ 1
def e202_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 202
    project_variable.device = 1

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e203_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 203
    project_variable.device = 1

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e204_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 204
    project_variable.device = 1

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 2
def e205_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 205
    project_variable.device = 2

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e206_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 206
    project_variable.device = 2

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e207_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 207
    project_variable.device = 2

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 3
def e208_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 208
    project_variable.device = 2

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e209_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 209
    project_variable.device = 2

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e210_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 210
    project_variable.device = 2

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
#------ 1
def e211_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 211
    project_variable.device = 2

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e212_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 212
    project_variable.device = 0

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e213_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 213
    project_variable.device = 0

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 2
def e214_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 214
    project_variable.device = 0

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e215_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 215
    project_variable.device = 0

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e216_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 216
    project_variable.device = 0

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 3
def e217_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 217
    project_variable.device = 1

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e218_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 218
    project_variable.device = 1

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e219_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 219
    project_variable.device = 1

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
#------ 1
def e220_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 220
    project_variable.device = 1

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e221_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 221
    project_variable.device = 1

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e222_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 222
    project_variable.device = 1

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 2
def e223_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 223
    project_variable.device = 1

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e224_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 224
    project_variable.device = 1

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e225_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 225
    project_variable.device = 2

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 3
def e226_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 226
    project_variable.device = 2

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e227_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 227
    project_variable.device = 2

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e228_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 228
    project_variable.device = 2

    project_variable.learning_rate = 0.0000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# --------------------------------------------------------
#                   LR = 0.00000001
# --------------------------------------------------------
#------ 1
def e229_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 229
    project_variable.device = 2

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e230_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 230
    project_variable.device = 2

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e231_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 231
    project_variable.device = 2

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 2
def e232_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 232
    project_variable.device = 2

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e233_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 233
    project_variable.device = 0

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e234_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 234
    project_variable.device = 0

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 3
def e235_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 235
    project_variable.device = 0

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e236_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 236
    project_variable.device = 0

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e237_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 237
    project_variable.device = 1

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (3, 3, 3)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
#------ 1
def e238_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 238
    project_variable.device = 1

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e239_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 239
    project_variable.device = 1

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e240_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 240
    project_variable.device = 2

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 2
def e241_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 241
    project_variable.device = 2

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e242_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 242
    project_variable.device = 2

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e243_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 243
    project_variable.device = 2

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 3
def e244_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 244
    project_variable.device = 0

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# TODO
def e245_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 245
    project_variable.device = 0

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e246_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 246
    project_variable.device = 1

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (5, 5, 5)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
#------ 1
def e247_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 247
    project_variable.device = 1

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e248_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 248
    project_variable.device = 1

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e249_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 249
    project_variable.device = 1

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [6]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 2
def e250_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 250
    project_variable.device = 1

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e251_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 251
    project_variable.device = 1

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e252_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 252
    project_variable.device = 1

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [12]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 3
def e253_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 253
    project_variable.device = 1

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e254_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 254
    project_variable.device = 2

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e255_C3DTTN_1L_kth():
    set_init_6()
    finding_good_values()
    project_variable.experiment_number = 255
    project_variable.device = 2

    project_variable.learning_rate = 0.00000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [18]
    project_variable.k_shape = (7, 7, 7)

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

# --------------------------------------------------------
#  find good LR, batchsize, out_channels 256 - 267
# --------------------------------------------------------
def set_init_7():
    project_variable.model_number = 8
    project_variable.end_epoch = 300
    project_variable.dataset = 'kth_actions'
    project_variable.data_points = [191, 192, 216]
    project_variable.repeat_experiments = 10
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.label_size = 6
    project_variable.optimizer = 'sgd'
    project_variable.experiment_state = 'new'
    project_variable.sheet_number = 14
    project_variable.model_number = 8
    project_variable.weight_transform = 'seq'
    project_variable.theta_init = 'eye'
    project_variable.srxy_init = 'eye'
    project_variable.k0_init = 'kaiming-normal'
    project_variable.load_num_frames = 30
    project_variable.k_shape = (7, 7, 7)



# --------------------------------------------------------
#                   LR = 0.000000001
# --------------------------------------------------------
# ------ 1
def e256_C3DTTN_1L_kth():
    set_init_7()
    project_variable.experiment_number = 256
    project_variable.device = 0

    project_variable.learning_rate = 0.000000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [18]

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e257_C3DTTN_1L_kth():
    set_init_7()
    project_variable.experiment_number = 257
    project_variable.device = 0

    project_variable.learning_rate = 0.000000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [18]

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e258_C3DTTN_1L_kth():
    set_init_7()
    project_variable.experiment_number = 258
    project_variable.device = 0

    project_variable.learning_rate = 0.000000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [18]

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 2
def e259_C3DTTN_1L_kth():
    set_init_7()
    project_variable.experiment_number = 259
    project_variable.device = 1

    project_variable.learning_rate = 0.000000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [24]

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e260_C3DTTN_1L_kth():
    set_init_7()
    project_variable.experiment_number = 260
    project_variable.device = 1

    project_variable.learning_rate = 0.000000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [24]

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e261_C3DTTN_1L_kth():
    set_init_7()
    project_variable.experiment_number = 261
    project_variable.device = 1

    project_variable.learning_rate = 0.000000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [24]

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

# --------------------------------------------------------
#                   LR = 0.0000000001
# --------------------------------------------------------
# ------ 1
def e262_C3DTTN_1L_kth():
    set_init_7()
    project_variable.experiment_number = 262
    project_variable.device = 1

    project_variable.learning_rate = 0.0000000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [18]

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e263_C3DTTN_1L_kth():
    set_init_7()
    project_variable.experiment_number = 263
    project_variable.device = 1

    project_variable.learning_rate = 0.0000000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [18]

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e264_C3DTTN_1L_kth():
    set_init_7()
    project_variable.experiment_number = 264
    project_variable.device = 1

    project_variable.learning_rate = 0.0000000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [18]

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)
# ------ 2
def e265_C3DTTN_1L_kth():
    set_init_7()
    project_variable.experiment_number = 265
    project_variable.device = 1

    project_variable.learning_rate = 0.0000000001
    project_variable.batch_size = 12
    project_variable.num_out_channels = [24]

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e266_C3DTTN_1L_kth():
    set_init_7()
    project_variable.experiment_number = 266
    project_variable.device = 2

    project_variable.learning_rate = 0.0000000001
    project_variable.batch_size = 16
    project_variable.num_out_channels = [24]

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

def e267_C3DTTN_1L_kth():
    set_init_7()
    project_variable.experiment_number = 267
    project_variable.device = 2

    project_variable.learning_rate = 0.0000000001
    project_variable.batch_size = 32
    project_variable.num_out_channels = [24]

    project_variable.transformations_per_filter = project_variable.k_shape[0] - 1
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels

    main_file.run(project_variable)

project_variable = ProjectVariable(debug_mode=False)
# cProfile.run('bottleneck()', sort='cumtime')

# e256_C3DTTN_1L_kth()
# e257_C3DTTN_1L_kth()
# e258_C3DTTN_1L_kth()
# e259_C3DTTN_1L_kth()

# e260_C3DTTN_1L_kth()
# e261_C3DTTN_1L_kth()
# e262_C3DTTN_1L_kth()
# e263_C3DTTN_1L_kth()
# e264_C3DTTN_1L_kth()
# e265_C3DTTN_1L_kth()
# e266_C3DTTN_1L_kth()
# e267_C3DTTN_1L_kth()