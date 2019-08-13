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


# -- TODO
def e127_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 127
    project_variable.device = 2
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e128_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 128
    project_variable.device = 2
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e129_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 129
    project_variable.device = 2
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
    project_variable.device = 2
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e131_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 131
    project_variable.device = 2
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


def e132_C3DTTN_kth():
    set_init_4()
    project_variable.experiment_number = 132
    project_variable.device = 2
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=False)
# cProfile.run('bottleneck()', sort='cumtime')


# e106_C3DTTN_kth()
# e107_C3DTTN_kth()
# e108_C3DTTN_kth()
# e109_C3DTTN_kth()
# e110_C3DTTN_kth()
# e111_C3DTTN_kth()
# e112_C3DTTN_kth()
# e113_C3DTTN_kth()
# e114_C3DTTN_kth()
# e115_C3DTTN_kth()
# e116_C3DTTN_kth()
# e117_C3DTTN_kth()
# e118_C3DTTN_kth()
# e119_C3DTTN_kth()
# e120_C3DTTN_kth()
# e121_C3DTTN_kth()
# e122_C3DTTN_kth()
# e123_C3DTTN_kth()
# e124_C3DTTN_kth()
# e125_C3DTTN_kth()
# e126_C3DTTN_kth()
e127_C3DTTN_kth()
# e128_C3DTTN_kth()
# e129_C3DTTN_kth()
# e130_C3DTTN_kth()
# e131_C3DTTN_kth()
# e132_C3DTTN_kth()