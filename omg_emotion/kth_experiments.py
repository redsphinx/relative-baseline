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
    project_variable.model_number = 5
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
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 44
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 3
    project_variable.do_batchnorm = [False, False, True, False, False]
    main_file.run(project_variable)

def e45_C3D_kth():
    set_init_2()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 45
    project_variable.device = 1
    project_variable.num_out_channels = [8, 16, 32, 64]
    project_variable.conv1_k_t = 3
    project_variable.do_batchnorm = [True, True, False, False, False]
    main_file.run(project_variable)

def e46_C3D_kth():
    set_init_2()
    project_variable.experiment_state = 'crashed'
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
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 47
    project_variable.device = 1
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    project_variable.do_batchnorm = [False, False, True, False, False]
    main_file.run(project_variable)

def e48_C3D_kth():
    set_init_2()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 48
    project_variable.device = 2
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    project_variable.do_batchnorm = [True, True, False, False, False]
    main_file.run(project_variable)

def e49_C3D_kth():
    set_init_2()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 49
    project_variable.device = 2
    project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.conv1_k_t = 5
    project_variable.do_batchnorm = [True, True, True, False, False]
    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=False)


# e44_C3D_kth()
# e45_C3D_kth()
e46_C3D_kth()
# e47_C3D_kth()
# e48_C3D_kth()
# e49_C3D_kth()
