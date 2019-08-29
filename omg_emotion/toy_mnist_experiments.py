from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


def pilot():
    project_variable.device = 2
    project_variable.model_number = 1
    project_variable.experiment_number = 0

    project_variable.batch_size = 20

    # project_variable.save_model = False
    # project_variable.save_data = False

    main_file.run(project_variable)


def dummy_data():
    project_variable.device = 2
    project_variable.model_number = 2
    project_variable.experiment_number = 1
    project_variable.batch_size = 20
    project_variable.dataset = 'dummy'
    main_file.run(project_variable)


def conv3dttnpilot():
    project_variable.device = 2
    project_variable.model_number = 3
    project_variable.experiment_number = 1
    project_variable.batch_size = 30
    project_variable.dataset = 'dummy'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.01

    project_variable.end_epoch = 20
    project_variable.theta_init = None

    main_file.run(project_variable)


def conv3dttn_mmnist_pilot():
    project_variable.device = 2
    project_variable.model_number = 3
    project_variable.experiment_number = 1
    project_variable.batch_size = 30
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001

    project_variable.end_epoch = 20
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'naive'

    main_file.run(project_variable)


def conv3d_mnist():
    project_variable.device = 2
    project_variable.model_number = 2
    project_variable.experiment_number = 1
    project_variable.batch_size = 30
    project_variable.end_epoch = 100
    project_variable.dataset = 'mov_mnist'
    main_file.run(project_variable)

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

def e1_conv3d_mnist():
    # regular 3d conv on mnist
    project_variable.experiment_number = 1
    project_variable.model_number = 2

    project_variable.device = 2
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001

    main_file.run(project_variable)


def e2_conv3dttn_mnist():
    # 3dttn with 'normal' theta init
    project_variable.experiment_number = 2
    project_variable.model_number = 3

    project_variable.device = 2
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = 'normal'

    main_file.run(project_variable)


def e3_conv3dttn_mnist():
    # 3dttn with 'eye' theta init
    project_variable.experiment_number = 3
    project_variable.model_number = 3

    project_variable.device = 2
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = 'eye'

    main_file.run(project_variable)


def e4_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 4
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None

    main_file.run(project_variable)

def e5_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 5
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e6_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 6
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye-like'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)

def e7_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 7
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e8_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 8
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye-like'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e9_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 9
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'normal'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)

def e10_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 10
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid'

    main_file.run(project_variable)


def e11_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 11
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid_small'

    main_file.run(project_variable)


def e12_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 12
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye-like'
    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid_small'

    main_file.run(project_variable)


def e13_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 13
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'normal'
    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid_small'

    main_file.run(project_variable)


########################################################################################################################
########################################################################################################################

def e14_conv3d_mnist():
    # regular 3d conv on mnist
    project_variable.experiment_number = 14
    project_variable.model_number = 2

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001

    main_file.run(project_variable)


def e15_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 15
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None

    main_file.run(project_variable)


def e16_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 16
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)


def e17_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 17
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye-like'
    project_variable.weight_transform = 'naive'

    main_file.run(project_variable)


def e18_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 18
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)


def e19_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 19
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye-like'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)


def e20_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 20
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'normal'
    project_variable.weight_transform = 'seq'

    main_file.run(project_variable)


def e21_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 21
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid'

    main_file.run(project_variable)


def e22_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 22
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid_small'

    main_file.run(project_variable)


def e23_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 23
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye-like'
    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid_small'

    main_file.run(project_variable)


def e24_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 24
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'normal'
    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid_small'

    main_file.run(project_variable)


def e25_lenet2d_mnist():
    project_variable.device = 2
    project_variable.model_number = 1
    project_variable.experiment_number = 25
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.optimizer = 'adam'

    main_file.run(project_variable)

def e26_lenet2d_mnist():
    project_variable.device = 2
    project_variable.model_number = 1
    project_variable.experiment_number = 26
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.optimizer = 'sgd'

    main_file.run(project_variable)


# TODO: rerun
def e27_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 27
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 5
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None

    project_variable.k0_init = 'ones'

    project_variable.repeat_experiments = 3
    project_variable.randomize_training_data = True



    main_file.run(project_variable)

# TODO: rerun
def e28_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 28
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None

    project_variable.k0_init = 'ones_var'

    main_file.run(project_variable)

# TODO: rerun
def e29_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 29
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None

    project_variable.k0_init = 'uniform'

    main_file.run(project_variable)


def this_debug():
    project_variable.experiment_number = 666
    project_variable.model_number = 2

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 10
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None

    project_variable.randomize_training_data = True
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [50, 100, 100]
    project_variable.repeat_experiments = 1

    main_file.run(project_variable)


#####################################################################################################################
#                                   LONG EXPERIMENT START: 30 - 53
#####################################################################################################################

def set_init():
    project_variable.model_number = 3
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.srxy_smoothness = 'sigmoid_small'
    project_variable.weight_transform = 'seq'
    project_variable.k0_init = 'normal'
    project_variable.randomize_training_data = True
    project_variable.repeat_experiments = 10

# --------------------------------------------------------
#                   out_channels = [6, 16]
# --------------------------------------------------------

def e30_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 30
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e31_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 31
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e32_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 32
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e33_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 33
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e34_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 34
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e35_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 35
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e36_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 36
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e37_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 37
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [7, 17]
# --------------------------------------------------------

def e38_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 38
    project_variable.device = 0
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e39_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 39
    project_variable.device = 0
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e40_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 40
    project_variable.device = 0
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e41_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 41
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e42_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 42
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e43_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 43
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e44_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 44
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e45_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 45
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [8, 18]
# --------------------------------------------------------

def e46_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 46
    project_variable.device = 1
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e47_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 47
    project_variable.device = 1
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e48_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 48
    project_variable.device = 0
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e49_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 49
    project_variable.device = 0
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e50_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 50
    project_variable.device = 0
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e51_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 51
    project_variable.device = 0
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e52_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 52
    project_variable.device = 0
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e53_conv3dttn_mnist():
    set_init()
    project_variable.experiment_number = 53
    project_variable.device = 0
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)
    
#####################################################################################################################
#                                   LONG EXPERIMENT START: 54 - 77
#####################################################################################################################

def set_init_2():
    project_variable.model_number = 2
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.randomize_training_data = True
    project_variable.repeat_experiments = 10
    project_variable.sheet_number = 2

# --------------------------------------------------------
#                   out_channels = [6, 16]
# --------------------------------------------------------

def e54_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 54
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e55_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 55
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e56_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 56
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e57_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 57
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e58_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 58
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e59_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 59
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e60_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 60
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e61_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 61
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [7, 17]
# --------------------------------------------------------

def e62_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 62
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e63_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 63
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e64_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 64
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e65_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 65
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e66_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 66
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e67_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 67
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e68_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 68
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e69_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 69
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [8, 18]
# --------------------------------------------------------

def e70_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 70
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e71_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 71
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e72_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 72
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e73_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 73
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e74_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 74
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e75_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 75
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e76_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 76
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e77_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 77
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)


#####################################################################################################################
#                                   LONG EXPERIMENT START: 78 - 100
#####################################################################################################################
def set_init_0():
    project_variable.model_number = 3
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [100, 200, 200]
    project_variable.k0_init = 'normal'
    project_variable.randomize_training_data = True
    project_variable.repeat_experiments = 10
    project_variable.sheet_number = 0

# --------------------------------------------------------
#                   baseline with 3DConv
# --------------------------------------------------------
def e78_conv3d_mnist():
    set_init_0()
    project_variable.experiment_number = 78
    project_variable.device = 0
    project_variable.model_number = 2
    main_file.run(project_variable)
    
# --------------------------------------------------------
#                   theta_init not None
# --------------------------------------------------------
def e79_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 79
    project_variable.device = 0
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'naive'
    project_variable.srxy_init = 'normal'
    project_variable.srxy_smoothness = None
    main_file.run(project_variable)

def e80_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 80
    project_variable.device = 0
    project_variable.theta_init = 'normal'
    project_variable.weight_transform = 'seq'
    project_variable.srxy_init = 'normal'
    project_variable.srxy_smoothness = None
    main_file.run(project_variable)


def e81_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 81
    project_variable.device = 0
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'naive'
    project_variable.srxy_init = 'normal'
    project_variable.srxy_smoothness = None
    main_file.run(project_variable)


def e82_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 82
    project_variable.device = 0
    project_variable.theta_init = 'eye'
    project_variable.weight_transform = 'seq'
    project_variable.srxy_init = 'normal'
    project_variable.srxy_smoothness = None
    main_file.run(project_variable)

# --------------------------------------------------------
#                   srxy normal
# --------------------------------------------------------
def e83_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 83
    project_variable.device = 0
    project_variable.theta_init = None
    project_variable.srxy_init = 'normal'
    
    project_variable.weight_transform = 'naive'
    project_variable.srxy_smoothness = None
    
    main_file.run(project_variable)


def e84_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 84
    project_variable.device = 0
    project_variable.theta_init = None
    project_variable.srxy_init = 'normal'

    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = None

    main_file.run(project_variable)

def e85_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 85
    project_variable.device = 0
    project_variable.theta_init = None
    project_variable.srxy_init = 'normal'

    project_variable.weight_transform = 'naive'
    project_variable.srxy_smoothness = 'sigmoid'

    main_file.run(project_variable)

def e86_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 86
    project_variable.device = 0
    project_variable.theta_init = None
    project_variable.srxy_init = 'normal'

    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid'

    main_file.run(project_variable)

def e87_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 87
    project_variable.device = 0
    project_variable.theta_init = None
    project_variable.srxy_init = 'normal'

    project_variable.weight_transform = 'naive'
    project_variable.srxy_smoothness = 'sigmoid_small'

    main_file.run(project_variable)

def e88_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 88
    project_variable.device = 1
    project_variable.theta_init = None
    project_variable.srxy_init = 'normal'

    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid_small'

    main_file.run(project_variable)


# --------------------------------------------------------
#                   srxy eye
# --------------------------------------------------------
def e89_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 89
    project_variable.device = 1
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'

    project_variable.weight_transform = 'naive'
    project_variable.srxy_smoothness = None

    main_file.run(project_variable)


def e90_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 90
    project_variable.device = 1
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'

    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = None

    main_file.run(project_variable)


def e91_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 91
    project_variable.device = 1
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'

    project_variable.weight_transform = 'naive'
    project_variable.srxy_smoothness = 'sigmoid'

    main_file.run(project_variable)


def e92_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 92
    project_variable.device = 1
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'

    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid'

    main_file.run(project_variable)


def e93_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 93
    project_variable.device = 1
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'

    project_variable.weight_transform = 'naive'
    project_variable.srxy_smoothness = 'sigmoid_small'

    main_file.run(project_variable)


def e94_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 94
    project_variable.device = 1
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'

    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid_small'

    main_file.run(project_variable)

# --------------------------------------------------------
#                   srxy eye-like
# --------------------------------------------------------
def e95_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 95
    project_variable.device = 2
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye-like'

    project_variable.weight_transform = 'naive'
    project_variable.srxy_smoothness = None

    main_file.run(project_variable)


def e96_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 96
    project_variable.device = 2
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye-like'

    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = None

    main_file.run(project_variable)


def e97_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 97
    project_variable.device = 2
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye-like'

    project_variable.weight_transform = 'naive'
    project_variable.srxy_smoothness = 'sigmoid'

    main_file.run(project_variable)


def e98_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 98
    project_variable.device = 2
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye-like'

    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid'

    main_file.run(project_variable)


def e99_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 99
    project_variable.device = 2
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye-like'

    project_variable.weight_transform = 'naive'
    project_variable.srxy_smoothness = 'sigmoid_small'

    main_file.run(project_variable)


def e100_conv3dttn_mnist():
    set_init_0()
    project_variable.experiment_number = 100
    project_variable.device = 2
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye-like'

    project_variable.weight_transform = 'seq'
    project_variable.srxy_smoothness = 'sigmoid_small'

    main_file.run(project_variable)

#####################################################################################################################
#                                   LONG EXPERIMENT START: 101 - 124
#####################################################################################################################

def set_init_3():
    project_variable.model_number = 3
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.srxy_smoothness = 'sigmoid'
    project_variable.weight_transform = 'naive'
    project_variable.k0_init = 'normal'
    project_variable.randomize_training_data = True
    project_variable.repeat_experiments = 10
    project_variable.sheet_number = 3

# --------------------------------------------------------
#                   out_channels = [6, 16]
# --------------------------------------------------------

def e101_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 101
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e102_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 102
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e103_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 103
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e104_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 104
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e105_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 105
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e106_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 106
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e107_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 107
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e108_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 108
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [7, 17]
# --------------------------------------------------------

def e109_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 109
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e110_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 110
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e111_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 111
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e112_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 112
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e113_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 113
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e114_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 114
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e115_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 115
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e116_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 116
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [8, 18]
# --------------------------------------------------------

def e117_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 117
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e118_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 118
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e119_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 119
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e120_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 120
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e121_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 121
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e122_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 122
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e123_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 123
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e124_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 124
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)


def debug():
    project_variable.experiment_number = 666
    project_variable.model_number = 2

    project_variable.device = 0
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = 'eye'

    project_variable.randomize_training_data = True
    project_variable.data_points = [100, 200, 200]
    project_variable.repeat_experiments = 10

    project_variable.transformation_groups = [1, 1]
    project_variable.k0_groups = [1, 1]
    project_variable.num_out_channels = [1, 1]

    main_file.run(project_variable)

#####################################################################################################################
#                                   LONG EXPERIMENT START: 125 - 136
#####################################################################################################################
def set_init_4():
    project_variable.model_number = 3
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.randomize_training_data = True
    project_variable.data_points = [100, 5000, 5000]
    project_variable.repeat_experiments = 10
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.srxy_smoothness = 'sigmoid'
    project_variable.weight_transform = 'naive'
    project_variable.k0_init = 'normal'
    project_variable.sheet_number = 4

# --------------------------------------------------------
#                   out_channels = [1, 1]
# --------------------------------------------------------
def e125_conv3d_mnist():
    set_init_4()
    project_variable.experiment_number = 125
    project_variable.model_number = 2
    project_variable.device = 0
    project_variable.num_out_channels = [1, 1]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)

def e126_conv3dttn_mnist():
    set_init_4()
    project_variable.experiment_number = 126
    project_variable.device = 0
    project_variable.num_out_channels = [1, 1]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)

# --------------------------------------------------------
#        out_channels = [6, 16] baselines 5000 test
# --------------------------------------------------------
def e127_conv3d_mnist():
    set_init_4()
    project_variable.experiment_number = 127
    project_variable.model_number = 2
    project_variable.device = 0
    main_file.run(project_variable)

def e128_conv3dttn_mnist():
    set_init_4()
    project_variable.experiment_number = 128
    project_variable.device = 0
    main_file.run(project_variable)

# --------------------------------------------------------
# transformation_groups = [1, 1], [2, 4], [3, 8], [4, 12]
# --------------------------------------------------------
def e129_conv3dttn_mnist():
    set_init_4()
    project_variable.experiment_number = 129
    project_variable.device = 0
    project_variable.transformation_groups = [1, 1]
    main_file.run(project_variable)

def e130_conv3dttn_mnist():
    set_init_4()
    project_variable.experiment_number = 130
    project_variable.device = 0
    project_variable.transformation_groups = [2, 4]
    main_file.run(project_variable)
    
def e131_conv3dttn_mnist():
    set_init_4()
    project_variable.experiment_number = 131
    project_variable.device = 0
    project_variable.transformation_groups = [3, 8]
    main_file.run(project_variable)

def e132_conv3dttn_mnist():
    set_init_4()
    project_variable.experiment_number = 132
    project_variable.device = 0
    project_variable.transformation_groups = [4, 12]
    main_file.run(project_variable)

# --------------------------------------------------------
#       k0_groups = [1, 1], [2, 4], [3, 8], [4, 12]
# --------------------------------------------------------
def e133_conv3dttn_mnist():
    set_init_4()
    project_variable.experiment_number = 133
    project_variable.device = 0
    project_variable.k0_groups = [1, 1]
    main_file.run(project_variable)

def e134_conv3dttn_mnist():
    set_init_4()
    project_variable.experiment_number = 134
    project_variable.device = 0
    project_variable.k0_groups = [2, 4]
    main_file.run(project_variable)

def e135_conv3dttn_mnist():
    set_init_4()
    project_variable.experiment_number = 135
    project_variable.device = 0
    project_variable.k0_groups = [3, 8]
    main_file.run(project_variable)

def e136_conv3dttn_mnist():
    set_init_4()
    project_variable.experiment_number = 136
    project_variable.device = 1
    project_variable.k0_groups = [4, 12]
    main_file.run(project_variable)

#####################################################################################################################
#                                   LONG EXPERIMENT START: 137 - 160
#####################################################################################################################

def set_init_5():
    project_variable.model_number = 3
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.srxy_smoothness = None
    project_variable.weight_transform = 'seq'
    project_variable.k0_init = 'normal'
    project_variable.randomize_training_data = True
    project_variable.repeat_experiments = 10
    project_variable.sheet_number = 5

# --------------------------------------------------------
#                   out_channels = [6, 16]
# --------------------------------------------------------

def e137_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 137
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e138_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 138
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e139_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 139
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e140_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 140
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e141_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 141
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e142_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 142
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e143_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 143
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e144_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 144
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [7, 17]
# --------------------------------------------------------

def e145_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 145
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e146_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 146
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e147_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 147
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e148_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 148
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e149_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 149
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e150_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 150
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e151_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 151
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e152_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 152
    project_variable.device = 1
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [8, 18]
# --------------------------------------------------------

def e153_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 153
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e154_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 154
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e155_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 155
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e156_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 156
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e157_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 157
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e158_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 158
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e159_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 159
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e160_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 160
    project_variable.device = 2
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)


#####################################################################################################################
#                                   LONG EXPERIMENT START: 161 - 170
#####################################################################################################################
def set_init_6():
    project_variable.model_number = 3
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.randomize_training_data = True
    project_variable.data_points = [100, 1000, 1000]
    project_variable.repeat_experiments = 10
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.srxy_smoothness = None
    project_variable.weight_transform = 'seq'
    project_variable.k0_init = 'normal'
    project_variable.sheet_number = 6


# --------------------------------------------------------
#                   out_channels = [1, 1]
# --------------------------------------------------------
def e161_conv3dttn_mnist():
    set_init_6()
    project_variable.experiment_number = 161
    project_variable.device = 0
    project_variable.num_out_channels = [1, 1]
    project_variable.transformation_groups = project_variable.num_out_channels
    project_variable.k0_groups = project_variable.num_out_channels
    main_file.run(project_variable)


# --------------------------------------------------------
#        out_channels = [6, 16] baselines 1000 test
# --------------------------------------------------------
def e162_conv3dttn_mnist():
    set_init_6()
    project_variable.experiment_number = 162
    project_variable.device = 0
    main_file.run(project_variable)


# --------------------------------------------------------
# transformation_groups = [1, 1], [2, 4], [3, 8], [4, 12]
# --------------------------------------------------------
def e163_conv3dttn_mnist():
    set_init_6()
    project_variable.experiment_number = 163
    project_variable.device = 1
    project_variable.transformation_groups = [1, 1]
    main_file.run(project_variable)


def e164_conv3dttn_mnist():
    set_init_6()
    project_variable.experiment_number = 164
    project_variable.device = 2
    project_variable.transformation_groups = [2, 4]
    main_file.run(project_variable)


def e165_conv3dttn_mnist():
    set_init_6()
    project_variable.experiment_number = 165
    project_variable.device = 2
    project_variable.transformation_groups = [3, 8]
    main_file.run(project_variable)

def e166_conv3dttn_mnist():
    set_init_6()
    project_variable.experiment_number = 166
    project_variable.device = 0
    project_variable.transformation_groups = [4, 12]
    main_file.run(project_variable)


# --------------------------------------------------------
#       k0_groups = [1, 1], [2, 4], [3, 8], [4, 12]
# --------------------------------------------------------
def e167_conv3dttn_mnist():
    set_init_6()
    project_variable.experiment_number = 167
    project_variable.device = 0
    project_variable.k0_groups = [1, 1]
    main_file.run(project_variable)


def e168_conv3dttn_mnist():
    set_init_6()
    project_variable.experiment_number = 168
    project_variable.device = 0
    project_variable.k0_groups = [2, 4]
    main_file.run(project_variable)


def e169_conv3dttn_mnist():
    set_init_6()
    project_variable.experiment_number = 169
    project_variable.device = 1
    project_variable.k0_groups = [3, 8]
    main_file.run(project_variable)


def e170_conv3dttn_mnist():
    set_init_6()
    project_variable.experiment_number = 170
    project_variable.device = 1
    project_variable.k0_groups = [4, 12]
    main_file.run(project_variable)

#####################################################################################################################
#                                   model_number=2, k0.shapes
#####################################################################################################################
def set_init_7():
    project_variable.model_number = 2
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.randomize_training_data = True
    project_variable.data_points = [100, 1000, 1000]
    project_variable.repeat_experiments = 10
    project_variable.sheet_number = 6
    project_variable.num_out_channels = [1, 1]
    project_variable.k0_groups = project_variable.num_out_channels
    project_variable.transformation_groups = project_variable.num_out_channels

def e171_conv3d_mnist():
    project_variable.experiment_number = 171
    project_variable.device = 1
    project_variable.k_shape = (3, 4, 4)
    main_file.run(project_variable)

def e172_conv3d_mnist():
    project_variable.experiment_number = 172
    project_variable.device = 2
    project_variable.k_shape = (5, 6, 6)
    main_file.run(project_variable)

def e173_conv3d_mnist():
    project_variable.experiment_number = 173
    project_variable.device = 2
    project_variable.k_shape = (4, 6, 6)
    main_file.run(project_variable)

#####################################################################################################################
#                                   k0_init experiments: 174 - 185
#####################################################################################################################
def set_init_8():
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.randomize_training_data = True
    project_variable.repeat_experiments = 10
    project_variable.sheet_number = 7

    project_variable.theta_init = None
    project_variable.srxy_smoothness = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'
# --------------------------------------------------------
#      model_number=2, data_points = [50, 1000, 1000]
# --------------------------------------------------------
def e174_conv3d_mnist():
    set_init_8()
    project_variable.experiment_number = 174
    project_variable.model_number = 2
    project_variable.k0_init = 'ones'
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 0
    main_file.run(project_variable)

def e175_conv3d_mnist():
    set_init_8()
    project_variable.experiment_number = 175
    project_variable.model_number = 2
    project_variable.k0_init = 'normal'
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 0
    main_file.run(project_variable)

def e176_conv3d_mnist():
    set_init_8()
    project_variable.experiment_number = 176
    project_variable.model_number = 2
    project_variable.load_model = [25, 1, 19]  # ex, mo, ep
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 0
    project_variable.k0_init = None
    main_file.run(project_variable)

# --------------------------------------------------------
#      model_number=2, data_points = [100, 1000, 1000]
# --------------------------------------------------------
def e177_conv3d_mnist():
    set_init_8()
    project_variable.experiment_number = 177
    project_variable.model_number = 2
    project_variable.k0_init = 'ones'
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 0
    main_file.run(project_variable)

def e178_conv3d_mnist():
    set_init_8()
    project_variable.experiment_number = 178
    project_variable.model_number = 2
    project_variable.k0_init = 'normal'
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 0
    main_file.run(project_variable)

def e179_conv3d_mnist():
    set_init_8()
    project_variable.experiment_number = 179
    project_variable.model_number = 2
    project_variable.load_model = [25, 1, 19]  # ex, mo, ep
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 0
    project_variable.k0_init = None
    main_file.run(project_variable)
# --------------------------------------------------------
#      model_number=3, data_points = [50, 1000, 1000]
# --------------------------------------------------------
def e180_conv3dttn_mnist():
    set_init_8()
    project_variable.experiment_number = 180
    project_variable.model_number = 3
    project_variable.k0_init = 'ones'
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e181_conv3dttn_mnist():
    set_init_8()
    project_variable.experiment_number = 181
    project_variable.model_number = 3
    project_variable.k0_init = 'uniform'
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e182_conv3dttn_mnist():
    set_init_8()
    project_variable.experiment_number = 182
    project_variable.model_number = 3
    project_variable.load_model = [25, 1, 19]  # ex, mo, ep
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    project_variable.k0_init = None
    main_file.run(project_variable)

# --------------------------------------------------------
#      model_number=2, data_points = [100, 1000, 1000]
# --------------------------------------------------------
def e183_conv3dttn_mnist():
    set_init_8()
    project_variable.experiment_number = 183
    project_variable.model_number = 3
    project_variable.k0_init = 'ones'
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e184_conv3dttn_mnist():
    set_init_8()
    project_variable.experiment_number = 184
    project_variable.model_number = 3
    project_variable.k0_init = 'uniform'
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e185_conv3dttn_mnist():
    set_init_8()
    project_variable.experiment_number = 185
    project_variable.model_number = 3
    project_variable.load_model = [25, 1, 19]  # ex, mo, ep
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 1
    project_variable.k0_init = None
    main_file.run(project_variable)

#####################################################################################################################
#                              transfer learning experiments: 186 - 246
#####################################################################################################################
def e186_conv2d_mnist():
    project_variable.experiment_number = 186
    project_variable.model_number = 1
    project_variable.device = 1
    project_variable.repeat_experiments = 10
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.optimizer = 'adam'
    project_variable.sheet_number = 666
    main_file.run(project_variable)

def set_init_9():
    set_init_8()
    project_variable.model_number = 2
    project_variable.k0_init = None
    project_variable.sheet_number = 8

# --------------------------------------------------------
# model_number=2; train data_points = 20,50,100; 186 run 0
# --------------------------------------------------------

def e187_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 187
    project_variable.load_model = [186, 1, 19, 0]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e188_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 188
    project_variable.load_model = [186, 1, 19, 0]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e189_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 189
    project_variable.load_model = [186, 1, 19, 0]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=2; train data_points = 20,50,100; 186 run 1
# --------------------------------------------------------

def e190_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 190
    project_variable.load_model = [186, 1, 19, 1]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e191_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 191
    project_variable.load_model = [186, 1, 19, 1]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e192_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 192
    project_variable.load_model = [186, 1, 19, 1]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=2; train data_points = 20,50,100; 186 run 2
# --------------------------------------------------------

def e193_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 193
    project_variable.load_model = [186, 1, 19, 2]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e194_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 194
    project_variable.load_model = [186, 1, 19, 2]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e195_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 195
    project_variable.load_model = [186, 1, 19, 2]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=2; train data_points = 20,50,100; 186 run 3
# --------------------------------------------------------

def e196_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 196
    project_variable.load_model = [186, 1, 19, 3]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e197_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 197
    project_variable.load_model = [186, 1, 19, 3]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e198_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 198
    project_variable.load_model = [186, 1, 19, 3]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=2; train data_points = 20,50,100; 186 run 4
# --------------------------------------------------------

def e199_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 199
    project_variable.load_model = [186, 1, 19, 4]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e200_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 200
    project_variable.load_model = [186, 1, 19, 4]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e201_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 201
    project_variable.load_model = [186, 1, 19, 4]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=2; train data_points = 20,50,100; 186 run 5
# --------------------------------------------------------

def e202_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 202
    project_variable.load_model = [186, 1, 19, 5]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e203_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 203
    project_variable.load_model = [186, 1, 19, 5]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e204_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 204
    project_variable.load_model = [186, 1, 19, 5]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=2; train data_points = 20,50,100; 186 run 6
# --------------------------------------------------------

def e205_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 205
    project_variable.load_model = [186, 1, 19, 6]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e206_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 206
    project_variable.load_model = [186, 1, 19, 6]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e207_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 207
    project_variable.load_model = [186, 1, 19, 6]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=2; train data_points = 20,50,100; 186 run 7
# --------------------------------------------------------

def e208_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 208
    project_variable.load_model = [186, 1, 19, 7]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e209_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 209
    project_variable.load_model = [186, 1, 19, 7]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e210_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 210
    project_variable.load_model = [186, 1, 19, 7]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=2; train data_points = 20,50,100; 186 run 8
# --------------------------------------------------------

def e211_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 211
    project_variable.load_model = [186, 1, 19, 8]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e212_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 212
    project_variable.load_model = [186, 1, 19, 8]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e213_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 213
    project_variable.load_model = [186, 1, 19, 8]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=2; train data_points = 20,50,100; 186 run 9
# --------------------------------------------------------

def e214_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 214
    project_variable.load_model = [186, 1, 19, 9]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e215_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 215
    project_variable.load_model = [186, 1, 19, 9]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e216_conv3d_mnist():
    set_init_9()
    project_variable.experiment_number = 216
    project_variable.load_model = [186, 1, 19, 9]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)


def set_init_10():
    set_init_9()
    project_variable.model_number = 3

# --------------------------------------------------------
# model_number=3; train data_points = 20,50,100; 186 run 0
# --------------------------------------------------------

def e217_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 217
    project_variable.load_model = [186, 1, 19, 0]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e218_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 218
    project_variable.load_model = [186, 1, 19, 0]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e219_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 219
    project_variable.load_model = [186, 1, 19, 0]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=3; train data_points = 20,50,100; 186 run 1
# --------------------------------------------------------

def e220_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 220
    project_variable.load_model = [186, 1, 19, 1]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e221_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 221
    project_variable.load_model = [186, 1, 19, 1]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e222_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 222
    project_variable.load_model = [186, 1, 19, 1]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=3; train data_points = 20,50,100; 186 run 2
# --------------------------------------------------------

def e223_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 223
    project_variable.load_model = [186, 1, 19, 2]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e224_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 224
    project_variable.load_model = [186, 1, 19, 2]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e225_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 225
    project_variable.load_model = [186, 1, 19, 2]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=3; train data_points = 20,50,100; 186 run 3
# --------------------------------------------------------

def e226_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 226
    project_variable.load_model = [186, 1, 19, 3]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e227_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 227
    project_variable.load_model = [186, 1, 19, 3]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e228_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 228
    project_variable.load_model = [186, 1, 19, 3]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=3; train data_points = 20,50,100; 186 run 4
# --------------------------------------------------------

def e229_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 229
    project_variable.load_model = [186, 1, 19, 4]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e230_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 230
    project_variable.load_model = [186, 1, 19, 4]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e231_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 231
    project_variable.load_model = [186, 1, 19, 4]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=3; train data_points = 20,50,100; 186 run 5
# --------------------------------------------------------

def e232_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 232
    project_variable.load_model = [186, 1, 19, 5]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e233_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 233
    project_variable.load_model = [186, 1, 19, 5]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e234_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 234
    project_variable.load_model = [186, 1, 19, 5]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=3; train data_points = 20,50,100; 186 run 6
# --------------------------------------------------------

def e235_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 235
    project_variable.load_model = [186, 1, 19, 6]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e236_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 236
    project_variable.load_model = [186, 1, 19, 6]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e237_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 237
    project_variable.load_model = [186, 1, 19, 6]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=3; train data_points = 20,50,100; 186 run 7
# --------------------------------------------------------

def e238_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 238
    project_variable.load_model = [186, 1, 19, 7]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e239_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 239
    project_variable.load_model = [186, 1, 19, 7]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)

def e240_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 240
    project_variable.load_model = [186, 1, 19, 7]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 1
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=3; train data_points = 20,50,100; 186 run 8
# --------------------------------------------------------

def e241_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 241
    project_variable.load_model = [186, 1, 19, 8]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e242_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 242
    project_variable.load_model = [186, 1, 19, 8]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e243_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 243
    project_variable.load_model = [186, 1, 19, 8]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)
# --------------------------------------------------------
# model_number=3; train data_points = 20,50,100; 186 run 9
# --------------------------------------------------------

def e244_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 244
    project_variable.load_model = [186, 1, 19, 9]  # ex, mo, ep, run
    project_variable.data_points = [20, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e245_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 245
    project_variable.load_model = [186, 1, 19, 9]  # ex, mo, ep, run
    project_variable.data_points = [50, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

def e246_conv3dttn_mnist():
    set_init_10()
    project_variable.experiment_number = 246
    project_variable.load_model = [186, 1, 19, 9]  # ex, mo, ep, run
    project_variable.data_points = [100, 1000, 1000]
    project_variable.device = 2
    main_file.run(project_variable)

#####################################################################################################################
#                                   OUT_CHANNELS M2 EXPERIMENT START: 247 - 267
#####################################################################################################################
# --------------------------------------------------------
#                   out_channels = [9, 19]
# --------------------------------------------------------

def e247_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 247
    project_variable.device = 2
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e248_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 248
    project_variable.device = 2
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e249_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 249
    project_variable.device = 2
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e250_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 250
    project_variable.device = 2
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e251_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 251
    project_variable.device = 2
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e252_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 252
    project_variable.device = 2
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e253_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 253
    project_variable.device = 2
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [10, 20]
# --------------------------------------------------------

def e254_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 254
    project_variable.device = 2
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e255_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 255
    project_variable.device = 2
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e256_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 256
    project_variable.device = 2
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e257_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 257
    project_variable.device = 2
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e258_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 258
    project_variable.device = 1
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e259_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 259
    project_variable.device = 1
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e260_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 260
    project_variable.device = 1
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [15, 30]
# --------------------------------------------------------

def e261_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 261
    project_variable.device = 1
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e262_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 262
    project_variable.device = 1
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e263_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 263
    project_variable.device = 1
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e264_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 264
    project_variable.device = 1
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e265_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 265
    project_variable.device = 1
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e266_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 266
    project_variable.device = 1
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e267_conv3d_mnist():
    set_init_2()
    project_variable.experiment_number = 267
    project_variable.device = 1
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)
#####################################################################################################################
#                                   OUT_CHANNELS M3 EXPERIMENT START: 268 - 288
#####################################################################################################################
# --------------------------------------------------------
#                   out_channels = [9, 19]
# --------------------------------------------------------

def e268_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 268
    project_variable.device = 0
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e269_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 269
    project_variable.device = 0
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e270_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 270
    project_variable.device = 0
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e271_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 271
    project_variable.device = 0
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e272_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 272
    project_variable.device = 0
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e273_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 273
    project_variable.device = 0
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e274_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 274
    project_variable.device = 0
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [10, 20]
# --------------------------------------------------------

def e275_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 275
    project_variable.device = 1
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e276_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 276
    project_variable.device = 1
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e277_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 277
    project_variable.device = 1
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e278_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 278
    project_variable.device = 1
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e279_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 279
    project_variable.device = 1
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e280_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 280
    project_variable.device = 1
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e281_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 281
    project_variable.device = 1
    project_variable.num_out_channels = [10, 20]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [15, 30]
# --------------------------------------------------------

def e282_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 282
    project_variable.device = 2
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e283_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 283
    project_variable.device = 2
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e284_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 284
    project_variable.device = 2
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e285_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 285
    project_variable.device = 2
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e286_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 286
    project_variable.device = 2
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e287_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 287
    project_variable.device = 2
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e288_conv3dttn_mnist():
    set_init_5()
    project_variable.experiment_number = 288
    project_variable.device = 2
    project_variable.num_out_channels = [15, 30]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

#####################################################################################################################
#                                   OUT_CHANNELS M2 EXPERIMENTS: 289 - 336
#####################################################################################################################

def set_init_11():
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.randomize_training_data = True
    project_variable.repeat_experiments = 20
    project_variable.balance_training_data = True
    project_variable.model_number = 2
    project_variable.sheet_number = 9
    project_variable.experiment_state = 'extra'

# --------------------------------------------------------
#                   out_channels = [6, 16]
# --------------------------------------------------------
def e289_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 289
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [10, 200, 200]
    project_variable.batch_size = 10
    main_file.run(project_variable)

def e290_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 290
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e291_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 291
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e292_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 292
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e293_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 293
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e294_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 294
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e295_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 295
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e296_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 296
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)
# --------------------------------------------------------
#                   out_channels = [9, 19]
# --------------------------------------------------------
def e297_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 297
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [10, 200, 200]
    project_variable.batch_size = 10
    main_file.run(project_variable)

def e298_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 298
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e299_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 299
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e300_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 300
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e301_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 301
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e302_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 302
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e303_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 303
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e304_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 304
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)
# --------------------------------------------------------
#                   out_channels = [12, 22]
# --------------------------------------------------------
def e305_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 305
    project_variable.device = 2
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [10, 200, 200]
    project_variable.batch_size = 10
    main_file.run(project_variable)

def e306_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 306
    project_variable.device = 2
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e307_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 307
    project_variable.device = 2
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e308_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 308
    project_variable.device = 2
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e309_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 309
    project_variable.device = 2
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e310_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 310
    project_variable.device = 2
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e311_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 311
    project_variable.device = 2
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e312_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 312
    project_variable.device = 2
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)
# --------------------------------------------------------
#                   out_channels = [15, 25]
# --------------------------------------------------------
def e313_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 313
    project_variable.device = 0
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [10, 200, 200]
    project_variable.batch_size = 10
    main_file.run(project_variable)

def e314_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 314
    project_variable.device = 0
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e315_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 315
    project_variable.device = 0
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e316_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 316
    project_variable.device = 1
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e317_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 317
    project_variable.device = 1
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e318_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 318
    project_variable.device = 1
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e319_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 319
    project_variable.device = 2
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e320_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 320
    project_variable.device = 2
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [18, 28]
# --------------------------------------------------------
def e321_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 321
    project_variable.device = 2
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [10, 200, 200]
    project_variable.batch_size = 10
    main_file.run(project_variable)

def e322_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 322
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e323_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 323
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e324_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 324
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e325_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 325
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e326_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 326
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e327_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 327
    project_variable.device = 1
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e328_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 328
    project_variable.device = 1
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [21, 31]
# --------------------------------------------------------
def e329_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 329
    project_variable.device = 1
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [10, 200, 200]
    project_variable.batch_size = 10
    main_file.run(project_variable)

def e330_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 330
    project_variable.device = 1
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e331_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 331
    project_variable.device = 1
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e332_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 332
    project_variable.device = 0
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)
#
def e333_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 333
    project_variable.device = 2
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e334_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 334
    project_variable.device = 2
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e335_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 335
    project_variable.device = 2
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e336_conv3d_mnist():
    set_init_11()
    project_variable.experiment_number = 336
    project_variable.device = 2
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

#####################################################################################################################
#                                   OUT_CHANNELS M2 EXPERIMENT START: 337 - 384
#####################################################################################################################

def set_init_12():
    project_variable.batch_size = 20
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.randomize_training_data = True
    project_variable.repeat_experiments = 30
    project_variable.balance_training_data = True
    project_variable.model_number = 3
    project_variable.sheet_number = 9
    project_variable.experiment_state = 'new'
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.srxy_smoothness = None
    project_variable.weight_transform = 'seq'

# --------------------------------------------------------
#                   out_channels = [6, 16]
# --------------------------------------------------------
def e337_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 337
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [10, 200, 200]
    project_variable.batch_size = 10
    main_file.run(project_variable)

def e338_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 338
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e339_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 339
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e340_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 340
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e341_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 341
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e342_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 342
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e343_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 343
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e344_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 344
    project_variable.device = 0
    project_variable.num_out_channels = [6, 16]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)
# --------------------------------------------------------
#                   out_channels = [9, 19]
# --------------------------------------------------------
def e345_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 345
    project_variable.device = 0
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [10, 200, 200]
    project_variable.batch_size = 10
    main_file.run(project_variable)

def e346_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 346
    project_variable.device = 0
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e347_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 347
    project_variable.device = 0
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e348_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 348
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e349_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 349
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e350_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 350
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e351_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 351
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e352_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 352
    project_variable.device = 1
    project_variable.num_out_channels = [9, 19]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)
# --------------------------------------------------------
#                   out_channels = [12, 22]
# --------------------------------------------------------
def e353_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 353
    project_variable.device = 1
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [10, 200, 200]
    project_variable.batch_size = 10
    main_file.run(project_variable)

def e354_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 354
    project_variable.device = 1
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e355_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 355
    project_variable.device = 1
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e356_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 356
    project_variable.device = 1
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e357_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 357
    project_variable.device = 1
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e358_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 358
    project_variable.device = 1
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e359_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 359
    project_variable.device = 2
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e360_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 360
    project_variable.device = 2
    project_variable.num_out_channels = [12, 22]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)
# --------------------------------------------------------
#                   out_channels = [15, 25]
# --------------------------------------------------------
def e361_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 361
    project_variable.device = 2
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [10, 200, 200]
    project_variable.batch_size = 10
    main_file.run(project_variable)

def e362_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 362
    project_variable.device = 2
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e363_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 363
    project_variable.device = 2
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e364_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 364
    project_variable.device = 2
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e365_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 365
    project_variable.device = 2
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e366_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 366
    project_variable.device = 2
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e367_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 367
    project_variable.device = 2
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e368_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 368
    project_variable.device = 2
    project_variable.num_out_channels = [15, 25]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [18, 28]
# --------------------------------------------------------

def e369_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 369
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [10, 200, 200]
    project_variable.batch_size = 10
    main_file.run(project_variable)

def e370_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 370
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e371_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 371
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e372_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 372
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e373_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 373
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e374_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 374
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e375_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 375
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e376_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 376
    project_variable.device = 0
    project_variable.num_out_channels = [18, 28]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

# --------------------------------------------------------
#                   out_channels = [21, 31]
# --------------------------------------------------------
def e377_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 377
    project_variable.device = 0
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [10, 200, 200]
    project_variable.batch_size = 10
    main_file.run(project_variable)

def e378_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 378
    project_variable.device = 1
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e379_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 379
    project_variable.device = 1
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e380_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 380
    project_variable.device = 1
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)
#
def e381_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 381
    project_variable.device = 1
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e382_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 382
    project_variable.device = 1
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e383_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 383
    project_variable.device = 1
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e384_conv3dttn_mnist():
    set_init_12()
    project_variable.experiment_number = 384
    project_variable.device = 1
    project_variable.num_out_channels = [21, 31]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)


#####################################################################################################################
#                                   basics LeNet5 3DTTN revisit
#####################################################################################################################
def set_init_13():
    project_variable.model_number = 3
    project_variable.batch_size = 20
    project_variable.dataset = 'mov_mnist'
    project_variable.optimizer = 'sgd'
    project_variable.eval_on = 'test'
    project_variable.weight_transform = 'seq'

    project_variable.data_points = [100, 200, 200]
    project_variable.k0_init = 'normal'
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True
    project_variable.repeat_experiments = 20
    project_variable.end_epoch = 100
    project_variable.label_size = 10
    project_variable.sheet_number = 0

# --------------------------------------------------------
#                   finding good LR
# --------------------------------------------------------
def e385_conv3dttn_mnist():
    set_init_13()
    project_variable.experiment_number = 385
    project_variable.device = 1
    project_variable.learning_rate = 1e-9
    project_variable.theta_init = 'eye'
    main_file.run(project_variable)

def e386_conv3dttn_mnist():
    set_init_13()
    project_variable.experiment_number = 386
    project_variable.device = 1
    project_variable.learning_rate = 1e-8
    project_variable.theta_init = 'eye'
    main_file.run(project_variable)

def e387_conv3dttn_mnist():
    set_init_13()
    project_variable.experiment_number = 387
    project_variable.device = 1
    project_variable.learning_rate = 1e-7
    project_variable.theta_init = 'eye'
    main_file.run(project_variable)

def e388_conv3dttn_mnist():
    set_init_13()
    project_variable.experiment_number = 388
    project_variable.device = 1
    project_variable.learning_rate = 1e-6
    project_variable.theta_init = 'eye'
    main_file.run(project_variable)
#####
#####
def e389_conv3dttn_mnist():
    set_init_13()
    project_variable.experiment_number = 389
    project_variable.device = 1
    project_variable.learning_rate = 5e-8
    project_variable.theta_init = 'eye'
    main_file.run(project_variable)

def e390_conv3dttn_mnist():
    set_init_13()
    project_variable.experiment_number = 390
    project_variable.device = 1
    project_variable.learning_rate = 5e-7
    project_variable.theta_init = 'eye'
    main_file.run(project_variable)
# --------------------------------------------------------
#                   finding good LR
# --------------------------------------------------------
def e391_conv3dttn_mnist():
    set_init_13()
    project_variable.experiment_number = 391
    project_variable.device = 2
    project_variable.learning_rate = 1e-9
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    main_file.run(project_variable)

def e392_conv3dttn_mnist():
    set_init_13()
    project_variable.experiment_number = 392
    project_variable.device = 2
    project_variable.learning_rate = 1e-8
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    main_file.run(project_variable)

def e393_conv3dttn_mnist():
    set_init_13()
    project_variable.experiment_number = 393
    project_variable.device = 2
    project_variable.learning_rate = 1e-7
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    main_file.run(project_variable)

def e394_conv3dttn_mnist():
    set_init_13()
    project_variable.experiment_number = 394
    project_variable.device = 2
    project_variable.learning_rate = 1e-6
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    main_file.run(project_variable)
#####
#####
def e395_conv3dttn_mnist():
    set_init_13()
    project_variable.experiment_number = 395
    project_variable.device = 2
    project_variable.learning_rate = 5e-8
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    main_file.run(project_variable)

def e396_conv3dttn_mnist():
    set_init_13()
    project_variable.experiment_number = 396
    project_variable.device = 2
    project_variable.learning_rate = 5e-7
    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    main_file.run(project_variable)

project_variable = ProjectVariable(debug_mode=False)

