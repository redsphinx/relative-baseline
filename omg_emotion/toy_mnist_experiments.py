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
    project_variable.device = 0
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e110_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 110
    project_variable.device = 0
    project_variable.num_out_channels = [7, 17]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e111_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 111
    project_variable.device = 0
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
    project_variable.device = 1
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [20, 200, 200]
    main_file.run(project_variable)

def e118_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 118
    project_variable.device = 1
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [30, 200, 200]
    main_file.run(project_variable)

def e119_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 119
    project_variable.device = 0
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [40, 200, 200]
    main_file.run(project_variable)

def e120_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 120
    project_variable.device = 0
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [50, 200, 200]
    main_file.run(project_variable)

def e121_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 121
    project_variable.device = 0
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [100, 200, 200]
    main_file.run(project_variable)

def e122_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 122
    project_variable.device = 0
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [500, 200, 200]
    main_file.run(project_variable)

def e123_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 123
    project_variable.device = 0
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [1000, 200, 200]
    main_file.run(project_variable)

def e124_conv3dttn_mnist():
    set_init_3()
    project_variable.experiment_number = 124
    project_variable.device = 0
    project_variable.num_out_channels = [8, 18]
    project_variable.data_points = [5000, 200, 200]
    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=False)

e101_conv3dttn_mnist()
# e102_conv3dttn_mnist()
# e103_conv3dttn_mnist()
# e104_conv3dttn_mnist()
# e105_conv3dttn_mnist()
# e106_conv3dttn_mnist()
# e107_conv3dttn_mnist()
# e108_conv3dttn_mnist()
#
# e109_conv3dttn_mnist()
# e110_conv3dttn_mnist()
# e111_conv3dttn_mnist()
# e112_conv3dttn_mnist()
# e113_conv3dttn_mnist()
# e114_conv3dttn_mnist()
# e115_conv3dttn_mnist()
# e116_conv3dttn_mnist()
#
# e117_conv3dttn_mnist()
# e118_conv3dttn_mnist()
# e119_conv3dttn_mnist()
# e120_conv3dttn_mnist()
# e121_conv3dttn_mnist()
# e122_conv3dttn_mnist()
# e123_conv3dttn_mnist()
# e124_conv3dttn_mnist()


# TODO: train first_weight and srxy parameters in alternating cycles
# TODO: experiment with different k_0
