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


project_variable = ProjectVariable(debug_mode=False)


# e38_conv3dttn_mnist()     # run 2 times
# e39_conv3dttn_mnist()     # run 2 times
# e40_conv3dttn_mnist()     # run 2 times
# e41_conv3dttn_mnist()     # run 2 times
# e42_conv3dttn_mnist()     # run 2 times
# e43_conv3dttn_mnist()     # run 2 times
# e44_conv3dttn_mnist()     # run 2 times
# e45_conv3dttn_mnist()     # run 2 times

# e46_conv3dttn_mnist()     # run 3 times
# e47_conv3dttn_mnist()     # run 3 times
# e48_conv3dttn_mnist()     # run 3 times
# e49_conv3dttn_mnist()     # run 3 times
# e50_conv3dttn_mnist()     # run 3 times
# e51_conv3dttn_mnist()     # run 3 times
# e52_conv3dttn_mnist()     # run 3 times
e53_conv3dttn_mnist()     # run 3 times




# TODO: train first_weight and srxy parameters in alternating cycles
# TODO: experiment with different k_0
