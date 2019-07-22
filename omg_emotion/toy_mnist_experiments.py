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


def e27_conv3dttn_mnist():
    # 3dttn with  s r x y params
    project_variable.experiment_number = 27
    project_variable.model_number = 3

    project_variable.device = 0
    project_variable.batch_size = 30
    project_variable.end_epoch = 20
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None

    project_variable.k0_init = 'ones'

    main_file.run(project_variable)


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


project_variable = ProjectVariable(debug_mode=False)

# e27_conv3dttn_mnist()
# e28_conv3dttn_mnist()
e29_conv3dttn_mnist()


# TODO: initialize with gabor filters
# TODO: understand the theta -> grid -> output transformations for paper
# TODO: add alexnet
# TODO: train first_weight and srxy parameters in alternating cycles
# TODO: experiment with different k_0