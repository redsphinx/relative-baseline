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
    # project_variable.srxy_init = 'eye'
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'

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

    project_variable.device = 2
    project_variable.batch_size = 30
    project_variable.end_epoch = 50
    project_variable.dataset = 'mov_mnist'

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.001
    project_variable.theta_init = None

    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=True)
# e1_conv3d_mnist()
# e4_conv3dttn_mnist()
conv3dttn_mmnist_pilot()
# conv3dttnpilot()
