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
#                                LONG EXPERIMENT MODEL 5 FINDING DECENT PARAMETERS
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
    project_variable.device = 0
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)


def e11_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 11
    project_variable.device = 0
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)


def e12_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 12
    project_variable.device = 0
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)


# --
def e13_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 13
    project_variable.device = 0
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [8, 16, 32, 64]
    main_file.run(project_variable)


def e14_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 14
    project_variable.device = 0
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [8, 16, 32, 64]
    main_file.run(project_variable)


def e15_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 15
    project_variable.device = 0
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [8, 16, 32, 64]
    main_file.run(project_variable)


# --
def e16_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 16
    project_variable.device = 0
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [16, 32, 64, 128]
    main_file.run(project_variable)


def e17_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 17
    project_variable.device = 0
    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [16, 32, 64, 128]
    main_file.run(project_variable)


def e18_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 18
    project_variable.device = 0
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
    project_variable.device = 0
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00001
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)


def e20_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 20
    project_variable.device = 0
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.00005
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)


def e21_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 21
    project_variable.device = 0
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)


# --
def e22_C3D_kth():
    set_init_1()
    project_variable.experiment_number = 22
    project_variable.device = 0
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
    project_variable.device = 0
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
    project_variable.device = 0
    project_variable.load_num_frames = 100
    project_variable.learning_rate = 0.0001
    project_variable.num_out_channels = [16, 32, 64, 128]
    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=False)

e1_C3D_kth()
e2_C3D_kth()
e3_C3D_kth()
e4_C3D_kth()
e5_C3D_kth()
e6_C3D_kth()
e7_C3D_kth()
e8_C3D_kth()
e9_C3D_kth()

e10_C3D_kth()
e11_C3D_kth()
e12_C3D_kth()
e13_C3D_kth()
e14_C3D_kth()
e15_C3D_kth()
e16_C3D_kth()
e17_C3D_kth()
e18_C3D_kth()

e19_C3D_kth()
e20_C3D_kth()
e21_C3D_kth()
e22_C3D_kth()
e23_C3D_kth()
e24_C3D_kth()
e25_C3D_kth()
e26_C3D_kth()
e27_C3D_kth()




'''
TODO

find good learning rate
is batch norm helpful
figure out the number of out channels

'''


