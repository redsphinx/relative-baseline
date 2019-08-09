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
#                                   OUT_CHANNELS M2 EXPERIMENT START: 337 - 384
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

    project_variable.load_num_frames = 50
    project_variable.learning_rate = 0.0001
    # project_variable.num_out_channels = [16, 32, 64, 128]
    project_variable.num_out_channels = [4, 8, 16, 32]
    main_file.run(project_variable)



project_variable = ProjectVariable(debug_mode=True)


e1_C3D_kth()






'''
TODO

find good learning rate
is batch norm helpful
figure out the number of out channels

'''


