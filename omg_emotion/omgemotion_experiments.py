from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


def e0_C3D_omgemo():
    project_variable.model_number = 11
    project_variable.end_epoch = 100
    project_variable.dataset = 'omg_emotion'
    project_variable.device = 2
    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 5e-5
    # project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [12, 22]

    project_variable.data_points = [50*7, 3*7, 5*7]
    project_variable.label_size = 7
    project_variable.batch_size = 14
    project_variable.load_num_frames = 60 # 60
    project_variable.label_type = 'categories'

    project_variable.repeat_experiments = 1
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True

    project_variable.experiment_state = 'new'
    project_variable.sheet_number = 100

    project_variable.eval_on = 'test'

    project_variable.experiment_number = 1
    main_file.run(project_variable)

# --------------------------------------------------------------------------------------------------------------------
#                                       Run LeNet-5-3DConv on omg_emotion
# --------------------------------------------------------------------------------------------------------------------

def set_init_1():
    project_variable.model_number = 12
    project_variable.end_epoch = 100
    project_variable.dataset = 'omg_emotion'
    project_variable.optimizer = 'adam'

    project_variable.data_points = [278 * 7, 68 * 7, 0]
    project_variable.label_size = 7
    project_variable.batch_size = 3 * 7
    project_variable.load_num_frames = 60  # 60
    project_variable.label_type = 'categories'

    project_variable.repeat_experiments = 10
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True

    project_variable.experiment_state = 'new'
    project_variable.eval_on = 'val'


def e1_3D_omgemo():
    set_init_1()
    project_variable.experiment_state = 'crashed'
    project_variable.experiment_number = 1
    project_variable.sheet_number = 20
    project_variable.device = 2

    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


def e2_3D_omgemo():
    set_init_1()
    project_variable.experiment_number = 2
    project_variable.sheet_number = 20
    project_variable.device = 2

    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

# --

def e3_3D_omgemo():
    set_init_1()
    project_variable.experiment_number = 3
    project_variable.sheet_number = 20
    project_variable.device = 0

    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


def e4_3D_omgemo():
    set_init_1()
    project_variable.experiment_number = 4
    project_variable.sheet_number = 20
    project_variable.device = 0

    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

# --
# --

def e5_3D_omgemo():
    set_init_1()
    project_variable.experiment_number = 5
    project_variable.sheet_number = 20
    project_variable.device = 1

    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [12, 22]

    main_file.run(project_variable)


def e6_3D_omgemo():
    set_init_1()
    project_variable.experiment_number = 6
    project_variable.sheet_number = 20
    project_variable.device = 0

    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [12, 22]

    main_file.run(project_variable)

# --

def e7_3D_omgemo():
    set_init_1()
    project_variable.experiment_number = 7
    project_variable.sheet_number = 20
    project_variable.device = 0

    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [12, 22]

    main_file.run(project_variable)


def e8_3D_omgemo():
    set_init_1()
    project_variable.experiment_number = 8
    project_variable.sheet_number = 20
    project_variable.device = 0

    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [12, 22]

    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=False)

# e1_3D_omgemo() # run in crashed mode
# e2_3D_omgemo()
# e3_3D_omgemo()
# e4_3D_omgemo()
e5_3D_omgemo()
# e6_3D_omgemo()
# e7_3D_omgemo()
# e8_3D_omgemo()
