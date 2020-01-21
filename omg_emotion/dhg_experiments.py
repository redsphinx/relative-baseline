from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


def set_init_1():
    project_variable.model_number = 12
    project_variable.end_epoch = 100
    project_variable.dataset = 'dhg'

    project_variable.num_in_channels = 1
    project_variable.data_points = [140 * 14,  20 * 14, 40 * 14]
    project_variable.label_size = 14
    project_variable.batch_size = 2 * 14
    project_variable.load_num_frames = 50  # 50
    project_variable.label_type = 'categories'

    project_variable.repeat_experiments = 10
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True

    project_variable.experiment_state = 'new'
    project_variable.eval_on = 'val'


def e6_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 6
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-3
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e7_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 7
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-4
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)

def e8_3D_dhg():
    set_init_1()
    project_variable.experiment_number = 8
    project_variable.sheet_number = 21
    project_variable.device = 2

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 1e-5
    project_variable.use_adaptive_lr = False
    project_variable.num_out_channels = [6, 16]

    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=False)


# e6_3D_dhg()
# e7_3D_dhg()
e8_3D_dhg()
