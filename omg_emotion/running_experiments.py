from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file
import numpy as np

project_variable = ProjectVariable(debug_mode=True)


def pilot():
    project_variable.device = 0
    project_variable.model_number = 0
    project_variable.experiment_number = 0

    project_variable.batch_size = 20

    # project_variable.save_model = False
    # project_variable.save_data = False

    main_file.run(project_variable)


def e1():
    import os
    from relative_baseline.omg_emotion import project_paths as PP
    from relative_baseline.omg_emotion import data_loading as D
    from tensorboardX import SummaryWriter


    project_variable.device = 0
    project_variable.model_number = 0
    project_variable.experiment_number = 1

    project_variable.batch_size = 16
    project_variable.end_epoch = 1

    project_variable.save_data = False

    path = os.path.join(PP.writer_path, 'experiment_%d_model_%d' % (project_variable.experiment_number,
                                                                    project_variable.model_number))
    if not os.path.exists(path):
        os.mkdir(path)
    project_variable.writer = SummaryWriter(path)

    project_variable.val = True
    project_variable.train = False
    project_variable.test = False

    data = D.load_data(project_variable)
    data_val = data[1][0]
    labels_val = data[2][0]

    for i in range(0, 100):
        if i == 1:
            print('ss')
        project_variable.current_epoch = i
        project_variable.load_model = [0, 0, i]  # experiment, model, epoch
        # TODO: make this nicer
        data_val = np.copy(data[1][0])
        labels_val = np.copy(data[2][0])
        main_file.run_many_val(project_variable, data_val, labels_val)


def e1_0():
    project_variable.device = 0
    project_variable.model_number = 0
    project_variable.experiment_number = 1

    project_variable.batch_size = 16
    project_variable.end_epoch = 1
    project_variable.save_data = False

    for i in range(0, 100):
        project_variable.current_epoch = i
        project_variable.load_model = [0, 0, i]  # experiment, model, epoch
        main_file.run(project_variable)

e1()
# e1_0()

# https://discuss.pytorch.org/t/loading-saved-models-gives-inconsistent-results-each-time/36312/3
