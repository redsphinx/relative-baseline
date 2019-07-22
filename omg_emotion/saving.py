import os
from relative_baseline.omg_emotion import project_paths as PP
import torch


def update_logs(project_variable, which, save_list):
    log_file = 'experiment_%d_model_%d_run_%d.txt' % (project_variable.experiment_number,
                                                      project_variable.model_number,
                                                      project_variable.at_which_run)

    log_path = os.path.join(PP.saving_data, which, log_file)

    line = ''
    for i in range(len(save_list)):
        line = line + str(save_list[i]) + ','

    line = line[0:-1] + '\n'

    with open(log_path, 'a') as my_file:
        my_file.write(line)


def save_model(project_variable, my_model):
    folder_model = 'experiment_%d_model_%d_run_%d' % (project_variable.experiment_number,
                                                      project_variable.model_number,
                                                      project_variable.at_which_run)
    folder_path = os.path.join(PP.saving_data, 'models', folder_model)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    name_model = 'epoch_%d' % project_variable.current_epoch
    save_path = os.path.join(folder_path, name_model)
    torch.save(my_model.state_dict(), save_path)

