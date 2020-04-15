import os
import numpy as np
from datetime import date, datetime

from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file
from relative_baseline.omg_emotion import project_paths as PP


def run_single_experiment(project_variable, lr, epochs, out_channels, device, model_number):
    project_variable.nas = True
    project_variable.device = device
    project_variable.end_epoch = epochs
    project_variable.learning_rate = lr
    project_variable.num_out_channels = out_channels
    project_variable.model_number = model_number

    project_variable.save_data = False
    project_variable.save_model = False
    project_variable.save_graphs = False

    project_variable.dataset = 'jester'

    # if you want all the data: train: 150, val: 10, test: 10
    # total_dp = {'train': 118562, 'val': 7393, 'test': 7394}
    project_variable.num_in_channels = 3
    project_variable.data_points = [2 * 27,  1 * 27, 0 * 27]
    project_variable.label_size = 27
    project_variable.batch_size = 2 * 27
    project_variable.load_num_frames = 30
    project_variable.label_type = 'categories'
    project_variable.use_adaptive_lr = True

    project_variable.repeat_experiments = 1
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True

    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'

    project_variable.experiment_state = 'new'
    project_variable.eval_on = 'val'

    project_variable.experiment_number = 1792792989823
    project_variable.sheet_number = 22

    project_variable.repeat_experiments = 1
    project_variable.data_points = [30 * 27, 5 * 27, 0 * 27]
    project_variable.optimizer = 'adam'

    return main_file.run(project_variable)


def auto_search(lr_size, epochs, repeat_run, model_number, conv_layer_channels, device, b=1, e=10, s=2):
    file_name = date.today().strftime('%d%m') + '_' + datetime.now().strftime('%H%M%S') + '.txt'
    save_path = os.path.join(PP.nas_location, file_name)

    header = 'lr,epochs,model_number,conv_layer_channels,train_acc,val_acc,have_collapsed,which_matr_ind\n'
    with open(save_path, 'a') as my_file:
        my_file.write(header)

    for cfg in range(len(conv_layer_channels)):
        out_channels = conv_layer_channels[cfg]

        for lr_incr in range(b, e, s):
            lr = lr_incr / lr_size

            for run in range(repeat_run):
                chnls = '['
                for i in range(len(out_channels)):
                    chnls = chnls + str(out_channels[i]) + ' '

                chnls = chnls[:-1] + ']'

                settings = '%f,%d,%d,%s,' % (lr, epochs, model_number, chnls)
                with open(save_path, 'a') as my_file:
                    my_file.write(settings)

                project_variable = ProjectVariable(debug_mode=True)
                train_acc, val_acc, has_collapsed, collapsed_matrix = \
                    run_single_experiment(project_variable, lr, epochs, out_channels, device, model_number)

                if has_collapsed:
                    ind = np.where(collapsed_matrix == 1)[0]
                else:
                    ind = 420

                results = '%f,%f,%s,%s\n' % (train_acc, val_acc, str(has_collapsed), str(ind))
                with open(save_path, 'a') as my_file:
                    my_file.write(results)


# conv_channels = [[16, 16, 32, 64], [16, 32, 64, 128]]
# auto_search(10000, 30, 3, 15, conv_channels, 1)

# conv_channels = [[32, 32, 64, 64], [32, 32, 64, 128]]
# auto_search(10000, 30, 3, 15, conv_channels, 2)

# conv_channels = [[32, 64, 128, 256]]
# auto_search(10000, 30, 3, 15, conv_channels, 0)

# conv_channels = [[16, 32, 64, 128, 256]]
# auto_search(10000, 30, 3, 16, conv_channels, 1)

# conv_channels = [[16, 32, 32, 64, 64]]
# auto_search(10000, 30, 3, 16, conv_channels, 2)
#
conv_channels = [[32, 32, 64, 64, 128]]
auto_search(10000, 30, 3, 16, conv_channels, 2)