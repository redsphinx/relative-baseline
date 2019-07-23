from relative_baseline.omg_emotion import training
from relative_baseline.omg_emotion import validation
from relative_baseline.omg_emotion import testing
from relative_baseline.omg_emotion import setup
from relative_baseline.omg_emotion import data_loading as D
from relative_baseline.omg_emotion import project_paths as PP
from relative_baseline.omg_emotion import utils as U
# from relative_baseline.omg_emotion import visualization as V

import os
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
import shutil

# temporary for debugging
# from .settings import ProjectVariable

def run(project_variable):
    # project_variable = ProjectVariable()

    # load all data once
    project_variable.val = True
    project_variable.test = True

    if not project_variable.randomize_training_data:
        project_variable.train = True
    else:
        project_variable.train = False

    data = D.load_data(project_variable)

    if project_variable.val:
        data_val = data[1][0]
        labels_val = data[2][0]

    if project_variable.test:
        data_test = data[1][1]
        labels_test = data[2][1]

    if project_variable.train:
        data_train = data[1][2]
        labels_train = data[2][2]

    # remove duplicate log files
    log_file = 'experiment_%d_model_%d_run_%d.txt' % (project_variable.experiment_number,
                                                      project_variable.model_number,
                                                      project_variable.at_which_run)
    _which = ['train', 'val', 'test']
    for w in _which:
        log_path = os.path.join(PP.saving_data, w, log_file)
        if os.path.exists(log_path):
            if os.path.isdir(log_path):
                shutil.rmtree(log_path)
            else:
                os.remove(log_path)

    for num_runs in range(project_variable.repeat_experiments):
        # load the training data (which is now randomized)
        if project_variable.randomize_training_data:
            project_variable.test = False
            project_variable.val = False
            project_variable.train = True
            data = D.load_data(project_variable)
            if project_variable.train:
                data_train = data[1][0]
                labels_train = data[2][0]

        print('-------------------------------------------------------\n\n'
              'RUN: %d / %d\n\n'
              '-------------------------------------------------------'
              % (num_runs, project_variable.repeat_experiments))


        # create writer for tensorboardX
        if not project_variable.debug_mode:
            path = os.path.join(PP.writer_path, 'experiment_%d_model_%d' % (project_variable.experiment_number,
                                                                            project_variable.model_number))
            subfolder = os.path.join(path, 'run_%d' % project_variable.at_which_run)

            path = subfolder

        else:
            path = os.path.join(PP.writer_path, 'debugging')

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            # clear directory before writing new events
            shutil.rmtree(path)
            time.sleep(2)
            os.mkdir(path)

        project_variable.writer = SummaryWriter(path)
        print('tensorboardX writer path: %s' % path)

        # setup model, optimizer & device
        my_model = setup.get_model(project_variable)
        device = setup.get_device(project_variable)

        if project_variable.device is not None:
            my_model.cuda(device)

        my_optimizer = setup.get_optimizer(project_variable, my_model)

        print('Loaded model number %d with %d trainable parameters' % (project_variable.model_number, U.count_parameters(my_model)))

        # add project settings to writer
        text = 'experiment number:      %d;' \
               'model number:           %d;' \
               'debug mode:             %s;' \
               'trainable parameters:   %d;' \
               'optimizer:              %s;' \
               'learning rate:          %f;' \
               'batch size:             %d;' \
               'epochs:                 %d;' \
               'theta init:             %s;' \
               'srxy init:              %s;' \
               'weight transform:       %s;' \
               'srxy smoothness:        %s;' \
               % (project_variable.experiment_number,
                  project_variable.model_number,
                  project_variable.debug_mode,
                  U.count_parameters(my_model),
                  project_variable.optimizer,
                  project_variable.learning_rate,
                  project_variable.batch_size,
                  project_variable.end_epoch,
                  project_variable.theta_init,
                  project_variable.srxy_init,
                  project_variable.weight_transform,
                  project_variable.srxy_smoothness)
        project_variable.writer.add_text('project settings', text)

        # start with epochs
        for e in range(project_variable.start_epoch+1, project_variable.end_epoch):
            project_variable.current_epoch = e

            # get data
            # splits = ['train', 'val', 'test']
            # final_data = [[img0, img1,...],
            #               [img0, img1,...],
            #               [img0, img1,...]]
            # final_labels = [[arousal, valence, categories],
            #                 [arousal, valence, categories],
            #                 [arousal, valence, categories]]

            # ------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------
            project_variable.train = True
            project_variable.val = False
            project_variable.test = False

            if project_variable.train:
                if project_variable.model_number == 0:
                    w = np.array([1955] * 7) / np.array([262, 96, 54, 503, 682, 339, 19])
                    w = w.astype(dtype=np.float32)
                    w = torch.from_numpy(w).cuda(device)
                    project_variable.loss_weights = w

                # data = D.load_data(project_variable)
                # data_train = data[1][0]
                # labels_train = data[2][0]

                # labels is list because can be more than one type of labels
                data = data_train, labels_train
                my_model.train()
                training.run(project_variable, data, my_model, my_optimizer, device)
            # ------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------
            project_variable.train = False
            project_variable.val = True
            project_variable.test = False

            if project_variable.val:
                if project_variable.model_number == 0:
                    w = np.array([481]*7) / np.array([51, 34, 17, 156, 141, 75, 7])
                    w = w.astype(dtype=np.float32)
                    w = torch.from_numpy(w).cuda(device)
                    project_variable.loss_weights = w

                data = data_val, labels_val
                validation.run(project_variable, data, my_model, device)
            # # ------------------------------------------------------------------------------------------------
            # # ------------------------------------------------------------------------------------------------
            project_variable.train = False
            project_variable.val = False
            project_variable.test = True

            if project_variable.test:
                if project_variable.model_number == 0:
                    w = np.array([ 1989] * 7) / np.array([329, 135, 50, 550, 678, 231, 16])
                    w = w.astype(dtype=np.float32)
                    w = torch.from_numpy(w).cuda(device)
                    project_variable.loss_weights = w

                data = data_test, labels_test
                testing.run(project_variable, data, my_model, device)
            # ------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------
        project_variable.at_which_run += 1

    if not project_variable.debug_mode:
        U.experiment_runs_statistics(project_variable.experiment_number, project_variable.model_number)


