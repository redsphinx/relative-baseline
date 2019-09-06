from relative_baseline.omg_emotion import training
from relative_baseline.omg_emotion import validation
from relative_baseline.omg_emotion import testing
from relative_baseline.omg_emotion import setup
from relative_baseline.omg_emotion import data_loading as D
from relative_baseline.omg_emotion import project_paths as PP
from relative_baseline.omg_emotion import utils as U
# from relative_baseline.omg_emotion import visualization as V
from relative_baseline.omg_emotion import sheets as S

import os
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
import shutil

# temporary for debugging
# from .settings import ProjectVariable


def run(project_variable):
    if project_variable.inference_only_mode:
        # TODO: implement mode where the model is only run on the test split
        
        pass


    else:

        START_LR = project_variable.learning_rate

        # write initial settings to spreadsheet
        if not project_variable.debug_mode:
            if project_variable.experiment_state == 'new':
                ROW = S.write_settings(project_variable)
            elif project_variable.experiment_state == 'crashed':
                project_variable.at_which_run = U.experiment_exists(project_variable.experiment_number,
                                                                    project_variable.model_number)
                ROW = S.get_specific_row(project_variable.experiment_number, project_variable.sheet_number)
            elif project_variable.experiment_state == 'extra':
                project_variable.at_which_run = 1 + U.experiment_exists(project_variable.experiment_number,
                                                                    project_variable.model_number)
                project_variable.repeat_experiments += project_variable.at_which_run
                ROW = S.get_specific_row(project_variable.experiment_number, project_variable.sheet_number)

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

        start = project_variable.at_which_run

        # load all data once as long as it's <1001 datapoints each
        project_variable.val = True

        if project_variable.eval_on == 'test':
            if project_variable.data_points[2] > 1000:
                print('Warning: test datapoints > 1000. Setting test datapoints to 1000.\n'
                      'To run on the entire test split, run experiment with the setting "inference_only_mode=True"')
                project_variable.data_points[2] = 1000
            project_variable.test = True
        else:
            project_variable.test = False

        if not project_variable.randomize_training_data:
            project_variable.train = True
        else:
            project_variable.train = False

        data = D.load_data(project_variable, seed=None)

        if project_variable.val:
            data_val = data[1][0]
            labels_val = data[2][0]

        if project_variable.test:
            data_test = data[1][1]
            labels_test = data[2][1]

        if project_variable.train:
            data_train = data[1][2]
            labels_train = data[2][2]

        if project_variable.same_training_data:
            np.random.seed(project_variable.data_points)
            training_seed_runs = np.random.randint(10000, size=project_variable.repeat_experiments)

        # ====================================================================================================
        # start with runs
        # ====================================================================================================
        for num_runs in range(start, project_variable.repeat_experiments):
            if project_variable.same_training_data:
                seed = training_seed_runs[num_runs]
            else:
                seed = None
            # load the training data (which is now randomized)
            if project_variable.randomize_training_data:
                project_variable.test = False
                project_variable.val = False
                project_variable.train = True
                data = D.load_data(project_variable, seed)
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

            if not project_variable.debug_mode:
                if num_runs == 0:
                    S.write_parameters(U.count_parameters(my_model), ROW, project_variable.sheet_number)

            # add project settings to writer
            text = 'experiment number:      %d;' \
                   'model number:           %d;' \
                   'trainable parameters:   %d;' \
                   % (project_variable.experiment_number,
                      project_variable.model_number,
                      U.count_parameters(my_model)
                      )
            project_variable.writer.add_text('project settings', text)
            # ====================================================================================================
            # start with epochs
            # ====================================================================================================

            # setup performance tracking for adapting learning rate
            # index 0   which epoch to check
            # index 1   the performance metric
            if project_variable.use_adaptive_lr:
                reduction_epochs = project_variable.end_epoch // project_variable.decrease_after_num_epochs
                track_performance = np.zeros((reduction_epochs))
                check_epochs = []
                for r in range(reduction_epochs):
                    check_epochs.append(r * project_variable.decrease_after_num_epochs)
            else:
                check_epochs = None
                track_performance = None

            for e in range(project_variable.start_epoch+1, project_variable.end_epoch):
                print('--------------------------------------------------------------------------\n'
                      'STARTING LEARNING RATE: %s\n'
                      '--------------------------------------------------------------------------'
                      % str(project_variable.learning_rate))

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
                # TRAINING
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
                    train_accuracy = training.run(project_variable, data, my_model, my_optimizer, device)
                # ------------------------------------------------------------------------------------------------
                # VALIDATION
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
                    val_accuracy = validation.run(project_variable, data, my_model, device)
                # ------------------------------------------------------------------------------------------------
                # TESTING
                # ------------------------------------------------------------------------------------------------
                # only run at the last epoch
                if e == project_variable.end_epoch - 1:
                    project_variable.train = False
                    project_variable.val = False
                    if project_variable.eval_on == 'test':
                        project_variable.test = True
                    else:
                        project_variable.test = False

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

                # at the end of an epoch
                if project_variable.use_adaptive_lr:
                    if e in check_epochs:
                        idx = check_epochs.index(e)
                        # update epoch with performance information
                        if project_variable.adapt_eval_on == 'train':
                            track_performance[idx] = train_accuracy
                        elif project_variable.adapt_eval_on == 'val':
                            track_performance[idx] = val_accuracy
                        if idx > 0:
                            if track_performance[idx] > track_performance[idx - 1]:
                                pass
                            else:
                                print('--------------------------------------------------------------------------\n'
                                      'LEARNING RATE REDUCED: from %s to %s\n'
                                      '--------------------------------------------------------------------------'
                                      % (str(project_variable.learning_rate),
                                         str(project_variable.learning_rate / project_variable.reduction_factor)))

                                project_variable.learning_rate /= project_variable.reduction_factor


            # at the end of a run
            project_variable.at_which_run += 1
            project_variable.writer.close()
            project_variable.learning_rate = START_LR  # reset the learning rate

        if not project_variable.debug_mode:
            # acc, std, best_run = U.experiment_runs_statistics(project_variable.experiment_number, project_variable.model_number)
            acc, std, best_run = U.experiment_runs_statistics(project_variable.experiment_number,
                                                              project_variable.model_number, mode=project_variable.eval_on)
            S.write_results(acc, std, best_run, ROW, project_variable.sheet_number)
            if project_variable.save_only_best_run:
                U.delete_runs(project_variable, best_run)


