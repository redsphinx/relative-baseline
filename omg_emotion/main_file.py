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
import math

# temporary for debugging
# from .settings import ProjectVariable


def run(project_variable):
    # only one of these can be True at a time
    # assert(project_variable.use_adaptive_lr and project_variable.use_clr is False)

    START_LR = project_variable.learning_rate
    if project_variable.theta_learning_rate is not None:
        START_LR_THETA = project_variable.theta_learning_rate

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

    if project_variable.inference_only_mode:
        if project_variable.eval_on == 'val':
            project_variable.val = True
        else:
            project_variable.val = False
    else:
        project_variable.val = True

    if project_variable.eval_on == 'test':
        project_variable.test = True
    else:
        project_variable.test = False

    if project_variable.inference_only_mode:
        project_variable.train = False
    else:
        if not project_variable.randomize_training_data:
            project_variable.train = True
        else:
            project_variable.train = False


    # HERE: create the dali iterators
    if project_variable.use_dali:
        train_file_list = os.path.join(PP.jester_location, 'filelist_train.txt')
        # train_file_list = os.path.join(PP.jester_location, 'filelist_val_TEST.txt')

        val_file_list = os.path.join(PP.jester_location, 'filelist_val.txt')
        # val_file_list = os.path.join(PP.jester_location, 'filelist_val_TEST.txt')

        test_file_list = os.path.join(PP.jester_location, 'filelist_test.txt')

        if project_variable.val:
            print('Loading validation iterator...')
            val_iter = D.create_dali_iterator(10 * 27, val_file_list, 4, False, 0,
                                              project_variable.dali_iterator_size[1], False)
        if project_variable.test:
            print('Loading test iterator...')
            test_iter = D.create_dali_iterator(10 * 27, test_file_list, 4, False, 0,
                                               project_variable.dali_iterator_size[2], False)
        if not project_variable.inference_only_mode:
            print('Loading training iterator...')
            train_iter = D.create_dali_iterator(project_variable.batch_size, train_file_list,
                                                project_variable.dali_workers,
                                                project_variable.randomize_training_data, 6,
                                                project_variable.dali_iterator_size[0], True)

    else:
        data = D.load_data(project_variable, seed=None)

        if project_variable.train:
            data_train = D.get_data('train', data)
            labels_train = D.get_labels('train', data)
        if project_variable.val:
            data_val = D.get_data('val', data)
            labels_val = D.get_labels('val', data)
        if project_variable.test:
            data_test = D.get_data('test', data)
            labels_test = D.get_labels('test', data)

        # to ensure the same data will be chosen between various models
        # useful when experimenting with low number of datapoints
        if project_variable.same_training_data:
            np.random.seed(project_variable.data_points)
            # each run has a unique seed based on the initial datapoints configuration
            training_seed_runs = np.random.randint(10000, size=project_variable.repeat_experiments)

    # keep track of how many runs have collapsed and at which epoch it stops training
    runs_collapsed = np.zeros(shape=project_variable.repeat_experiments, dtype=int)
    which_epoch_stopped = np.ones(shape=project_variable.repeat_experiments, dtype=int) * -1

    # ====================================================================================================
    # start with runs
    # ====================================================================================================
    for num_runs in range(start, project_variable.repeat_experiments):
        if not project_variable.use_dali:
            if project_variable.same_training_data:
                seed = training_seed_runs[num_runs]
            else:
                seed = None

        # HERE data loading for 'train'
        if not project_variable.use_dali:
            # load the training data (which is now randomized)
            if not project_variable.inference_only_mode:
                if project_variable.randomize_training_data:
                    project_variable.test = False
                    project_variable.val = False
                    project_variable.train = True
                    data = D.load_data(project_variable, seed)
                    if project_variable.train:
                        data_train = D.get_data('train', data)
                        labels_train = D.get_labels('train', data)
                    
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

        if not project_variable.nas:
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

        if project_variable.inference_only_mode:
            pass
        else:
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

        # # load the weights for weighted loss
        # w = None
        # if project_variable.model_number == 0:
        #     w = np.array([1955] * 7) / np.array([262, 96, 54, 503, 682, 339, 19])
        # elif project_variable.dataset == 'jester' and project_variable.use_dali:
        #     w = np.array([0.0379007, 0.03862456, 0.0370375, 0.03737979, 0.03620443,
        #                   0.03648918, 0.03675273, 0.03750421, 0.03627937, 0.03738865,
        #                   0.03676129, 0.03696806, 0.03817587, 0.0391227, 0.04904935,
        #                   0.04642222, 0.0371072, 0.03684716, 0.0366673, 0.03648918,
        #                   0.03607197, 0.03593228, 0.0370462, 0.03637139, 0.03608847,
        #                   0.036873, 0.01644524])
        # if w is not None:
        #     w = w.astype(dtype=np.float32)
        #     w = torch.from_numpy(w).cuda(device)
        #     project_variable.loss_weights = w

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

        # keeping track of collapsed training confusion matrix
        # if there is collapse for 3 times in a row, we stop the experiment
        collapse_limit = 3
        collapse_tracker = 0

        # keeping track of validation accuracy for early stopping
        check_every_num_epoch = 10
        checking_at_epochs = [9 + (i * check_every_num_epoch) for i in range(project_variable.end_epoch // check_every_num_epoch)]
        checking_at_epochs = [0] + checking_at_epochs
        val_acc_tracker = 0
        val_loss_tracker = math.inf

        # variable depending on settings 'stop_at_collapse=True' and/or 'early_stopping=True'
        stop_experiment = False

        for e in range(project_variable.start_epoch+1, project_variable.end_epoch):
            if stop_experiment:
                break

            else:
                if project_variable.inference_only_mode:
                    pass
                else:
                    print('--------------------------------------------------------------------------\n'
                          'STARTING LEARNING RATE: %s\n'
                          '--------------------------------------------------------------------------'
                          % str(project_variable.learning_rate))

                    if project_variable.theta_learning_rate is not None:
                        print('--------------------------------------------------------------------------\n'
                              'STARTING THETA LEARNING RATE: %s\n'
                              '--------------------------------------------------------------------------'
                              % str(project_variable.theta_learning_rate))

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
                if project_variable.inference_only_mode:
                    project_variable.train = False
                else:
                    project_variable.train = True
                project_variable.val = False
                project_variable.test = False

                if project_variable.train:
                    w = None
                    if project_variable.model_number == 0:
                        w = np.array([1955] * 7) / np.array([262, 96, 54, 503, 682, 339, 19])
                    elif project_variable.dataset == 'jester' and project_variable.use_dali:
                        w = np.array([0.0379007, 0.03862456, 0.0370375, 0.03737979, 0.03620443,
                                      0.03648918, 0.03675273, 0.03750421, 0.03627937, 0.03738865,
                                      0.03676129, 0.03696806, 0.03817587, 0.0391227, 0.04904935,
                                      0.04642222, 0.0371072, 0.03684716, 0.0366673, 0.03648918,
                                      0.03607197, 0.03593228, 0.0370462, 0.03637139, 0.03608847,
                                      0.036873, 0.01644524])
                    if w is not None:
                        w = w.astype(dtype=np.float32)
                        w = torch.from_numpy(w).cuda(device)
                        project_variable.loss_weights = w

                    # if project_variable.model_number == 0:
                    #     w = np.array([1955] * 7) / np.array([262, 96, 54, 503, 682, 339, 19])
                    #     w = w.astype(dtype=np.float32)
                    #     w = torch.from_numpy(w).cuda(device)
                    #     project_variable.loss_weights = w

                    # data = D.load_data(project_variable)
                    # data_train = data[1][0]
                    # labels_train = data[2][0]

                    # labels is list because can be more than one type of labels
                    # HERE data
                    if project_variable.use_dali:
                        data = train_iter
                        my_model.train()
                    else:
                        data = data_train, labels_train
                        my_model.train()

                    if project_variable.nas or project_variable.stop_at_collapse:
                        train_accuracy, (has_collapsed, collapsed_matrix) = training.run(project_variable, data, my_model, my_optimizer, device)
                    else:
                        train_accuracy = training.run(project_variable, data, my_model, my_optimizer, device)
                # ------------------------------------------------------------------------------------------------
                # VALIDATION
                # ------------------------------------------------------------------------------------------------
                if project_variable.inference_only_mode:
                    if project_variable.eval_on == 'val':
                        project_variable.val = True
                    else:
                        project_variable.val = False
                else:
                    project_variable.val = True

                project_variable.train = False
                project_variable.test = False

                if project_variable.val:
                    w = None
                    if project_variable.model_number == 0:
                        w = np.array([481]*7) / np.array([51, 34, 17, 156, 141, 75, 7])
                    elif project_variable.dataset == 'jester' and project_variable.use_dali:
                        w = np.array([0.03913151, 0.03897372, 0.03606524, 0.03929058, 0.03464331, 0.0377558,
                                      0.03945095, 0.03913151, 0.03593117, 0.03593117, 0.03835509, 0.04044135,
                                      0.03731847, 0.0370325, 0.04931369, 0.04646867, 0.0354047, 0.03760889,
                                      0.03620031, 0.0377558, 0.03675089, 0.03593117, 0.03620031, 0.03451958,
                                      0.03464331, 0.03647352, 0.01327676])
                    if w is not None:
                        w = w.astype(dtype=np.float32)
                        w = torch.from_numpy(w).cuda(device)
                        project_variable.loss_weights = w


                    # HERE data
                    if project_variable.use_dali:
                        data = val_iter
                    else:
                        data = data_val, labels_val

                    if project_variable.early_stopping:
                        val_accuracy, val_loss = validation.run(project_variable, data, my_model, device)
                    else:
                        val_accuracy = validation.run(project_variable, data, my_model, device)
                # ------------------------------------------------------------------------------------------------
                # TESTING
                # ------------------------------------------------------------------------------------------------
                # only run at the last epoch
                if e == project_variable.end_epoch - 1 or project_variable.inference_only_mode:
                    project_variable.train = False
                    project_variable.val = False
                    if project_variable.eval_on == 'test':
                        project_variable.test = True
                    else:
                        project_variable.test = False

                    if project_variable.test:
                        w = None
                        if project_variable.model_number == 0:
                            w = np.array([ 1989] * 7) / np.array([329, 135, 50, 550, 678, 231, 16])
                        elif project_variable.dataset == 'jester' and project_variable.use_dali:
                            w = np.array([0.03897281, 0.04044657, 0.03819954, 0.03674154, 0.03716712, 0.0356529,
                                          0.03513242, 0.03578544, 0.03674154, 0.03804855, 0.03450281, 0.03437958,
                                          0.03674154, 0.0414926, 0.05093271, 0.05039939, 0.03804855, 0.03578544,
                                          0.03775013, 0.03513242, 0.03487784, 0.03605349, 0.03688231, 0.03760267,
                                          0.03760267, 0.03591897, 0.01300849])
                        if w is not None:
                            w = w.astype(dtype=np.float32)
                            w = torch.from_numpy(w).cuda(device)
                            project_variable.loss_weights = w

                        # HERE data
                        if project_variable.use_dali:
                            data = test_iter
                        else:
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

                                if project_variable.theta_learning_rate is not None:
                                    print('--------------------------------------------------------------------------\n'
                                          'THETA LEARNING RATE REDUCED: from %s to %s\n'
                                          '--------------------------------------------------------------------------'
                                          % (str(project_variable.theta_learning_rate),
                                             str(project_variable.theta_learning_rate / project_variable.reduction_factor)))

                                    project_variable.theta_learning_rate /= project_variable.reduction_factor

                # decide if the experiment should stop
                if project_variable.stop_at_collapse and project_variable.early_stopping:

                    if has_collapsed:
                        collapse_tracker = collapse_tracker + 1
                        if collapse_tracker >= collapse_limit:
                            stop_experiment = True
                            runs_collapsed[num_runs] = 1
                            which_epoch_stopped[num_runs] = e

                    elif project_variable.current_epoch in checking_at_epochs:
                        if val_accuracy < val_acc_tracker and val_loss > val_loss_tracker:
                            stop_experiment = True
                            which_epoch_stopped[num_runs] = e
                        val_acc_tracker = val_accuracy
                        val_loss_tracker = val_loss

                elif project_variable.stop_at_collapse or project_variable.nas:
                    if has_collapsed:
                        collapse_tracker = collapse_tracker + 1
                        if collapse_tracker >= collapse_limit:
                            stop_experiment = True
                            runs_collapsed[num_runs] = 1
                            which_epoch_stopped[num_runs] = e

                elif project_variable.early_stopping:
                    if project_variable.current_epoch in checking_at_epochs:
                        if val_accuracy < val_acc_tracker and val_loss > val_loss_tracker:
                            stop_experiment = True
                            which_epoch_stopped[num_runs] = e
                        val_acc_tracker = val_accuracy
                        val_loss_tracker = val_loss

        # at the end of a run
        project_variable.at_which_run += 1
        project_variable.writer.close()
        project_variable.learning_rate = START_LR  # reset the learning rate
        if project_variable.theta_learning_rate is not None:
            project_variable.theta_learning_rate = START_LR_THETA # reset theta learning rate

    if not project_variable.debug_mode:
        # acc, std, best_run = U.experiment_runs_statistics(project_variable.experiment_number, project_variable.model_number)
        acc, std, best_run = U.experiment_runs_statistics(project_variable.experiment_number,
                                                          project_variable.model_number, mode=project_variable.eval_on)

        S.write_results(acc, std, best_run, ROW, project_variable.sheet_number)

        if project_variable.stop_at_collapse or project_variable.early_stopping:
            best_run_stopped = which_epoch_stopped[best_run]
            num_runs_collapsed = sum(runs_collapsed)
            S.extra_write_results(int(best_run_stopped), int(num_runs_collapsed), ROW, project_variable.sheet_number)

        if project_variable.save_only_best_run:
            U.delete_runs(project_variable, best_run)

    # if project_variable.stop_at_collapse and project_variable.early_stopping:
    #     return train_accuracy, val_accuracy, has_collapsed, collapsed_matrix, val_loss
    # elif project_variable.stop_at_collapse or project_variable.nas:
    #     return train_accuracy, val_accuracy, has_collapsed, collapsed_matrix
    # elif project_variable.early_stopping:
    #     return val_accuracy, val_loss


#   ------------------------------------------------------/--------------------------------------------/---------------
#  ------------------------------------------------------/--------------------------------------------/---------------
# ------------------------------------------------------/--------------------------------------------/---------------

def run_test_batch(project_variable):
    experiment_number_start = project_variable.experiment_number
    all_accuracies = []

    if not project_variable.debug_mode:
        if project_variable.experiment_state == 'new':
            project_variable.experiment_number = [experiment_number_start,
                                                  experiment_number_start+project_variable.inference_in_batches[1]-1]
            project_variable.load_model = [project_variable.inference_in_batches[2], project_variable.inference_in_batches[3]]
            ROW = S.write_settings(project_variable)

    project_variable.test = True
    project_variable.val = False
    project_variable.train = False

    project_variable.current_epoch = 0

    data = D.load_data(project_variable, seed=None)
    data_test= D.get_data('test', data)
    labels_test= D.get_labels('test', data)
    device = setup.get_device(project_variable)


    for _i in range(project_variable.inference_in_batches[1]):
        project_variable.experiment_number = experiment_number_start + _i

        # remove duplicate log files
        log_file = 'experiment_%d_model_%d_run_%d.txt' % (project_variable.experiment_number,
                                                          project_variable.model_number,
                                                          project_variable.at_which_run)
        log_path = os.path.join(PP.saving_data, 'test', log_file)
        if os.path.exists(log_path):
            if os.path.isdir(log_path):
                shutil.rmtree(log_path)
            else:
                os.remove(log_path)

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

        # --------------------------------------------------------------------------------------
        load_this = [project_variable.inference_in_batches[2],
                     project_variable.inference_in_batches[3],
                     99, _i]
        print('loading model %s' % str(load_this))
        project_variable.load_model = load_this
        my_model = setup.get_model(project_variable)
        my_model.cuda(device)

        the_data = data_test, labels_test
        accuracy = testing.run(project_variable, the_data, my_model, device)
        all_accuracies.append(accuracy)

        project_variable.writer.close()

    if not project_variable.debug_mode:
        acc = np.mean(all_accuracies)
        std = np.std(all_accuracies)
        best_run = all_accuracies.index(max(all_accuracies))
        S.write_results(acc, std, best_run, ROW, project_variable.sheet_number)
