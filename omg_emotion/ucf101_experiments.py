import numpy as np
from multiprocessing import Pool

from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file
import subprocess
import time


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def set_init_1():
    project_variable.end_epoch = 100
    project_variable.dataset = 'ucf101'

    # if you want all the data: train: 4200, val: 250, test: 250
    # "missing" files are the REAL test
    # total_dp = {'train': 118562, 'val': 7393, 'test': 7394}
    project_variable.num_in_channels = 3
    project_variable.data_points = [2 * 27,  1 * 27, 0 * 27]
    project_variable.label_size = 101
    project_variable.load_num_frames = 30
    project_variable.label_type = 'categories'

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


def e1_conv3DTTN_ucf101():
    set_init_1()
    project_variable.model_number = 20
    project_variable.experiment_number = 31
    project_variable.sheet_number = 23
    project_variable.device = 0
    project_variable.end_epoch = 100
    project_variable.batch_size = 1
    project_variable.batch_size_val_test = 1

    project_variable.load_model = True  # exp, model, epoch, run
    project_variable.load_from_fast = True

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    project_variable.nas = False

    project_variable.stop_at_collapse = True
    project_variable.early_stopping = True

    project_variable.optimizer = 'adam'
    project_variable.learning_rate = 0.00005
    project_variable.use_adaptive_lr = True
    project_variable.num_out_channels = [0]

    # go = False
    # while not go:
    #     gpu_available = get_gpu_memory_map()
    #     if gpu_available[project_variable.device] < 100:
    #         go = True
    #     else:
    #         print('waiting for gpu %d...' % project_variable.device)
    #         time.sleep(10)

    main_file.run(project_variable)


# project_variable = ProjectVariable(debug_mode=False)
project_variable = ProjectVariable(debug_mode=True)


e1_conv3DTTN_ucf101()

