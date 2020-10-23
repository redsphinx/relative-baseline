import numpy as np
from multiprocessing import Pool
from datetime import datetime

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


def wait_for_gpu(wait, device_num=None, threshold=100):

    if wait:
        go = False
        while not go:
            gpu_available = get_gpu_memory_map()
            if gpu_available[device_num] < threshold:
                go = True
            else:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print('%s Waiting for gpu %d...' % (current_time, device_num))
                time.sleep(10)
    else:
        return


def set_init_1():
    project_variable.end_epoch = 200
    project_variable.dataset = 'kinetics400'

    # total_dp = {'train': 9537, 'val/test': 3783}
    project_variable.num_in_channels = 3
    project_variable.data_points = [2 * 27,  1 * 27, 0 * 27]
    project_variable.label_size = 400
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


def e001_3T_kinetics():
    set_init_1()
    project_variable.model_number = 23 # googlenet
    project_variable.experiment_number = 2000
    project_variable.sheet_number = 23
    project_variable.device = 1
    project_variable.end_epoch = 200
    project_variable.batch_size = 1
    project_variable.batch_size_val_test = 1

    project_variable.inference_only_mode = True

    project_variable.load_model = False
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

    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=True)


e001_3T_kinetics()

