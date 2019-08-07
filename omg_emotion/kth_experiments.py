from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


def pilot():
    project_variable.device = 0
    project_variable.model_number = 3
    project_variable.experiment_number = 666
    project_variable.batch_size = 10
    project_variable.end_epoch = 2
    project_variable.dataset = 'kth_actions'
    # project_variable.dataset = 'mov_mnist'
    project_variable.data_points = [12, 12, 12]
    # project_variable.data_points = [10, 10, 10]
    project_variable.repeat_experiments = 1
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.label_size = 6
    # project_variable.label_size = 10

    main_file.run(project_variable)


def pilot_2():
    project_variable.device = 0
    project_variable.model_number = 4
    project_variable.experiment_number = 666
    project_variable.batch_size = 6
    project_variable.end_epoch = 5
    project_variable.dataset = 'kth_actions'
    project_variable.data_points = [96, 12, 12]
    project_variable.repeat_experiments = 1
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.label_size = 6
    project_variable.optimizer = 'adam'

    main_file.run(project_variable)


project_variable = ProjectVariable(debug_mode=True)

pilot_2()

'''
Warning: NaN or Inf found in input tensor.
Traceback (most recent call last):
  File "/home/gabras/deployed/relative_baseline/omg_emotion/kth_experiments.py", line 27, in <module>
    pilot()
  File "/home/gabras/deployed/relative_baseline/omg_emotion/kth_experiments.py", line 22, in pilot
    main_file.run(project_variable)
  File "/home/gabras/deployed/relative_baseline/omg_emotion/main_file.py", line 184, in run
    training.run(project_variable, data, my_model, my_optimizer, device)
  File "/home/gabras/deployed/relative_baseline/omg_emotion/training.py", line 113, in run
    project_variable.writer.add_histogram('fc1/weight', my_model.fc1.weight, project_variable.current_epoch)
  File "/home/gabras/miniconda3/envs/pytorch_tf_2/lib/python3.7/site-packages/tensorboardX/writer.py", line 443, in add_histogram
    histogram(tag, values, bins, max_bins=max_bins), global_step, walltime)
  File "/home/gabras/miniconda3/envs/pytorch_tf_2/lib/python3.7/site-packages/tensorboardX/summary.py", line 138, in histogram
    hist = make_histogram(values.astype(float), bins, max_bins)
  File "/home/gabras/miniconda3/envs/pytorch_tf_2/lib/python3.7/site-packages/tensorboardX/summary.py", line 176, in make_histogram
    raise ValueError('The histogram is empty, please file a bug report.')
ValueError: The histogram is empty, please file a bug report.
'''