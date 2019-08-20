from relative_baseline.omg_emotion import project_paths as PP

class ProjectVariable(object):
    def __init__(self, debug_mode=True):
        print("\nRUNNING ON '%s' SERVER\n" % PP.server)
        if debug_mode:
            print("running in debug mode")

        """
        Default values for all the experimental variables.
        """
        self._writer = None

        self._debug_mode = debug_mode
        
        # int, which gpu to use {None, 0, 1, etc}
        self._device = None

        # list of int, which model to load, [experiment, model, epoch]
        self._load_model = None
        # only saves the model from the best run
        self._save_only_best_run = True

        # which model to load. mapping in legend.txt
        self._model_number = None
        # int, experiment data for log
        self._experiment_number = None
        
        # which google sheet to write to
        self._sheet_number = None
        
        # bool
        self._pretrain_resnet18_weights = True
        
        # int, the current epoch
        self._current_epoch = None
    
        # UNUSED? ================================================================================
        # list of str, which datasets to train, val and test on
        # implemented sets: omg_emotion, affectnet
        self._dataset_train = ['omg_emotion']
        self._dataset_val = ['omg_emotion']
        self._dataset_test = ['omg_emotion']
        # UNUSED? ================================================================================
        
        # instead of having 3 dataset splits, have just 1 dataset parameter
        # implemented datasets: omg_emotion, mnist, dummy, mov_mnist, kth_actions.  status affectnet??
        self._dataset = 'mnist'
        self._randomize_training_data = False
        self._balance_training_data = False
        self._same_training_data = False
        self._data_points = [100, 100, 100]  # [train, val, test]
        
        # bool, which procedures to perform
        self._train = None
        self._val = None
        self._test = None
        
        # list of str, which labels to use.
        # omg_emotion: ['categories', 'arousal', 'valence']
        # affect_net: [categories, arousal, valence, face, landmarks]
        self._label_type = ['categories']

        # int, label size for the output type
        # omg_emotion categories: 7
        # affectnet categories: 11
        # mnist: 10
        self._label_size = 10

        # float, learning rate
        self._learning_rate = 0.001
        # str, loss function
        self._loss_function = 'cross_entropy'
        # list, weights for balanced loss, necessary for resnet18
        self._loss_weights = None
        # list of str, optimizer
        # supported: adam, sgd
        self._optimizer = 'sgd'
        # momentum
        self._momentum = 0.9
        
        # number of out_channels in the convolution layers of CNNs
        self._num_out_channels = [6, 16]

        # int, seed for shuffling
        self._seed = 6

        # depending on debug mode
        if debug_mode:
            self._batch_size = 30
            self._start_epoch = -1
            self._end_epoch = 5
            self._train_steps = 10
            self._val_steps = 1
            self._test_steps = 1
            self._save_data = False
            self._save_model = False
            self._save_graphs = False
        else:
            self._batch_size = 30
            self._start_epoch = -1
            self._end_epoch = 20
            self._train_steps = 50
            self._val_steps = 10
            self._test_steps = 10
            self._save_data = True
            self._save_model = True
            self._save_graphs = True

        self._repeat_experiments = 1
        self._at_which_run = 0
        # how to initialize experiment files and saves: 'new': new experiment, 'crashed': experiment crashed before 
        # finishing, 'extra': experiment finished, run an additional batch of the same experiment
        self._experiment_state = 'new'

        # ----------------------------------------------------------------------------------------------------------
        # settings only for 3dconvttn stuff
        # ----------------------------------------------------------------------------------------------------------
        # how to initialize theta: 'normal', 'eye', 'eye-like' or None. if None, theta is created from affine params
        self._theta_init = 'eye'
        # how to initialize SRXY: 'normal', 'eye'=[1,0,0,0], 'eye-like'=[1+e,e,e,e]
        self._srxy_init = 'normal'
        # how to transform weights in kernel: 'naive'=weights are a transform of first_weight, 'seq'=sequential
        self._weight_transform = 'naive'
        # which kind of smoothness constraint for the srxy values: None, 'sigmoid', 'sigmoid_small'
        self._srxy_smoothness = None
        # k_0 initialization: 'normal', 'ones', 'ones_var'=mean=1,std=0.5, 'uniform'
        self._k0_init = 'normal'
        # share transformation parameters across all filters in a layer.
        # here we set how many sets of transformations are learned.
        # note that 1 <= transformation_groups <= num_out_channels
        self._transformation_groups = self.num_out_channels
        # filters share k0
        self._k0_groups = self.num_out_channels
        # shape of convolution filter
        self._k_shape = (5, 5, 5)
        #
        # time dimension of the 3D max pooling
        self._max_pool_temporal = 2
        # height=width dimension of the convolutional kernels
        self._conv_k_hw = 3
        # ----------------------------------------------------------------------------------------------------------
        # setting for video datasets
        # ----------------------------------------------------------------------------------------------------------
        self._load_num_frames = 30
        # time dimension of the kernel in conv1
        self._conv1_k_t = 3
        # where to add batchnorm after each non-linear activation layer
        self._do_batchnorm = [False, False, False, False, False]
        # ----------------------------------------------------------------------------------------------------------

    @property
    def writer(self):
        return self._writer

    @writer.setter
    def writer(self, value):
        self._writer = value

    @property
    def debug_mode(self):
        return self._debug_mode

    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        self._device = value

    @property
    def load_model(self):
        return self._load_model

    @load_model.setter
    def load_model(self, value):
        self._load_model = value

    @property
    def save_only_best_run(self):
        return self._save_only_best_run

    @save_only_best_run.setter
    def save_only_best_run(self, value):
        self._save_only_best_run = value

    @property
    def model_number(self):
        return self._model_number

    @model_number.setter
    def model_number(self, value):
        self._model_number = value

    @property
    def experiment_number(self):
        return self._experiment_number

    @experiment_number.setter
    def experiment_number(self, value):
        self._experiment_number = value

    @property
    def sheet_number(self):
        return self._sheet_number

    @sheet_number.setter
    def sheet_number(self, value):
        self._sheet_number = value

    @property
    def pretrain_resnet18_weights(self):
        return self._pretrain_resnet18_weights

    @pretrain_resnet18_weights.setter
    def pretrain_resnet18_weights(self, value):
        self._pretrain_resnet18_weights = value
    
    @property
    def current_epoch(self):
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value):
        self._current_epoch = value

    @property
    def dataset_train(self):
        return self._dataset_train

    @dataset_train.setter
    def dataset_train(self, value):
        self._dataset_train = value

    @property
    def dataset_val(self):
        return self._dataset_val

    @dataset_val.setter
    def dataset_val(self, value):
        self._dataset_val = value

    @property
    def dataset_test(self):
        return self._dataset_test

    @dataset_test.setter
    def dataset_test(self, value):
        self._dataset_test = value

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    @property
    def randomize_training_data(self):
        return self._randomize_training_data

    @randomize_training_data.setter
    def randomize_training_data(self, value):
        self._randomize_training_data = value

    @property
    def balance_training_data(self):
        return self._balance_training_data

    @balance_training_data.setter
    def balance_training_data(self, value):
        self._balance_training_data = value

    @property
    def same_training_data(self):
        return self._same_training_data

    @same_training_data.setter
    def same_training_data(self, value):
        self._same_training_data = value

    @property
    def data_points(self):
        return self._data_points

    @data_points.setter
    def data_points(self, value):
        self._data_points = value
    
    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = value

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, value):
        self._test = value

    @property
    def label_type(self):
        return self._label_type

    @label_type.setter
    def label_type(self, value):
        self._label_type = value

    @property
    def label_size(self):
        return self._label_size

    @label_size.setter
    def label_size(self, value):
        self._label_size = value
    
    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    @property
    def loss_weights(self):
        return self._loss_weights

    @loss_weights.setter
    def loss_weights(self, value):
        self._loss_weights = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def momentum(self):
        return self._momentum

    @momentum.setter
    def momentum(self, value):
        self._momentum = value

    @property
    def num_out_channels(self):
        return self._num_out_channels

    @num_out_channels.setter
    def num_out_channels(self, value):
        self._num_out_channels = value

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def start_epoch(self):
        return self._start_epoch

    @start_epoch.setter
    def start_epoch(self, value):
        self._start_epoch = value
        
    @property
    def end_epoch(self):
        return self._end_epoch

    @end_epoch.setter
    def end_epoch(self, value):
        self._end_epoch = value

    @property
    def train_steps(self):
        return self._train_steps

    @train_steps.setter
    def train_steps(self, value):
        self._train_steps = value

    @property
    def val_steps(self):
        return self._val_steps

    @val_steps.setter
    def val_steps(self, value):
        self._val_steps = value

    @property
    def test_steps(self):
        return self._test_steps

    @test_steps.setter
    def test_steps(self, value):
        self._test_steps = value

    @property
    def save_data(self):
        return self._save_data

    @save_data.setter
    def save_data(self, value):
        self._save_data = value

    @property
    def save_model(self):
        return self._save_model

    @save_model.setter
    def save_model(self, value):
        self._save_model = value

    @property
    def save_graphs(self):
        return self._save_graphs

    @save_graphs.setter
    def save_graphs(self, value):
        self._save_graphs = value

    @property
    def repeat_experiments(self):
        return self._repeat_experiments

    @repeat_experiments.setter
    def repeat_experiments(self, value):
        self._repeat_experiments = value

    @property
    def at_which_run(self):
        return self._at_which_run

    @at_which_run.setter
    def at_which_run(self, value):
        self._at_which_run = value

    @property
    def experiment_state(self):
        return self._experiment_state

    @experiment_state.setter
    def experiment_state(self, value):
        self._experiment_state = value

    @property
    def theta_init(self):
        return self._theta_init

    @theta_init.setter
    def theta_init(self, value):
        self._theta_init = value

    @property
    def srxy_init(self):
        return self._srxy_init

    @srxy_init.setter
    def srxy_init(self, value):
        self._srxy_init = value

    @property
    def weight_transform(self):
        return self._weight_transform

    @weight_transform.setter
    def weight_transform(self, value):
        self._weight_transform = value

    @property
    def srxy_smoothness(self):
        return self._srxy_smoothness

    @srxy_smoothness.setter
    def srxy_smoothness(self, value):
        self._srxy_smoothness = value

    @property
    def k0_init(self):
        return self._k0_init

    @k0_init.setter
    def k0_init(self, value):
        self._k0_init = value

    @property
    def transformation_groups(self):
        return self._transformation_groups

    @transformation_groups.setter
    def transformation_groups(self, value):
        self._transformation_groups = value
    
    @property
    def k0_groups(self):
        return self._k0_groups

    @k0_groups.setter
    def k0_groups(self, value):
        self._k0_groups = value

    @property
    def k_shape(self):
        return self._k_shape

    @k_shape.setter
    def k_shape(self, value):
        self._k_shape = value

    @property
    def max_pool_temporal(self):
        return self._max_pool_temporal

    @max_pool_temporal.setter
    def max_pool_temporal(self, value):
        self._max_pool_temporal = value
    
    @property
    def conv_k_hw(self):
        return self._conv_k_hw

    @conv_k_hw.setter
    def conv_k_hw(self, value):
        self._conv_k_hw = value
        
    @property
    def load_num_frames(self):
        return self._load_num_frames

    @load_num_frames.setter
    def load_num_frames(self, value):
        self._load_num_frames = value

    @property
    def conv1_k_t(self):
        return self._conv1_k_t

    @conv1_k_t.setter
    def conv1_k_t(self, value):
        self._conv1_k_t = value

    @property
    def do_batchnorm(self):
        return self._do_batchnorm

    @do_batchnorm.setter
    def do_batchnorm(self, value):
        self._do_batchnorm = value

