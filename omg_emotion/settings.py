

class ProjectVariable(object):
    def __init__(self):
        """
        Default values for all the experimental variables.
        """
        # int, which gpu to use {None, 0, 1, etc}
        self._device = 0

        # str, which model to load
        self._load_model = None

        # int, experiment data for log
        self._model_number = None
        self._experiment_number = None
        
        # int, the current epoch
        self._current_epoch = None

        # list of str, which datasets to train, val and test on
        self._dataset_train = ['omg_emotion']
        self._dataset_val = ['omg_emotion']
        self._dataset_test = ['omg_emotion']

        # bool, which procedures to perform
        self._train = True
        self._val = True
        self._test = True

        # list of float, learning rate
        self._learning_rate = [0.0001]

        # str, loss function
        self._loss_function = 'cross_entropy'

        # list of str, optimizer
        self._optimizer = ['adam']

        # bool, if in debug mode
        self._debug_mode = False

        # depending on debug mode
        if self._debug_mode:
            self._batch_size = 16
            self._start_epoch = 0
            self._end_epoch = 2
            self._train_steps = 10
            self._val_steps = 1
            self._test_steps = 1
            self._save_data = False
            self._save_model = False
            self._save_graphs = False
        else:
            self._batch_size = 32
            self._start_epoch = -1
            self._end_epoch = 100
            self._train_steps = 50
            self._val_steps = 10
            self._test_steps = 10
            self._save_data = True
            self._save_model = True
            self._save_graphs = True
    
    
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
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def debug_mode(self):
        return self._debug_mode

    @debug_mode.setter
    def debug_mode(self, value):
        self._debug_mode = value

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