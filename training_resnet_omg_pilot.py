import chainer
import numpy as np
# from deepimpression2.model_53 import Deepimpression
from model_0 import Siamese
import data_loading as L

import deepimpression2.constants as C
from chainer.functions import sigmoid_cross_entropy, mean_absolute_error, softmax_cross_entropy, mean_squared_error
from chainer.optimizers import Adam
import h5py as h5
import deepimpression2.paths as P
# import deepimpression2.chalearn20.data_utils as D
import deepimpression2.chalearn30.data_utils as D
import time
from chainer.backends.cuda import to_gpu, to_cpu
import deepimpression2.util as U
import os
import cupy as cp
from chainer.functions import expand_dims
from random import shuffle
from tqdm import tqdm


my_model = Siamese()

my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8, weight_decay_rate=0.0001)
# my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
my_optimizer.setup(my_model)

if C.ON_GPU:
    my_model = my_model.to_gpu(device=C.DEVICE)

print('Initializing')
print('model initialized with %d parameters' % my_model.count_params())

epochs = 100
batches = 32

frame_matrix, valid_story_idx_all = L.make_frame_matrix()

train_total_steps = int(1600 / batches)  # we have 40**2 possible pairs of id-stories in training
# train_total_steps = int(100 / batches)  # we have 40**2 possible pairs of id-stories in training
val_total_steps = int(100 / batches)  # we have 10**2 possible pairs of id-stories in validation


def run(which, model, optimizer):
    _loss_steps = []
    assert (which in ['train', 'test', 'val'])
    val_idx = None
    steps = None
    if which == 'train':
        val_idx = valid_story_idx_all[0]
        steps = train_total_steps
    elif which == 'val':
        val_idx = valid_story_idx_all[1]
        steps = val_total_steps
    elif which == 'test':
        raise NotImplemented

    # print('%s, steps: %d' % (which, steps))


    for s in tqdm(range(steps)):
        data_left, data_right, labels = L.load_data(which, frame_matrix, val_idx, batches)

        # labels, data = L.dummy_load_data()  # for debugging purposes only

        if C.ON_GPU:
            data_left = to_gpu(data_left, device=C.DEVICE)
            data_right = to_gpu(data_right, device=C.DEVICE)
            labels = to_gpu(labels, device=C.DEVICE)

        with cp.cuda.Device(C.DEVICE):
            if which == 'train':
                config = True
            else:
                config = False

            with chainer.using_config('train', config):
                if which == 'train':
                    model.cleargrads()
                prediction = model(data_left, data_right)

                loss = mean_squared_error(prediction, labels)
                _loss_steps.append(float(loss.data))

                if which == 'train':
                    loss.backward()
                    optimizer.update()

    return _loss_steps

        # TODO: implement
        # L.update_step_logs(which, float(loss.data))


print('Enter training loop with validation')
for e in range(0, epochs):
    # ----------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------
    loss_train = run(which='train', model=my_model, optimizer=my_optimizer)
    L.update_logs(which='train', loss=float(np.mean(loss_train)), epoch=e, model_num=0, experiment_number=0)
    # L.make_epoch_plot(which)
    # ----------------------------------------------------------------------------
    # validation
    # ----------------------------------------------------------------------------
    loss_val = run(which='val', model=my_model, optimizer=my_optimizer)
    L.update_logs(which='val', loss=float(np.mean(loss_val)), epoch=e, model_num=0, experiment_number=0)
    
    print('epoch %d, train_loss: %f, val_loss: %f' % (e, float(np.mean(loss_train)), float(np.mean(loss_val))))
    # ----------------------------------------------------------------------------
    # test
    # ----------------------------------------------------------------------------
    # times = 1
    # for i in range(1):
    #     if times == 1:
    #         ordered = True
    #         save_all_results = True
    #     else:
    #         ordered = False
    #         save_all_results = False
    #
    #     run(which='test', steps=test_steps, which_labels=test_labels, frames=id_frames,
    #         model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_test,
    #         loss_saving=test_loss, which_data=test_on, ordered=ordered, save_all_results=save_all_results,
    #         twostream=True)

    # save model
    # if ((e + 1) % 10) == 0:
    # save_location = '/scratch/users/gabras/data/omg_empathy/saving_data/models'
    # name = os.path.join(save_location, 'epoch_%d_0' % e)
    # chainer.serializers.save_npz(name, my_model)