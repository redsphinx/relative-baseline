import mxnet as mx
import numpy as np
import h5py as h5
import deepimpression2.constants as C
import deepimpression2.paths as P
import deepimpression2.chalearn30.data_utils as D
from random import shuffle
import time


print('Initializing')

# fetch data
train_labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
training_steps = len(train_labels) // C.TRAIN_BATCH_SIZE
id_frames = h5.File(P.NUM_FRAMES, 'r')



# data iterator

def run(module, steps, which_labels, frames, which='train', ordered=False, twostream=False, same_frame=False):
    print('steps: ', steps)
    assert(which in ['train', 'test', 'val'])

    which_batch_size = C.TRAIN_BATCH_SIZE

    _labs = list(which_labels)

    if not ordered:
        shuffle(_labs)

    for s in range(steps):
        labels_selected = _labs[s * which_batch_size:(s + 1) * which_batch_size]
        assert (len(labels_selected) == which_batch_size)
        labels_face, face_data, _ = D.load_data(labels_selected, which_labels, frames, which_data='face', resize=True,
                                             ordered=ordered, twostream=twostream, same_frame=same_frame)

        # shuffle data and labels in same order
        if which != 'test':
            shuf = np.arange(which_batch_size)
            shuffle(shuf)
            face_data = face_data[shuf]
            labels_face = labels_face[shuf]

        # -----------------------------------------------------------
        metric.reset()
        module.forward(face_data, is_train=True)  # compute predictions
        module.update_metric(metric, labels_face)  # accumulate prediction accuracy
        module.backward()  # compute gradients
        module.update()  # update parameters
        # -----------------------------------------------------------

    print('Epoch %d, Training %s' % (epoch, metric.get()))


# load symbols
path = '/home/gabras/deployed/relative-baseline/AOGNet/aognet_cifar10_ps_4_bottleneck_1M-symbol.json'
sym = mx.symbol.load(path)

# visualize the network
# grp = mx.viz.plot_network(sym, node_attrs={"shape": "oval", "fixedsize": "false"})
# grp.save('aognet_cifar10_ps_4_bottleneck_1M.dot', '/home/gabras/deployed/relative-baseline/AOGNet/')

# symbol to module
mod = mx.mod.Module(symbol=sym,
                    context=mx.cpu())

# training
mod.bind(data_shapes=[],
         label_shapes=[])
mod.init_params(initializer=mx.init.Xavier())
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))
metric = mx.metric.create('mse')


for epoch in range(5):
    run(module=mod, steps=training_steps, which_labels=train_labels, frames=id_frames)

