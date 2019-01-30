import logging
import random
logging.getLogger().setLevel(logging.INFO)

import mxnet as mx
import numpy as np


#---------------------------------------------------------------------------------------
# PRELIMINARY
#---------------------------------------------------------------------------------------

mx.random.seed(1234)
np.random.seed(1234)
random.seed(1234)

fname = mx.test_utils.download('https://s3.us-east-2.amazonaws.com/mxnet-public/letter_recognition/letter-recognition.data')
data = np.genfromtxt(fname, delimiter=',')[:,1:]
label = np.array([ord(l.split(',')[0])-ord('A') for l in open(fname, 'r')])

batch_size = 32
ntrain = int(data.shape[0]*0.8)
train_iter = mx.io.NDArrayIter(data[:ntrain, :], label[:ntrain], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data[ntrain:, :], label[ntrain:], batch_size)

#---------------------------------------------------------------------------------------
# DEFINE NETWORK
#---------------------------------------------------------------------------------------

net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=26)
net = mx.sym.SoftmaxOutput(net, name='softmax')
mx.viz.plot_network(net, node_attrs={"shape":"oval","fixedsize":"false"})

#---------------------------------------------------------------------------------------
# SYMBOL TO MODULE
#---------------------------------------------------------------------------------------

mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

#---------------------------------------------------------------------------------------
# TRAINING -- INTERMEDIATE LEVEL
#---------------------------------------------------------------------------------------
intermediate = True
if intermediate:
    # allocate memory given the input data and label shapes
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    # initialize parameters by uniform random numbers
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    # use SGD with learning rate 0.1 to train
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
    # use accuracy as the metric
    metric = mx.metric.create('acc')
    # train 5 epochs, i.e. going over the data iter one pass
    for epoch in range(5):
        train_iter.reset()
        metric.reset()
        for batch in train_iter:
            mod.forward(batch, is_train=True)       # compute predictions
            mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
            mod.backward()                          # compute gradients
            mod.update()                            # update parameters
        print('Epoch %d, Training %s' % (epoch, metric.get()))

#---------------------------------------------------------------------------------------
# TRAINING -- HIGH LEVEL
#---------------------------------------------------------------------------------------
high = False

# for saving / loading
model_prefix = 'mx_mlp'
checkpoint = mx.callback.do_checkpoint(model_prefix)

if high:
    # reset train_iter to the beginning
    train_iter.reset()

    # create a module
    mod = mx.mod.Module(symbol=net,
                        context=mx.cpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])

    # fit the module
    mod.fit(train_iter,
            eval_data=val_iter,
            optimizer='sgd',
            optimizer_params={'learning_rate': 0.1},
            eval_metric='acc',
            num_epoch=7)
    # to save checkpoint when training
            # epoch_end_callback=checkpoint)

# #---------------------------------------------------------------------------------------
# # PREDICT AND EVALUATE
# #---------------------------------------------------------------------------------------
# # for prediction only
# y = mod.predict(val_iter)
# assert y.shape == (4000, 26)
#
# # for evaluation only
# score = mod.score(val_iter, ['acc'])
# print("Accuracy score is %f" % (score[0][1]))
# assert score[0][1] > 0.76, "Achieved accuracy (%f) is less than expected (0.76)" % score[0][1]
#
# #---------------------------------------------------------------------------------------
# # SAVE AND LOAD
# #---------------------------------------------------------------------------------------
#
# sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
# assert sym.tojson() == net.tojson()
#
# # assign the loaded parameters to the module
# mod.set_params(arg_params, aux_params)
#
# # or we can load like this during training:
# load_during_training = False
# if load_during_training:
#     mod = mx.mod.Module(symbol=sym)
#     mod.fit(train_iter,
#             num_epoch=21,
#             arg_params=arg_params,
#             aux_params=aux_params,
#             begin_epoch=3)
#     assert score[0][1] > 0.77, "Achieved accuracy (%f) is less than expected (0.77)" % score[0][1]