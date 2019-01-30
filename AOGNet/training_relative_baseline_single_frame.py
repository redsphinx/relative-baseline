import mxnet as mx
import numpy as np


print('Initializing')

# fetch data TODO
data = []
label = []
batch_size = 32
ntrain = int(data.shape[0]*0.8)
train_iter = mx.io.NDArrayIter(data[:ntrain, :], label[:ntrain], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data[ntrain:, :], label[ntrain:], batch_size)

# load symbols
path = '/home/gabras/deployed/relative-baseline/AOGNet/aognet_cifar10_ps_4_bottleneck_1M-symbol.json'
sym = mx.symbol.load(path)

# visualize the network
# grp = mx.viz.plot_network(sym, node_attrs={"shape": "oval", "fixedsize": "false"})
# grp.save('aognet_cifar10_ps_4_bottleneck_1M.dot', '/home/gabras/deployed/relative-baseline/AOGNet/')

# symbol to module
mod = mx.mod.Module(symbol=sym,
                    context=mx.cpu(),
                    data_names=[], #['data'], TODO
                    label_names=[]) #['softmax_label']) TODO

# training
mod.bind() # TODO
mod.init_params(initializer=mx.init.Xavier())
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))
metric = mx.metric.create('mse')

for epoch in range(5):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)  # compute predictions
        mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
        mod.backward()  # compute gradients
        mod.update()  # update parameters
    print('Epoch %d, Training %s' % (epoch, metric.get()))

print('yes')