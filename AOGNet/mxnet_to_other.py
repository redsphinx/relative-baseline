import mmdnn
import subprocess
import os



mxnet_model = '/home/gabi/PycharmProjects/deployed-schmidhuber/relative-baseline/AOGNet/aognet_cifar10_ps_4_bottleneck_1M-symbol.json'
pythorch_model = ''
command = 'python -m mmdnn.conversion._script.convertToIR -f mxnet -n %s -w' % mxnet_model
subprocess.call(command, shell=True)
