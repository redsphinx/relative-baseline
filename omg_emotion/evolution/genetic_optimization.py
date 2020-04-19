import os
import numpy as np


CONV_LAYER_TYPES = ['conv3dttn', 'conv3d']
POOL_LAYER_TYPES = ['max', 'avg']
ARCHITECTURE_COMPONENTS = ['conv', 'pool', 'fc']

# example:
# lr=0.0003, num_conv=2, num_chan=[6, 16], kernels=[5, 5], layer_type=[0, 0], pooling_conv=0, pooling_final=1,
# arch_order=[0, 1, 0, 1, 2, 2]

def write_genome(genotype):
    lr, num_conv, num_chan, kernels, padding, layer_type, pooling_conv, pooling_final, fc_size, arch_order = genotype

    assert len(num_chan) == num_conv
    assert len(kernels) == num_conv
    assert len(layer_type) == num_conv

    # convert layer type to list of strings
    conv_layer_type = [CONV_LAYER_TYPES[i] for i in layer_type]
    # convert architecture order to list of strings
    architecture_order = [ARCHITECTURE_COMPONENTS[i] for i in arch_order]


    genome = {'lr': lr,
              'num_conv_layers': num_conv,
              'num_channels': num_chan,
              'kernel_size_per_layer': kernels,
              'conv_layer_type': conv_layer_type,
              'padding': padding,
              'pooling_after_conv': POOL_LAYER_TYPES[pooling_conv],
              'pooling_final': POOL_LAYER_TYPES[pooling_final],
              'fc_layer': fc_size,
              'architecture_order': architecture_order}

    return genome


def assess_genotype_viability(genotype):
    in_features = None

    return in_features, False

def attempt_fix_with_padding():
    pass


def generate_genotype(results):
    genotype = []
    viable = False

    while not viable:
        viable, in_features = assess_genotype_viability(genotype)

    return genotype, in_features