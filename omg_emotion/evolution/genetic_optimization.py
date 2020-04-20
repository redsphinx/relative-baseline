import os
import numpy as np


CONV_LAYER_TYPES = ['conv3dttn', 'conv3d']
POOL_LAYER_TYPES = ['max', 'avg']
ARCHITECTURE_COMPONENTS = ['conv', 'pool', 'fc']
PROBABILITY_DICT_L1 = {'lr': 0.3,
                       'num_conv_layers': 0.4,
                       'num_channels': 0.2,
                       'kernel_size_per_layer': 0,
                       'conv_layer_type': 0,
                       'pooling_after_conv': 0,
                       'pooling_final': 0,
                       'fc_layer': 0.1,
                       'architecture_order': 0
                       }
PROBABILITY_DICT_L2 = {'lr': 0,
                       'num_conv_layers': 0.3,
                       'num_channels': 0.45,
                       'kernel_size_per_layer': 0.05,
                       'conv_layer_type': 0,
                       'pooling_after_conv': 0.05,
                       'pooling_final': 0.05,
                       'fc_layer': 0.1,
                       'architecture_order': 0
                       }
EXCEPTION_IND = [4, 8, 9]

# example:
# lr=0.0003, num_conv=2, num_chan=[6, 16], kernels=[5, 5], layer_type=[0, 0], pooling_conv=0, pooling_final=1,
# arch_order=[0, 1, 0, 1, 2, 2]

def write_genome(genotype, in_features):
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
              'architecture_order': architecture_order,
              'in_features': in_features}

    return genome


def fitness_function(results_collapsed, results_val, results_train):
    # results = {'expnum': [has_collapsed, val_acc, train_acc]}
    score_dict = {}

    results_val = {k: v for k, v in sorted(results_val.items(), reverse=True, key=lambda item: item[1])}
    results_train = {k: v for k, v in sorted(results_train.items(), reverse=True, key=lambda item: item[1])}

    for i in results_collapsed.keys():
        score = 0
        if int(results_collapsed[i]) == 0:
            score = score + 100

        if results_val[i] > 1/27:
            score = score + 20

        # best val score
        if list(results_val.keys()).index(i) == 0:
            score = score + 6
        elif list(results_val.keys()).index(i) == 1:
            score = score + 3

        if results_train[i] > 1 / 27:
            score = score + 10

        # best train score
        if list(results_train.keys()).index(i) == 0:
            score = score + 3
        elif list(results_train.keys()).index(i) == 1:
            score = score + 1

        score_dict[i] = score

    return score_dict


def crossover(results_collapsed, fitness, prev_genotypes):
    # if there are 2 or more individuals in the population who did not collapse, copy what they have in common
    new_genotype = [None] * 11
    crossover_ind = []

    col_arr = np.array(list(results_collapsed.items()), dtype=int)
    if sum(col_arr[:, 1]) <= 1:
        # copy parts that are same, if any
        genos = []
        for i in results_collapsed.keys():
            if results_collapsed[i] == '0':
                # get the genotype
                genos.append(prev_genotypes[i])

        # of them did not collapse
        if len(genos) == 2:
            for i in range(len(genos[0])):
                if i not in EXCEPTION_IND:
                    if genos[0][i] == genos[1][i]:
                        new_genotype[i] = genos[0][i]
                        crossover_ind.append(i)

        # all of them did not collapse
        else:
            for i in range(len(genos[0])):
                if i not in EXCEPTION_IND:
                    if genos[0][i] == genos[1][i] == genos[2][i]:
                        new_genotype[i] = genos[0][i]
                        crossover_ind.append(i)
                    elif genos[0][i] == genos[1][i]:
                        new_genotype[i] = genos[0][i]
                        crossover_ind.append(i)
                    elif genos[1][i] == genos[2][i]:
                        new_genotype[i] = genos[1][i]
                        crossover_ind.append(i)
                    elif genos[0][i] == genos[2][i]:
                        new_genotype[i] = genos[0][i]
                        crossover_ind.append(i)

    return new_genotype, crossover_ind


def mutation(genotype, crossover_ind):





def assess_genotype_viability(genotype):
    in_features = None

    return False, in_features

def attempt_fix_with_padding():
    pass


def generate_genotype(results, prev_genotypes):
    genotype = []
    in_features = []
    viable = False

    fitness = fitness_function(results[0], results[1], results[2])
    # fitness is a dictionary containing fitness score per model, sorted by best model first
    # it checks the has_collapsed, train_acc and val_acc variables

    genotype, crossover_ind = crossover(results[0], fitness, prev_genotypes)

    genotype = mutation(genotype, crossover_ind)

    while not viable:
        viable, in_features = assess_genotype_viability(genotype)

    return genotype, in_features