import os
import numpy as np
from random import randrange
import random


CONV_LAYER_TYPES = ['conv3dttn', 'conv3d']
POOL_LAYER_TYPES = ['max', 'avg']
ARCHITECTURE_COMPONENTS = ['conv', 'pool', 'fc']
PROBABILITY_DICT_L1 = {'lr': 0.3,
                       'num_conv_layers': 0.4,
                       'num_channels': 0.2,
                       'kernel_size_per_layer': 0,
                       'padding': 0,
                       'conv_layer_type': 0,
                       'pooling_after_conv': 0,
                       'pooling_final': 0,
                       'fc_layer': 0.1,
                       'architecture_order': 0,
                       'in_features': 0
                       }
PROBABILITY_DICT_L2 = {'lr': 0,
                       'num_conv_layers': 0.3,
                       'num_channels': 0.45,
                       'kernel_size_per_layer': 0.05,
                       'padding': 0,
                       'conv_layer_type': 0,
                       'pooling_after_conv': 0.05,
                       'pooling_final': 0.05,
                       'fc_layer': 0.1,
                       'architecture_order': 0,
                       'in_features': 0
                       }
EXCEPTION_IND = [4, 8, 9]
GENOTYPE_KEYS = list(PROBABILITY_DICT_L2.keys())
LEN_GENOTYPE = len(GENOTYPE_KEYS)

# 0 'lr'
# 1 'num_conv_layers'
# 2 'num_channels'
# 3 'kernel_size_per_layer'
# 4 'padding'
# 5 'conv_layer_type'
# 6 'pooling_after_conv'
# 7 'pooling_final'
# 8 'fc_layer'
# 9 'architecture_order'
# 10 'in_features'


# example:
# lr=0.0003, num_conv=2, num_chan=[6, 16], kernels=[5, 5], layer_type=[0, 0], pooling_conv=0, pooling_final=1,
# arch_order=[0, 1, 0, 1, 2, 2]

def write_genome(genotype):
    lr, num_conv, num_chan, kernels, padding, layer_type, pooling_conv, pooling_final, fc_size, arch_order, in_features = genotype

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
    # else, do a recombination of the genotype
    new_genotype = [None] * LEN_GENOTYPE
    crossover_ind = []

    fitness_sorted = {k: v for k, v in sorted(fitness.items(), reverse=True, key=lambda item: item[1])}
    most_fit = list(fitness_sorted)[0]

    col_arr = np.array(list(results_collapsed.items()), dtype=int)
    # if only 1 or less collapsed
    if sum(col_arr[:, 1]) <= 1:
        # copy parts that are same, if any
        genos = []
        for i in results_collapsed.keys():
            if results_collapsed[i] == '0':
                # get the genotype
                genos.append(prev_genotypes[i])

        # 2 of them did not collapse
        if len(genos) == 2:
            for i in range(LEN_GENOTYPE):
                if i not in EXCEPTION_IND:
                    if genos[0][i] == genos[1][i]:
                        new_genotype[i] = genos[0][i]
                        crossover_ind.append(i)

        # all of them did not collapse
        else:
            for i in range(LEN_GENOTYPE):
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

        # for the indices that are still None, copy the value of the best individual
        for i in range(LEN_GENOTYPE):
            if i not in crossover_ind and i not in EXCEPTION_IND:
                new_genotype[i] = prev_genotypes[int(most_fit)][i]
    # if all collapsed
    else:
        for i in range(LEN_GENOTYPE):
            if i not in crossover_ind and i not in EXCEPTION_IND:
                new_genotype[i] = prev_genotypes[randrange(3)+1][i]

    return new_genotype, crossover_ind


def decision(probability):
    return random.random() < probability


def mutate_by_unit(genotype, value_index, param, direction):
    value = genotype[value_index]

    if param == 'lr':
        unit = 0.0001
        if direction:
            value = value + unit
        else:
            value = value - unit
    elif param == 'num_conv_layers':
        unit = 1
        if direction:
            value = value + unit
        else:
            if value - unit >= 2:
                value = value - unit
            else:
                value = value + unit

    elif param == 'num_channels':
        unit = 6

        if len(value) != genotype[1]:
            value.append(value[-1])

        # you can only add or subtract channels if the result is that the layer before it has less or
        # equal number of channels

        eligible_chan_ind_ADD = []
        eligible_chan_ind_SUB = []
        for i in range(len(value)):
            if i != len(value) - 1:
                if value[i+1] - value[i] >= unit:
                    eligible_chan_ind_ADD.append(i)

                if value[i] - unit >= unit:
                    if i != 0:
                        if value[i] - unit >= value[i-1]:
                            eligible_chan_ind_SUB.append(i)
                    else:
                        eligible_chan_ind_SUB.append(i)

        if direction:
            indx = eligible_chan_ind_ADD[randrange(len(eligible_chan_ind_ADD))]
            genotype[value_index][indx] = genotype[value_index][indx] + unit
        else:
            indx = eligible_chan_ind_SUB[randrange(len(eligible_chan_ind_SUB))]
            genotype[value_index][indx] = genotype[value_index][indx] + unit

        value = genotype[value_index]

    elif param == 'kernel_size_per_layer':
        unit = 2

        if len(genotype[value_index]) != genotype[1]:
            genotype[value_index].append(genotype[value_index][-1])

        # choose one of the kernels
        indx = randrange(len(genotype[value_index]))

        if direction:
            genotype[value_index][indx] = genotype[value_index][indx] + unit
        else:
            if genotype[value_index][indx] - unit < 0:
                genotype[value_index][indx] = genotype[value_index][indx] + unit

        value = genotype[value_index]

    elif param == 'conv_layer_type':
        if len(genotype[value_index]) != genotype[1]:
            genotype[value_index].append(genotype[value_index][-1])

        value = genotype[value_index]
    elif param == 'pooling_after_conv':
        value = abs(genotype[value_index] - 1)
    elif param == 'pooling_final':
        value = abs(genotype[value_index] - 1)
    elif param == 'fc_layer':
        unit = 128
        if direction:
            value = genotype[value_index] + unit
        else:
            if value - unit > 512:
                value = genotype[value_index] - unit
            else:
                value = genotype[value_index] + unit//2

    return value



def mutation(genotype, crossover_ind, collapsed):
    # determine if in L1 or L2
    # if none collapsed, L2 else L1
    collapsed = np.array(list(collapsed.items()), dtype=int)

    if sum(collapsed[:, 1]) == 0:
        level = 2
    else:
        level = 1

    # if L1, modify prob table, else modify L2 table
    if level == 1:
        prob_tab = PROBABILITY_DICT_L1
    else:
        prob_tab = PROBABILITY_DICT_L2

    # use crossover_ind to lower chances that good genes get changed
    for i in crossover_ind:
        prob_tab[GENOTYPE_KEYS[i]] = prob_tab[GENOTYPE_KEYS[i]] / 2

    # for item in table, decide if we're going to mutate depending on it's percentage

    g1 = genotype
    g2 = genotype
    g3 = genotype
    new_genos = [g1, g2, g3]

    for i, k in GENOTYPE_KEYS:
        if i not in EXCEPTION_IND:
            for m in range(3):
                # get the prob
                if decision(prob_tab[k]):
                    direction = randrange(2)
                    # 0 = decrease, 1 = increase
                    new_genos[m][i] = mutate_by_unit(new_genos[m], i, k, direction)

    # TODO future
    # if genotype is novel, keep it
    # else, generate a new one

    return new_genos


def create_architecture_order(genotypes):
    # [0: 'conv', 1: 'pool', 2: 'fc']
    # for now we only use conv layers in the middle
    first = [0, 1]
    last = [1, 2, 2]

    for i in range(len(genotypes)):
        middle = [0] * genotypes[i][1]
        arch = first + middle + last
        genotypes[i][9] = arch

    return genotypes


def layer_out_size(t, h, w, layer_type, k, p, s, div):
    assert layer_type in ['conv', 'pool']

    if layer_type == 'conv':
        t = (t - k + 2 * p) / s + 1
        h = (h - k + 2 * p) / s + 1
        w = (w - k + 2 * p) / s + 1
    elif layer_type == 'pool':
        t = int(np.floor(t / div))
        h = int(np.floor(h / div))
        w = int(np.floor(w / div))

    print("after %s: t=%d, h=%d, w=%d" % (layer_type, t, h, w))

    return t, h, w


def names_count(list_of_names):
    types = ['conv', 'pool', 'fc']
    counting_types = [0, 0, 0]
    new_list = []

    for name in list_of_names:
        new_list.append(counting_types[types.index(name)])
        counting_types[types.index(name)] = counting_types[types.index(name)] + 1

    return new_list

# 0 'lr'
# 1 'num_conv_layers'
# 2 'num_channels'
# 3 'kernel_size_per_layer'
# 4 'padding'
# 5 'conv_layer_type'
# 6 'pooling_after_conv'
# 7 'pooling_final'
# 8 'fc_layer'
# 9 'architecture_order'
# 10 'in_features'
def calculate_in_features(genotype):
    # genotype is a list
    t = 30
    h = 50
    w = 75

    architecture_order = [ARCHITECTURE_COMPONENTS[i] for i in genotype[9]]
    names_ind = names_count(architecture_order)
    padding = [0] * genotype[1]
    viable = False
    where_padding_added = [0] * len(padding)
    loc_start_pad = 0

    while not viable:
        for j, layer in architecture_order:
            if layer != 'fc':
                if layer == 'conv':
                    k = genotype[3][names_ind[j]]
                    p = padding[names_ind[j]]
                    s = 1
                    div = None
                else:
                    k = None
                    p = None
                    s = None
                    div = 2

                t, h, w = layer_out_size(t, h, w, layer, k, p, s, div)

        if t <= 0:
            # start padding the beginning. add padding to the next layer if it is not sufficient.
            # if padding of 1 is reached in all layers, start again at the beginning.
            padding[loc_start_pad] = 1
            where_padding_added[loc_start_pad] = where_padding_added[loc_start_pad] + 1
            loc_start_pad = loc_start_pad + 1
            if loc_start_pad == len(padding) - 1:
                loc_start_pad = 0

        else:
            viable = True

    return t, h, w, padding, viable


def assess_genotype_viability(genotype):

    t, h, w, padding, viable = calculate_in_features(genotype)

    in_features = t * h * w

    genotype[4] = padding
    genotype[10] = in_features

    return genotype


def generate_genotype(results, prev_genotypes):
    # results[0], results[1], results[2] = col, val, train

    fitness = fitness_function(results[0], results[1], results[2])
    # fitness is a dictionary containing fitness score per model, sorted by best model first
    # it checks the has_collapsed, train_acc and val_acc variables

    new_geno, crossover_ind = crossover(results[0], fitness, prev_genotypes)
    new_genotypes = mutation(new_geno, crossover_ind, results[0])
    new_genotypes = create_architecture_order(new_genotypes)


    for i in range(3):
        genotype = assess_genotype_viability(new_genotypes[i])
        new_genotypes[i] = genotype

    return new_genotypes