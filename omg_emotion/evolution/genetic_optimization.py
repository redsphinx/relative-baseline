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
    new_genotype = [None] * 8
    crossover_ind = []

    fitness_sorted = {k: v for k, v in sorted(fitness.items(), reverse=True, key=lambda item: item[1])}
    most_fit = list(fitness_sorted)[0]

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

        # for the indices that are still None, copy the value of the best individual
        for i in range(11):
            if i not in crossover_ind and i not in EXCEPTION_IND:
                new_genotype[i] = prev_genotypes[int(most_fit)][i]
    else:
        for i in range(11):
            if i not in crossover_ind and i not in EXCEPTION_IND:
                new_genotype[i] = prev_genotypes[randrange(3)+1][i]

    return new_genotype, crossover_ind


def decision(probability):
    return random.random() < probability


def mutate_by_unit(value, param, direction):
    pass




def mutation(genotype, crossover_ind, collapsed):
    # determine if in L1 or L2
    # if none collapsed, L2 else L1
    collapsed = np.array(list(collapsed.items()), dtype=int)

    prob_tab_keys = list(PROBABILITY_DICT_L1.keys())

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
        prob_tab[prob_tab_keys[i]] = prob_tab[prob_tab_keys[i]] / 2

    # for item in table, decide if we're going to mutate depending on it's percentage

    g1 = genotype
    g2 = genotype
    g3 = genotype
    new_genos = [g1, g2, g3]

    for i, k in prob_tab_keys:
        if i not in EXCEPTION_IND:
            for m in range(3):
                # get the prob
                if decision(prob_tab[k]):
                    direction = randrange(2)
                    # Fix: check if the changes are inorder so that there's no conflict with sizes
                    # 0 = decrease, 1 = increase
                    new_genos[m][i] = mutate_by_unit(new_genos[m][i], prob_tab[k], direction)

    # TODO future
    # if genotype is novel, keep it
    # else, generate a new one

    return new_genos


def calculate_in_features():
    pass

def attempt_fix_with_padding(genotype):

    # TODO: write padding and in_features
    return genotype
    pass


def assess_genotype_viability(genotype):
    in_features = None
    viable = False

    while not viable:
        genotype, viable = attempt_fix_with_padding(genotype)

    return genotype


def generate_genotype(results, prev_genotypes):
    # results[0], results[1], results[2] = col, val, train

    fitness = fitness_function(results[0], results[1], results[2])
    # fitness is a dictionary containing fitness score per model, sorted by best model first
    # it checks the has_collapsed, train_acc and val_acc variables

    new_geno, crossover_ind = crossover(results[0], fitness, prev_genotypes)
    new_genotypes = mutation(new_geno, crossover_ind, results[0])

    for i in range(3):
        genotype, in_features = assess_genotype_viability(new_genotypes[i])
        new_genotypes[i] = genotype

    return new_genotypes