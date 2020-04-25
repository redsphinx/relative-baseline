import time
import os
import numpy as np
from datetime import date, datetime
from multiprocessing import Pool

from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file
from relative_baseline.omg_emotion import project_paths as PP
from relative_baseline.omg_emotion.evolution import genetic_optimization as GO


def run_single_experiment(project_variable, lr, epochs, out_channels, device, model_number):
    project_variable.nas = True
    project_variable.device = device
    project_variable.end_epoch = epochs
    project_variable.learning_rate = lr
    project_variable.num_out_channels = out_channels
    project_variable.model_number = model_number

    project_variable.save_data = False
    project_variable.save_model = False
    project_variable.save_graphs = False

    project_variable.dataset = 'jester'

    # if you want all the data: train: 150, val: 10, test: 10
    # total_dp = {'train': 118562, 'val': 7393, 'test': 7394}
    project_variable.num_in_channels = 3
    # project_variable.data_points = [2 * 27,  1 * 27, 0 * 27]
    project_variable.batch_size = 2 * 27
    project_variable.use_dali = True
    project_variable.dali_workers = 8
    # for now, use 'all' for val, since idk how to reset the iterator
    project_variable.dali_iterator_size = [5 * 27, 'all', 0]


    project_variable.label_size = 27

    project_variable.load_num_frames = 30
    project_variable.label_type = 'categories'
    project_variable.use_adaptive_lr = True

    project_variable.repeat_experiments = 1
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True

    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'

    project_variable.experiment_state = 'new'
    project_variable.eval_on = 'val'

    project_variable.experiment_number = 1792792989823
    project_variable.sheet_number = 22

    project_variable.repeat_experiments = 1
    project_variable.data_points = [30 * 27, 5 * 27, 0 * 27]
    project_variable.optimizer = 'adam'

    return main_file.run(project_variable)


def auto_search(lr_size, epochs, repeat_run, model_number, conv_layer_channels, device, b=1, e=10, s=2):
    file_name = date.today().strftime('%d%m') + '_' + datetime.now().strftime('%H%M%S') + '.txt'
    save_path = os.path.join(PP.nas_location, file_name)

    header = 'lr,epochs,model_number,conv_layer_channels,train_acc,val_acc,have_collapsed,which_matr_ind\n'
    with open(save_path, 'a') as my_file:
        my_file.write(header)

    for cfg in range(len(conv_layer_channels)):
        out_channels = conv_layer_channels[cfg]

        for lr_incr in range(b, e, s):
            lr = lr_incr / lr_size

            for run in range(repeat_run):
                chnls = '['
                for i in range(len(out_channels)):
                    chnls = chnls + str(out_channels[i]) + ' '

                chnls = chnls[:-1] + ']'

                settings = '%f,%d,%d,%s,' % (lr, epochs, model_number, chnls)
                with open(save_path, 'a') as my_file:
                    my_file.write(settings)

                project_variable = ProjectVariable(debug_mode=True)
                train_acc, val_acc, has_collapsed, collapsed_matrix = \
                    run_single_experiment(project_variable, lr, epochs, out_channels, device, model_number)

                if has_collapsed:
                    ind = np.where(collapsed_matrix == 1)[0]
                else:
                    ind = 420

                results = '%f,%f,%s,%s\n' % (train_acc, val_acc, str(has_collapsed), str(ind))
                with open(save_path, 'a') as my_file:
                    my_file.write(results)


# conv_channels = [[16, 16, 32, 64], [16, 32, 64, 128]]
# auto_search(10000, 30, 3, 15, conv_channels, 1)

# conv_channels = [[32, 32, 64, 64], [32, 32, 64, 128]]
# auto_search(10000, 30, 3, 15, conv_channels, 2)

# conv_channels = [[32, 64, 128, 256]]
# auto_search(10000, 30, 3, 15, conv_channels, 0)

# conv_channels = [[16, 32, 64, 128, 256]]
# auto_search(10000, 30, 3, 16, conv_channels, 1)

# conv_channels = [[16, 32, 32, 64, 64]]
# auto_search(10000, 30, 3, 16, conv_channels, 2)
#
# conv_channels = [[32, 32, 64, 64, 128]]
# auto_search(10000, 30, 3, 16, conv_channels, 2)


def apply_same_settings(project_variable):
    project_variable.nas = True

    project_variable.model_number = 16
    project_variable.sheet_number = 23

    project_variable.end_epoch = 10
    # project_variable.end_epoch = 3
    project_variable.dataset = 'jester'

    project_variable.num_in_channels = 3
    project_variable.label_size = 27
    project_variable.batch_size = 5 * 27
    # project_variable.batch_size = 27
    project_variable.load_num_frames = 30
    project_variable.label_type = 'categories'

    project_variable.repeat_experiments = 1
    project_variable.save_only_best_run = True
    project_variable.same_training_data = True
    project_variable.randomize_training_data = True
    project_variable.balance_training_data = True

    project_variable.theta_init = None
    project_variable.srxy_init = 'eye'
    project_variable.weight_transform = 'seq'

    project_variable.use_dali = True
    project_variable.dali_workers = 32
    project_variable.dali_iterator_size = ['all', 'all', 0]
    # project_variable.dali_iterator_size = [5*27, 10*27, 0]

    project_variable.stop_at_collapse = True
    project_variable.experiment_state = 'new'
    project_variable.optimizer = 'adam'
    project_variable.use_adaptive_lr = True
    project_variable.adapt_eval_on = 'val'
    project_variable.eval_on = 'val'

    return project_variable


def apply_unique_settings(project_variable, genome):
    project_variable.learning_rate = genome['lr']
    project_variable.num_out_channels = genome['num_channels']
    project_variable.genome = genome
    return project_variable


def process_results(results):
    # [(1, True, 0.0, 0.0), (2, True, 0.0, 0.037037037037037035), (3, True, 0.0, 0.07407407407407407)]
    col = {}
    val = {}
    train = {}
    for i in range(len(results)):
        col[str(results[i][0])] = results[i][1]
        val[str(results[i][0])] = results[i][2]
        train[str(results[i][0])] = results[i][3]

    return col, val, train


def evolutionary_search(debug_mode=True):
    # generations = 100
    generations = 100
    genetic_search_path = os.path.join(PP.jester_location, 'genetic_search_log.txt')
    if not os.path.exists(genetic_search_path):
        # delimiter = ';'
        with open(genetic_search_path, 'a') as my_file:
            header = 'generation;exp_num;collapsed;val_acc;train_acc;' \
                     'lr;num_conv_layers;num_channels;kernels;padding;conv_type;pooling_a_conv;' \
                     'pooling_final;fc_layer;arch_order;in_features\n'
            my_file.write(header)

    results = None

    genotype_1, genotype_2, genotype_3 = None, None, None

    for gen in range(generations):
        if gen == 0:
            # manually set first values
            # in_features_1 = 1*6*12
            ##                0   1   2                  3                  4       5            6  7  8      9                          10
            # genotype_1 = (3e-4, 4, [12, 18, 24, 32], [3, 5, 5, 5], [0, 0, 0, 0], [0, 0, 0, 0], 0, 1, 600, [0, 1, 0, 0, 0, 1, 2, 2], in_features_1)

            genotype_1 = (2e-4, 6, [16, 38, 48, 48, 48, 60], [3, 9, 5, 3, 9, 5], [2, 2, 2, 1, 1, 0], [0, 0, 0, 0, 0, 0],
                          0, 1, 560, [0, 1, 0, 0, 0, 0, 0, 1, 2, 2], 72)

            genome_1 = GO.write_genome(genotype_1)

            # in_features_2 = 336
            # genotype_2 = (3e-4, 3, [16, 32, 32], [3, 5, 5], [0, 0, 0], [0, 0, 0], 0, 1, 496, [0, 1, 0, 0, 1, 2, 2], in_features_2)
            genotype_2 = (1.5e-4, 4, [10, 38, 42, 42], [3, 5, 5, 5], [0, 0, 0, 0], [0, 0, 0, 0],
                          0, 1, 768, [0, 1, 0, 0, 0, 1, 2, 2], 72)
            genome_2 = GO.write_genome(genotype_2)

            # in_features_3 = 182
            # genotype_3 = (3e-4, 3, [16, 32, 48], [5, 5, 5], [0, 0, 0], [0, 0, 0], 0, 1, 256, [0, 1, 0, 0, 1, 2, 2], in_features_3)
            genotype_3 = (5e-5, 3, [28, 38, 38], [3, 5, 5], [0, 0, 0], [0, 0, 0],
                          0, 1, 704, [0, 1, 0, 0, 1, 2, 2], 336)
            genome_3 = GO.write_genome(genotype_3)


        else:
            assert results is not None

            new_genotypes= GO.generate_genotype(results, [genotype_1, genotype_2, genotype_3])

            genotype_1, genotype_2, genotype_3 = new_genotypes

            genome_1 = GO.write_genome(genotype_1)
            genome_2 = GO.write_genome(genotype_2)
            genome_3 = GO.write_genome(genotype_3)


        # run models
        pv1 = ProjectVariable(debug_mode)
        pv1 = apply_same_settings(pv1)
        pv1 = apply_unique_settings(pv1, genome_1)
        pv1.experiment_number = 10001
        pv1.individual_number = 1
        pv1.device = 0

        pv2 = ProjectVariable(debug_mode)
        pv2 = apply_same_settings(pv2)
        pv2 = apply_unique_settings(pv2, genome_2)
        pv2.experiment_number = 10002
        pv2.individual_number = 2
        pv2.device = 1

        pv3 = ProjectVariable(debug_mode)
        pv3 = apply_same_settings(pv3)
        pv3 = apply_unique_settings(pv3, genome_3)
        pv3.experiment_number = 10003
        pv3.individual_number = 3
        pv3.device = 2

        pool = Pool(processes=3)
        results = pool.map(main_file.run, [pv1, pv2, pv3])

        # results = [(1, False, 0.0, 0.0),
        #            (2, False, 0.0, 0.037037037037037035),
        #            (3, False, 0.0, 0.07407407407407407)]

        pool.close()
        pool.join()

        col, val, train = process_results(results)
        results = col, val, train


        for k in col.keys():
            # keys are 1, 2, 3
            if k == '1':
                pv = pv1
            elif k == '2':
                pv = pv2
            else:
                pv = pv3

            line = '%d;%d;%s;%f;%f;%f;%d;%s;%s;%s;%s;%s;%s;%d;%s;%d\n' % \
                   (gen, pv.individual_number, str(col[k]), val[k], train[k], pv.learning_rate, pv.genome['num_conv_layers'],
                    str(pv.num_out_channels), str(pv.genome['kernel_size_per_layer']), str(pv.genome['padding']), 
                    str(pv.genome['conv_layer_type']), pv.genome['pooling_after_conv'], pv.genome['pooling_final'], 
                    pv.genome['fc_layer'], str(pv.genome['architecture_order']), pv.genome['in_features'])

            with open(genetic_search_path, 'a') as my_file:
                my_file.write(line)

        del pv1, pv2, pv3


evolutionary_search()


def debug_model_single(debug_mode=True):
    # THIS WORKS

    in_features_1 = 540
    genotype_1 = (3e-4, 2, [12, 18], [5, 5], [0, 0], [0, 0], 0, 1, 600, [0, 1, 0, 1, 2, 2], in_features_1)
    genome_1 = GO.write_genome(genotype_1)

    pv1 = ProjectVariable(debug_mode)
    pv1 = apply_same_settings(pv1)
    pv1 = apply_unique_settings(pv1, genome_1)
    pv1.experiment_number = 1000134978789
    pv1.device = 1

    results = main_file.run(pv1)

    print('asdf')


def debug_model_parallel(debug_mode=True):
    in_features_1 = 540
    #               0   1   2        3        4       5      6  7  8    9                    10
    genotype_1 = (3e-4, 2, [12, 18], [5, 5], [0, 0], [0, 0], 0, 1, 600, [0, 1, 0, 1, 2, 2], in_features_1)
    genome_1 = GO.write_genome(genotype_1)

    in_features_2 = 336
    genotype_2 = (
    3e-4, 3, [16, 32, 32], [3, 5, 5], [0, 0, 0], [0, 0, 0], 0, 1, 496, [0, 1, 0, 0, 1, 2, 2], in_features_2)
    genome_2 = GO.write_genome(genotype_2)

    in_features_3 = 182
    genotype_3 = (
    3e-4, 3, [16, 32, 48], [5, 5, 5], [0, 0, 0], [0, 0, 0], 0, 1, 256, [0, 1, 0, 0, 1, 2, 2], in_features_3)
    genome_3 = GO.write_genome(genotype_3)

    pv1 = ProjectVariable(debug_mode)
    pv1 = apply_same_settings(pv1)
    pv1 = apply_unique_settings(pv1, genome_1)
    pv1.experiment_number = 10001
    pv1.individual_number = 1
    pv1.device = 0

    pv2 = ProjectVariable(debug_mode)
    pv2 = apply_same_settings(pv2)
    pv2 = apply_unique_settings(pv2, genome_2)
    pv2.experiment_number = 10002
    pv2.individual_number = 2
    pv2.device = 1

    pv3 = ProjectVariable(debug_mode)
    pv3 = apply_same_settings(pv3)
    pv3 = apply_unique_settings(pv3, genome_3)
    pv3.experiment_number = 10003
    pv3.individual_number = 3
    pv3.device = 2

    pool = Pool(processes=3)
    results = pool.map(main_file.run, [pv1, pv2, pv3])

    col, val, train = process_results(results)

    print('asdf')


# debug_model_parallel()
