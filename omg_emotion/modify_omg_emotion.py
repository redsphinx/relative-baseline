import os
import numpy as np
from relative_baseline.omg_emotion import project_paths as PP


def check_for_data_overlap():

    path = PP.data_path
    which = ['Training', 'Test', 'Validation']

    # train, val, test
    uniques = [[], [], []]

    for w in which:
        annotation_path = os.path.join(path, w, 'Annotations/annotations.csv')

        all_annotations = np.genfromtxt(annotation_path, delimiter=',', dtype=str)

        if w == 'Training':
            num = 0
        elif w == 'Test':
            num = 1
        elif w == 'Validation':
            num = 2
        else:
            print('invalid which')

        for i in range(len(all_annotations)):
            tmp_1 = str(all_annotations[i][0])
            tmp_2 = str(all_annotations[i][1])
            tmp = tmp_1 + tmp_2
            uniques[num].append(tmp)

        uniques[num] = set(uniques[num])

    print('overlap train with test: ')
    overlap = uniques[0].intersection(uniques[2])
    if len(overlap) > 0:
        print('OVERLAP!!! :(')
        print(overlap)
    else:
        print('no overlap')

    print('overlap train with val: ')
    overlap = uniques[0].intersection(uniques[1])
    if len(overlap) > 0:
        print('OVERLAP!!! :(')
        print(overlap)
    else:
        print('no overlap')

    print('overlap val with test: ')
    overlap = uniques[1].intersection(uniques[2])
    if len(overlap) > 0:
        print('OVERLAP!!! :(')
        print(overlap)
    else:
        print('no overlap')


