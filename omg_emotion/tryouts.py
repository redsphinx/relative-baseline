import pickle
import bz2
import torch
import os
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plot_results():

    fig = plt.figure()

    # xs = np.array([254058,	263906,	300264,	318662,	346470,	375668,	392676,	434924,	438882,	485088,	496430,	560186])
    # 20
    # series1 = np.array([0.4555, None, 0.5115, None, 0.4455, None, 0.513, 0.391, None, None, 0.429, None]).astype(np.double)
    # series2 = np.array([None, 0.555, None, 0.586, None, 0.6375, None, None, 0.4475, 0.4675, None, 0.428]).astype(np.double)
    # num = 10
    # series1 = np.array([None,	0.3575,	None,	0.351833333333333,	None,	0.400166666666667,	None,	0.383166666666667,	None,	None,	0.402,	0.4213333333]).astype(np.double)
    # series2 = np.array([0.391333333333333,	None,	0.384166666666667,	None,	0.405333333333333,	None,	0.410833333333333,	None,	0.417,	0.395,	None,	None]).astype(np.double)
    # num = 20
    # series1 = np.array([None,	0.464166666666667,	None,	0.4575,	None,	0.511166666666667,	None,	0.476833333333333,	None,	None,	0.504,	0.4775]).astype(np.double)
    # series2 = np.array([0.508,	None,	0.519333333333333,	None,	0.520833333333333,	None,	0.538666666666667,	None,	0.502833333333333,	0.529,	None,	None]).astype(np.double)
    # num = 30
    # series1 = np.array([None,	0.533666666666667,	None,	0.5275,	None,	0.5525,	None,	0.5345,	None,	None,	0.551666666666667,	0.5745]).astype(np.double)
    # series2 = np.array([0.559333333333333,	None,	0.547166666666667,	None,	0.594166666666667,	None,	0.577166666666667,	None,	0.569333333333333,	0.567333333333333,	None,	None]).astype(np.double)
    # num = 40
    # series1 = np.array([None,	0.717333333333333,	None,	0.721166666666667,	None,	0.719166666666667,	None,	0.718333333333333,	None,	None,	0.718,	0.7091666667]).astype(np.double)
    # series2 = np.array([0.741833333333333,	None,	0.718833333333333,	None,	0.759,	None,	0.7225,	None,	0.7425,	0.720166666666667,	None,	None]).astype(np.double)
    # num = 50
    # series1 = np.array([None,	0.709666666666667,	None,	0.743333333333333,	None,	0.7225,	None,	0.726166666666667,	None,	None,	0.731,	0.7276666667]).astype(np.double)
    # series2 = np.array([0.755833333333334,	None,	0.733,	None,	0.731166666666667,	None,	0.761833333333333,	None,	0.753,	0.735,	None,	None]).astype(np.double)
    # num = 100
    # series1 = np.array([None,	0.873166666666667,	None,	0.889333333333333,	None,	0.897666666666667,	None,	0.9015,	None,	None,	0.906,	0.8965]).astype(np.double)
    # series2 = np.array([0.8815,	None,	0.896666666666667,	None,	0.882666666666667,	None,	0.898,	None,	0.899666666666667,	0.898833333333334,	None,	None]).astype(np.double)
    # num = 500
    # series1 = np.array([None,	0.972,	None,	0.976166666666667,	None,	0.974666666666667,	None,	0.9805,	None,	None,	0.981333333333333,	0.981]).astype(np.double)
    # series2 = np.array([0.9695,	None,	0.9595,	None,	0.968,	None,	0.9705,	None,	0.971,	0.972166666666667,	None,	None]).astype(np.double)
    # num = 1000
    # series1 = np.array([None,	0.983,	None,	0.989333333333333,	None,	0.989333333333333,	None,	0.991166666666667,	None,	None,	0.991666666666667,	0.9921666667]).astype(np.double)
    # series2 = np.array([0.977333333333333,	None,	0.9785,	None,	0.975666666666667,	None,	0.983,	None,	0.9825,	0.987,	None,	None]).astype(np.double)

    xs = np.array([1213078,	1260646,	1851362,	1925942,	2623742,	2731382,	3530218,	3676966,	4570790,	4762694,	5745458,	5988566,	7054222,	7354582])
    # num = 6
    # series1 = np.array([np.nan,	0.208045977011494,	np.nan,	0.230316091954023,	np.nan,	0.227011494252874,	np.nan,	0.229310344827586,	np.nan,	0.245833333333333,	np.nan,	0.229885057471264,	np.nan,	0.2326149425])
    # series2 = np.array([0.22183908045977,	np.nan,	0.219827586206897,	np.nan,	0.222844827586207,	np.nan,	0.229022988505747,	np.nan,	0.225862068965517,	np.nan,	0.217385057471264,	np.nan,	0.220689655172414,	np.nan])

    # err1 = np.array([np.nan,	0.0520575144090395,	np.nan,	0.0612684178967938,	np.nan,	0.0578543268376306,	np.nan,	0.0463624290945777,	np.nan,	0.0597516342450405,	np.nan,	0.0467075580994916,	np.nan,	0.05159806634]) #/ 2
    # series1_min = series1 - err1
    # series1_max = series1 + err1
    # err2 = np.array([0.0488514198709851,	np.nan,	0.0554679584211911,	np.nan,	0.0515219948490961,	np.nan,	0.0613396380464259,	np.nan,	0.0439091367139001,	np.nan,	0.0542793577598474,	np.nan,	0.0619591141575624,	np.nan]) #/ 2
    # series2_min = series2 - err2
    # series2_max = series2 + err2
    # for i in range(len(series1_min)):
    #     if str(series1_min[i]) == 'nan':
    #         series1_min[i] = np.nan
    #     if str(series1_max[i]) == 'nan':
    #         series1_max[i] = np.nan
    #     if str(series2_min[i]) == 'nan':
    #         series2_min[i] = np.nan
    #     if str(series2_max[i]) == 'nan':
    #         series2_max[i] = np.nan
    
    # num = 66
    # series1 = np.array([np.nan,	0.551580459770115,	np.nan,	0.566954022988506,	np.nan,	0.557327586206897,	np.nan,	0.55301724137931,	np.nan,	0.57801724137931,	np.nan,	0.556896551724138,	np.nan,	0.5564655172])
    # series2 = np.array([0.330890804597701,	np.nan,	0.324856321839081,	np.nan,	0.350862068965517,	np.nan,	0.346264367816092,	np.nan,	0.349281609195402,	np.nan,	0.344827586206897,	np.nan,	0.358764367816092,	np.nan])
    # num = 126
    # series1 = np.array([np.nan,	0.639367816091954,	np.nan,	0.63448275862069,	np.nan,	0.63448275862069,	np.nan,	0.63448275862069,	np.nan,	0.645545977011494,	np.nan,	0.642528735632184,	np.nan,	0.6337643678])
    # series2 = np.array([0.375862068965517,	np.nan,	0.374137931034483,	np.nan,	0.372557471264368,	np.nan,	0.403591954022989,	np.nan,	0.393103448275862,	np.nan,	0.381896551724138,	np.nan,	0.391235632183908,	np.nan])
    num = 191
    series1 = np.array([np.nan,	0.688505747126437,	np.nan,	0.686781609195402,	np.nan,	0.695258620689655,	np.nan,	0.685919540229885,	np.nan,	0.68448275862069,	np.nan,	0.685201149425287,	np.nan,	0.6813218391])
    series2 = np.array([0.404166666666667,	np.nan,	0.398275862068966,	np.nan,	0.410252463054187,	np.nan,	0.404885057471264,	np.nan,	0.426005747126437,	np.nan,	0.420977011494253,	np.nan,	0.416954022988506,	np.nan])

    s1mask = np.isfinite(series1)
    s2mask = np.isfinite(series2)

    plt.plot(xs[s1mask], series1[s1mask], linestyle='-', marker='o', label='3DConv')
    plt.plot(xs[s2mask], series2[s2mask], linestyle='-', marker='o', label='3DConvTTN')
    # plt.fill_between(xs[s1mask], series1_max[s1mask], series1_min[s1mask], alpha=0.3)
    # plt.fill_between(xs[s2mask], series2_max[s2mask], series2_min[s2mask], alpha=0.3)
    
    

    # plt.errorbar(xs[s1mask], series1[s1mask], yerr=err1[s1mask], linestyle='-', marker='o', label='3DConv')
    # plt.errorbar(xs[s2mask], series2[s2mask], yerr=err2[s2mask], linestyle='-', marker='o', label='3DConvTTN')

    plt.ylabel('accuracy')
    plt.xlabel('parameters')
    plt.title('%d datapoints' % num)
    plt.legend(('3DConv', '3DConvTTN'))

    plt.savefig('picture_%d.jpg' % num)


plot_results()


def C3D_experiments():
    def auto_in_features(input_shape, type, params):
        t, h, w = input_shape

        assert type in ['conv', 'pool']

        if type == 'conv':
            k_t, k_h, pad, stride = params
            t = (t + 2 * pad - k_t) / stride + 1
            h = (h + 2 * pad - k_h) / stride + 1
        elif type == 'pool':
            k_t, k_h = params
            t = int(np.floor(t / k_t))
            h = int(np.floor(h / k_h))

        w = h
        return t, h, w

    conv_k_t = [3, 5, 7, 9, 11]

    max_pool_temp = [1]

    conv_k_hw = [3]

    ip = [30, 100]

    iputshep = (ip[1], 60, 60)

    print(iputshep)
    t, h, w = auto_in_features(iputshep, 'conv', (conv_k_t[4], conv_k_hw[0], 0, 1))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp[0], 2))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'conv', (conv_k_t[4]*2, conv_k_hw[0], 0, 1))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp[0], 2))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'conv', (conv_k_t[4]*3, conv_k_hw[0], 0, 1))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'conv', (conv_k_t[4]*3, conv_k_hw[0], 0, 1))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'pool', (max_pool_temp[0], 2))
    print(t, h, w)
    in_features = t * h * w * 64
    print(in_features)


C3D_experiments()


