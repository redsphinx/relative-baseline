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
import skvideo.io
from PIL import Image


def plot_results():

    fig = plt.figure()

    xs = np.array([254058,	263906,	300264,	318662,	346470,	375668,	392676,	434924,	438882,	485088,	496430,	531294, 560186])
    # 20
    # series1 = np.array([0.4555, None, 0.5115, None, 0.4455, None, 0.513, 0.391, None, None, 0.429, None]).astype(np.double)
    # series2 = np.array([None, 0.555, None, 0.586, None, 0.6375, None, None, 0.4475, 0.4675, None, 0.428]).astype(np.double)

    # num = 10
    # series1 = np.array([np.nan,	0.3575,	np.nan,	0.351833333333333,	np.nan,	0.400166666666667,	np.nan,	0.383166666666667,	np.nan,	np.nan,	0.402,	np.nan,	0.4213333333])
    # series2 = np.array([0.409833333333333,	np.nan,	0.428666666666667,	np.nan,	0.473833333333333,	np.nan,	0.472833333333333,	np.nan,	0.469333333333333,	0.427,	np.nan,	0.4355,	np.nan])
    # series1 = np.array([None,	0.3575,	None,	0.351833333333333,	None,	0.400166666666667,	None,	0.383166666666667,	None,	None,	0.402,	0.4213333333]).astype(np.double)
    # series2 = np.array([0.391333333333333,	None,	0.384166666666667,	None,	0.405333333333333,	None,	0.410833333333333,	None,	0.417,	0.395,	None,	None]).astype(np.double)

    # num = 20
    # series1 = np.array([np.nan,	0.464166666666667,	np.nan,	0.4575,	np.nan,	0.511166666666667,	np.nan,	0.476833333333333,	np.nan,	np.nan,	0.504,	np.nan,	0.4775])
    # series2 = np.array([0.535,	np.nan,	0.560333333333334,	np.nan,	0.562,	np.nan,	0.567,	np.nan,	0.574333333333333,	0.5975,	np.nan,	0.597666666666667,	np.nan])
    # series1 = np.array([None,	0.464166666666667,	None,	0.4575,	None,	0.511166666666667,	None,	0.476833333333333,	None,	None,	0.504,	0.4775]).astype(np.double)
    # series2 = np.array([0.508,	None,	0.519333333333333,	None,	0.520833333333333,	None,	0.538666666666667,	None,	0.502833333333333,	0.529,	None,	None]).astype(np.double)

    # num = 30
    # series1 = np.array([np.nan,	0.533666666666667,	np.nan,	0.5275,	np.nan,	0.5525,	np.nan,	0.5345,	np.nan,	np.nan,	0.551666666666667,	np.nan,	0.5745])
    # series2 = np.array([0.662166666666667,	np.nan,	0.659833333333333,	np.nan,	0.666166666666667,	np.nan,	0.663666666666667,	np.nan,	0.694,	0.656166666666667,	np.nan,	0.6045,	np.nan])
    # series1 = np.array([None,	0.533666666666667,	None,	0.5275,	None,	0.5525,	None,	0.5345,	None,	None,	0.551666666666667,	0.5745]).astype(np.double)
    # series2 = np.array([0.559333333333333,	None,	0.547166666666667,	None,	0.594166666666667,	None,	0.577166666666667,	None,	0.569333333333333,	0.567333333333333,	None,	None]).astype(np.double)

    # num = 40
    # series1 = np.array([np.nan,	0.717333333333333,	np.nan,	0.721166666666667,	np.nan,	0.719166666666667,	np.nan,	0.718333333333333,	np.nan,	np.nan,	0.718,	np.nan,	0.7091666667])
    # series2 = np.array([0.712333333333333,	np.nan,	0.7065,	np.nan,	0.705166666666667,	np.nan,	0.722,	np.nan,	0.679666666666667,	0.650166666666667,	np.nan,	0.642166666666667,	np.nan])
    # series1 = np.array([None,	0.717333333333333,	None,	0.721166666666667,	None,	0.719166666666667,	None,	0.718333333333333,	None,	None,	0.718,	0.7091666667]).astype(np.double)
    # series2 = np.array([0.741833333333333,	None,	0.718833333333333,	None,	0.759,	None,	0.7225,	None,	0.7425,	0.720166666666667,	None,	None]).astype(np.double)

    # num = 50
    # series1 = np.array([np.nan,	0.709666666666667,	np.nan,	0.743333333333333,	np.nan,	0.7225,	np.nan,	0.726166666666667,	np.nan,	np.nan,	0.731,	np.nan,	0.7276666667])
    # series2 = np.array([0.749333333333333,	np.nan,	0.722166666666667,	np.nan,	0.720666666666667,	np.nan,	0.723666666666667,	np.nan,	0.727,	0.713833333333333,	np.nan,	0.6415,	np.nan])
    # series1 = np.array([None,	0.709666666666667,	None,	0.743333333333333,	None,	0.7225,	None,	0.726166666666667,	None,	None,	0.731,	0.7276666667]).astype(np.double)
    # series2 = np.array([0.755833333333334,	None,	0.733,	None,	0.731166666666667,	None,	0.761833333333333,	None,	0.753,	0.735,	None,	None]).astype(np.double)

    # num = 100
    # series1 = np.array([np.nan,	0.873166666666667,	np.nan,	0.889333333333333,	np.nan,	0.897666666666667,	np.nan,	0.9015,	np.nan,	np.nan,	0.906,	np.nan,	0.8965])
    # series2 = np.array([0.851,	np.nan,	0.852166666666667,	np.nan,	0.799,	np.nan,	0.804,	np.nan,	0.783166666666667,	0.818666666666667,	np.nan,	0.685333333333333,	np.nan])
    # series1 = np.array([None,	0.873166666666667,	None,	0.889333333333333,	None,	0.897666666666667,	None,	0.9015,	None,	None,	0.906,	0.8965]).astype(np.double)
    # series2 = np.array([0.8815,	None,	0.896666666666667,	None,	0.882666666666667,	None,	0.898,	None,	0.899666666666667,	0.898833333333334,	None,	None]).astype(np.double)

    # num = 500
    # series1 = np.array([np.nan,	0.972,	np.nan,	0.976166666666667,	np.nan,	0.974666666666667,	np.nan,	0.9805,	np.nan,	np.nan,	0.981333333333333,	np.nan,	0.981])
    # series2 = np.array([0.949666666666667,	np.nan,	0.9535,	np.nan,	0.9415,	np.nan,	0.884,	np.nan,	0.901,	0.7945,	np.nan,	0.743,	np.nan])
    # series1 = np.array([None,	0.972,	None,	0.976166666666667,	None,	0.974666666666667,	None,	0.9805,	None,	None,	0.981333333333333,	0.981]).astype(np.double)
    # series2 = np.array([0.9695,	None,	0.9595,	None,	0.968,	None,	0.9705,	None,	0.971,	0.972166666666667,	None,	None]).astype(np.double)

    # num = 1000
    # series1 = np.array([np.nan,	0.983,	np.nan,	0.989333333333333,	np.nan,	0.989333333333333,	np.nan,	0.991166666666667,	np.nan,	np.nan,	0.991666666666667,	np.nan,	0.9921666667])
    # series2 = np.array([0.975,	np.nan,	0.946833333333333,	np.nan,	0.877666666666667,	np.nan,	0.901166666666667,	np.nan,	0.835166666666667,	0.859166666666667,	np.nan,	0.827333333333333,	np.nan])
    # series1 = np.array([None,	0.983,	None,	0.989333333333333,	None,	0.989333333333333,	None,	0.991166666666667,	None,	None,	0.991666666666667,	0.9921666667]).astype(np.double)
    # series2 = np.array([0.977333333333333,	None,	0.9785,	None,	0.975666666666667,	None,	0.983,	None,	0.9825,	0.987,	None,	None]).astype(np.double)

    # xs = np.array([1213078,	1260646,	1851362,	1925942,	2623742,	2731382,	3530218,	3676966,	4570790,	4762694,	5745458,	5988566,	7054222,	7354582])
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
    # num = 191
    # series1 = np.array([np.nan,	0.688505747126437,	np.nan,	0.686781609195402,	np.nan,	0.695258620689655,	np.nan,	0.685919540229885,	np.nan,	0.68448275862069,	np.nan,	0.685201149425287,	np.nan,	0.6813218391])
    # series2 = np.array([0.404166666666667,	np.nan,	0.398275862068966,	np.nan,	0.410252463054187,	np.nan,	0.404885057471264,	np.nan,	0.426005747126437,	np.nan,	0.420977011494253,	np.nan,	0.416954022988506,	np.nan])

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


# plot_results()


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


# C3D_experiments()


def plot_results_datapoints_accuracy():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # xs = np.array([10, 20, 30, 40, 50, 100, 500, 1000, 2000])

    # xs = np.array([10, 20, 30, 40, 50, 100, 500, 1000, 2000, 5000])
    # xs = np.array([1,2,3,4,5,10,50,100,200,5000])
    xs = np.array([1,2,3,4,5,6,7,8,9,10])
    xs_2 = ['1', '2', '3', '4', '5', '10', '50', '100', '200', '500']
    # xs = np.array([1, 2, 3, 4, 5, 10, 50, 100, 200, 500])
    # xs = 10.0**np.linspace(xs)


    model_2_acc = np.array(
        [0.3737833333, 0.4982566667, 0.6247766667, 0.6710766667, 0.7495833333, 0.8366866667, 0.9592766667, 0.97637, 0.98757, 0.99439])
    labels_2 = [0.374, 0.498, 0.625, 0.671, 0.75, 0.837, 0.96, 0.977, 0.988, 0.994]
    loc_2_y = [-0.03, -0.04, -0.04, -0.04, -0.04, 0.02, 0.01, 0.01, 0.01, 0.01]
    loc_2_x = [1, 0, 0, 0, 0, -10, -50, -250, -400, -700]
    model_2_stde = np.array([0.0122180041  ,0.01310738272 ,0.01074435116 ,0.009515661574,0.008344632451,0.005919209282,0.002494984673,0.001364422141, 0.0007161649407, 0.0002911414318])
    # model_2_stde = np.array([0.06692076451,0.07179209187,0.05884923496,0.05211942494,0.04570543427,0.03242084446,0.01366559386,0.007473247844])

    model_3_acc = np.array(
        [0.47685, 0.58781, 0.6681366667, 0.7206133333, 0.7725033333, 0.8287233333, 0.9254366667, 0.9677166667, 0.9718333333, 0.98799])
    labels_3 = [0.477, 0.588, 0.669, 0.721, 0.773, 0.829, 0.925, 0.968, 0.972, 0.988]
    loc_3_y = [0.03, 0.03, 0.015, 0.015, 0.02, -0.05, -0.04, -0.04, -0.03, -0.03]
    loc_3_x = [-2, -5, -9, -15, -10, -10, -50, -200, -350, -750]
    model_3_stde = np.array([0.01387213494 ,0.01078555964 ,0.009109147397,0.007932831543,0.006682841184,0.007343126315,0.005807509474,0.002008161588, 0.00220658778, 0.001000559288])
    # model_3_stde = np.array([0.07598081227,0.05907494308,0.04989285509,0.04344990781,0.03660342865,0.04021995925,0.03180903942,0.01099915401])

    plt.errorbar(xs, model_2_acc, yerr=model_2_stde, linestyle='--', color='g', marker='o', markersize=4, elinewidth=1, barsabove=True, ecolor='black', capsize=3, label='LeNet-5-3DConv')
    # for k in range(len(model_2_acc)):
    #     i = xs[k]
    #     t = i + loc_2_x[k]
    #     j = labels_2[k]
    #     s = j + loc_2_y[k]
    #     ax.annotate(str(j), xy=(t, s), color='g')

    plt.errorbar(xs, model_3_acc, yerr=model_3_stde,  linestyle='-', color='r', marker='o', markersize=4, elinewidth=1, barsabove=True, ecolor='black', capsize=3, label='LeNet-5-3DConvTTN')
    # for k in range(len(model_3_acc)):
    #     i = xs[k]
    #     t = i + loc_3_x[k]
    #     j = labels_3[k]
    #     s = j + loc_3_y[k]
    #     ax.annotate(str(j), xy=(t, s), color='r')

    # --------
    # not used
    # star = [9.8, 19, 29, 37, 47, 490, 950, 1900]
    # y_loc_star = [0.53, 0.64, 0.71, 0.75, 0.81, 0.86, 0.90, 0.92]
    # for k in range(len(star)):
    #     i = star[k]
    #     ax.annotate(str('*'), xy=(i, y_loc_star[k]), color='b')
    # --------

    # star_loc_x = [9.5, 18, 27, 36, 46, 450, 900, 1800, 4500]
    # star_loc_y = [0.54, 0.64, 0.70, 0.75, 0.82, 0.86, 0.90, 0.92, 0.93]
    #
    # for k in range(len(star_loc_x)):
    #     ax.annotate(str(' * '), xy=(star_loc_x[k], star_loc_y[k]), color='b')

    # plt.xscale('log')
    plt.grid(True)

    plt.xticks(xs, xs_2)


    plt.ylabel('Accuracy')
    # plt.xlabel('Training videos')
    # plt.title('Training Videos vs. Accuracy')
    plt.xlabel('Samples per class')
    plt.title('Samples per class vs. Accuracy')
    plt.legend(('LeNet-5-3DConv', 'LeNet-5-3DConvTTN'))

    plt.savefig('presentation_1.jpg', format='jpg')
    # plt.savefig('picture_666_12.eps', format='eps')


plot_results_datapoints_accuracy()


'''
375,668
496,430
496,430
496,430
496,430
496,430
496,430
496,430
496,430
496,430
'''


'''
import numpy as np
import scipy.stats as stats
m2_mean = [0.3737833333, 0.4982566667, 0.6247766667, 0.6710766667, 0.7495833333, 0.8366866667, 0.9592766667, 0.97637, 0.98757, 0.99439]
m2_std = [0.06692076451,0.07179209187,0.05884923496,0.05211942494,0.04570543427,0.03242084446,0.01366559386,0.007473247844, 0.003922596929, 0.001594647296]
m3_mean = [0.47685, 0.58781, 0.6681366667, 0.7206133333, 0.7725033333, 0.8287233333, 0.9254366667, 0.9677166667, 0.9718333333, 0.98799]
m3_std = [0.07598081227,0.05907494308,0.04989285509,0.04344990781,0.03660342865,0.04021995925,0.03180903942,0.01099915401, 0.01208597902, 0.005480288922]
nums = [10, 20, 30, 40, 50, 100, 500, 1000, 2000, 5000]
for i in range(len(nums)):
    print(nums[i])
    t, p = stats.ttest_ind_from_stats(m2_mean[i], m2_std[i], 30, m3_mean[i], m3_std[i], 30, False)
    if np.abs(t) > 2 and p < 0.05:
        print('result is significant')
    else:
        print('result is not significant')
    print('t = ', t, ' p = ', p)

10
result is significant
t =  -5.575524535942804  p =  7.022023942185273e-07
20
result is significant
t =  -5.275773906508184  p =  2.2187817134151176e-06
30
result is significant
t =  -3.078216237373812  p =  0.00321100439889691
40
result is significant
t =  -3.998565685300635  p =  0.0001879498764672029
50
result is significant
t =  -2.143898936230878  p =  0.0364484729844081
100
result is not significant
t =  0.8443083202512734  p =  0.4021219806985964
500
result is significant
t =  5.353779344764144  p =  3.982411409788355e-06
1000
result is significant
t =  3.564224702921899  p =  0.0008020691407754687
2000
result is significant
t =  6.7833472994078745  p =  7.242109800359888e-08
5000
result is significant
t =  6.141700191847464  p =  5.725557998814051e-07
'''


def calc_architecture_viability(t, h, w, layer_type, k, p, s, div):
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


def run_calc_av():
    t, h, w = 30, 50, 75
    t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=1, s=1, div=None)
    t, h, w = calc_architecture_viability(t, h, w, 'pool', k=None, p=None, s=None, div=2)
    t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=1, s=1, div=None)
    t, h, w = calc_architecture_viability(t, h, w, 'pool', k=None, p=None, s=None, div=2)
    t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=1, s=1, div=None)
    t, h, w = calc_architecture_viability(t, h, w, 'pool', k=None, p=None, s=None, div=2)


def run_calc_av_1():
    t, h, w = 30, 50, 75
    t, h, w = calc_architecture_viability(t, h, w, 'conv', k=3, p=1, s=1, div=None)
    t, h, w = calc_architecture_viability(t, h, w, 'pool', k=None, p=None, s=None, div=2)
    t, h, w = calc_architecture_viability(t, h, w, 'conv', k=3, p=1, s=1, div=None)
    t, h, w = calc_architecture_viability(t, h, w, 'conv', k=3, p=1, s=1, div=None)

    t, h, w = calc_architecture_viability(t, h, w, 'conv', k=3, p=0, s=1, div=None)
    t, h, w = calc_architecture_viability(t, h, w, 'conv', k=3, p=0, s=1, div=None)
    t, h, w = calc_architecture_viability(t, h, w, 'conv', k=3, p=0, s=1, div=None)

    t, h, w = calc_architecture_viability(t, h, w, 'conv', k=3, p=0, s=1, div=None)
    t, h, w = calc_architecture_viability(t, h, w, 'conv', k=3, p=0, s=1, div=None)
    t, h, w = calc_architecture_viability(t, h, w, 'conv', k=3, p=0, s=1, div=None)
    # t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=0, s=1, div=None)
    # t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=0, s=1, div=None)
    # t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=0, s=1, div=None)


    # t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=1, s=1, div=None)
    # t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=1, s=1, div=None)
    # t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=1, s=1, div=None)
    # t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=1, s=1, div=None)
    # t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=1, s=1, div=None)
    # t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=1, s=1, div=None)
    # t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=1, s=1, div=None)
    # t, h, w = calc_architecture_viability(t, h, w, 'conv', k=5, p=1, s=1, div=None)

    t, h, w = calc_architecture_viability(t, h, w, 'pool', k=None, p=None, s=None, div=2)


# run_calc_av_1()


import os
from PIL import Image


def process_files(folder, new_folder=True, resize=None, start_from=1):
    file_path = os.path.join('/media/gabi/DATADRIVE1/garden/', folder)
    dest_path = file_path

    if new_folder:
        new_file_path = os.path.join('/media/gabi/DATADRIVE1/garden/', 'new_%s' % folder)
        if not os.path.exists(new_file_path):
            os.mkdir(new_file_path)
        dest_path = new_file_path

    frames = os.listdir(file_path)
    frames.sort()

    if resize is not None:
        _tmp = os.path.join(file_path, frames[0])
        _img = Image.open(_tmp)
        img_size = _img.size
        w = resize[0]
        h = resize[1]

    for ind, frame in enumerate(frames):

        frame_path = os.path.join(file_path, frame)

        img_frame = Image.open(frame_path)

        if resize is not None:
            img_frame = img_frame.resize((w, h), Image.ANTIALIAS)

        name = '%05d.jpg' % (start_from)
        name_path = os.path.join(dest_path, name)

        img_frame.save(name_path)
        start_from = start_from + 1


process_files('copy_rose_25-', resize=(2304, 1296))
