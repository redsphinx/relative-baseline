from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plot_results():

    fig = plt.figure()

    #                                                               3        3      2        2      3         2

    xs = np.array([254058,	263906,	300264,	318662,	346470,	375668,	392676,	434924,	438882,	485088,	496430,	560186])

    # 20
    # series1 = np.array([0.4555, None, 0.5115, None, 0.4455, None, 0.513, 0.391, None, None, 0.429, None]).astype(np.double)
    # series2 = np.array([None, 0.555, None, 0.586, None, 0.6375, None, None, 0.4475, 0.4675, None, 0.428]).astype(np.double)


    num = 10
    series1 = np.array([None,	0.3575,	None,	0.351833333333333,	None,	0.400166666666667,	None,	0.383166666666667,	None,	None,	0.402,	0.4213333333]).astype(np.double)
    series2 = np.array([0.391333333333333,	None,	0.384166666666667,	None,	0.405333333333333,	None,	0.410833333333333,	None,	0.417,	0.395,	None,	None]).astype(np.double)
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

    s1mask = np.isfinite(series1)
    s2mask = np.isfinite(series2)

    plt.plot(xs[s1mask], series1[s1mask], linestyle='-', marker='o', label='3DConv')
    plt.plot(xs[s2mask], series2[s2mask], linestyle='-', marker='o', label='3DConvTTN')

    plt.ylabel('accuracy')
    plt.xlabel('parameters')
    plt.title('%d datapoints' % num)
    plt.legend(('3DConv', '3DConvTTN'))

    plt.savefig('picture_%d.jpg' % num)


def C3D_experiments():
    def auto_in_features(input_shape, type, params):
        t, h, w = input_shape

        assert type in ['conv', 'pool']

        if type == 'conv':
            k_t, k_h, k_w, pad = params
            t = t + 2 * pad - k_t + 1
            h = h + 2 * pad - k_h + 1
            w = w + 2 * pad - k_w + 1
        elif type == 'pool':
            k_t, k_h, k_w = params
            t = int(np.floor(t / k_t))
            h = int(np.floor(h / k_h))
            w = int(np.floor(w / k_w))
            if t == 0:
                t += 1
            if h == 0:
                h += 1
            if w == 0:
                w += 1

        return t, h, w

    iputshep = (300, 60, 60)
    print(iputshep)
    t, h, w = auto_in_features(iputshep, 'conv', (3, 3, 3, 0))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
    print(t, h, w)
    t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
    print(t, h, w)
    # t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
    # print(t, h, w)
    # t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
    # print(t, h, w)
    # t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
    # print(t, h, w)
    # t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
    # print(t, h, w)
    # t, h, w = auto_in_features((t, h, w), 'conv', (3, 3, 3, 0))
    # print(t, h, w)
    # t, h, w = auto_in_features((t, h, w), 'pool', (2, 2, 2))
    # print(t, h, w)
    in_features = t * h * w * 32
    print(in_features)


C3D_experiments()
