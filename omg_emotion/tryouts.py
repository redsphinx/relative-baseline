# from relative_baseline.omg_emotion import project_paths as PP
# from relative_baseline.omg_emotion import utils as U
# from relative_baseline.omg_emotion.settings import ProjectVariable
# from relative_baseline.omg_emotion import setup
#
#
# project_variable = ProjectVariable(debug_mode=True)
# project_variable.model_number = 3
# project_variable.num_out_channels = [6, 16]
#
# my_model = setup.get_model(project_variable)
#
# print('Loaded model number %d with %d trainable parameters' % (project_variable.model_number, U.count_parameters(my_model)))
#
# device = setup.get_device(project_variable)
#
# if project_variable.device is not None:
#     my_model.cuda(device)
#
# my_optimizer = setup.get_optimizer(project_variable, my_model)
#
# print('Loaded model number %d with %d trainable parameters' % (project_variable.model_number, U.count_parameters(my_model)))


# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
# ax = fig.gca(projection='3d')

xs = np.array([254234, 263906, 269884, 281908, 285584, 300160])

# 20
# series1 = np.array([0.4555, None, 0.5115, None, 0.4455, None]).astype(np.double)
# series2 = np.array([None, 0.555, None, 0.586, None, 0.6375]).astype(np.double)
# 30
# series1 = np.array([0.593, None, 0.509, None, 0.564, None]).astype(np.double)
# series2 = np.array([None, 0.641, None, 0.5885, None, 0.599]).astype(np.double)
# 40
# series1 = np.array([0.734, None, 0.6745, None, 0.6855, None]).astype(np.double)
# series2 = np.array([None, 0.753, None, 0.778, None, 0.7705]).astype(np.double)
# 50
# series1 = np.array([0.75, None, 0.7355, None, 0.744, None]).astype(np.double)
# series2 = np.array([None, 0.7725, None, 0.749, None, 0.7535]).astype(np.double)
# 100
# series1 = np.array([0.892, None, 0.903, None, 0.915, None]).astype(np.double)
# series2 = np.array([None, 0.9275, None, 0.9245, None, 0.936]).astype(np.double)
# 500
# series1 = np.array([0.972, None, 0.97, None, 0.973, None]).astype(np.double)
# series2 = np.array([None, 0.989, None, 0.995, None, 0.994]).astype(np.double)
# 1000
# series1 = np.array([0.98, None, 0.988, None, 0.9865, None]).astype(np.double)
# series2 = np.array([None, 0.998, None, 0.997, None, 0.995]).astype(np.double)
# 5000
series1 = np.array([0.9965, None, 0.9955, None, 0.995, None]).astype(np.double)
series2 = np.array([None, 0.999, None, 1, None, 1]).astype(np.double)

s1mask = np.isfinite(series1)
s2mask = np.isfinite(series2)



# series1 = np.array([1, 3, 3, None, None, 5, 8, 9]).astype(np.double)
# s1mask = np.isfinite(series1)
# series2 = np.array([2, None, 5, None, 4, None, 3, 2]).astype(np.double)
# s2mask = np.isfinite(series2)

plt.plot(xs[s1mask], series1[s1mask], linestyle='-', marker='o', label='m2')
plt.plot(xs[s2mask], series2[s2mask], linestyle='-', marker='o', label='m3')

num = 5000

plt.ylabel('accuracy')
plt.xlabel('parameters')
plt.title('%d datapoints' % num)
plt.legend(('m2', 'm3'))

plt.savefig('picture_%d.jpg' % num)


# # Make data.
# X = np.array([20, 30, 40, 50, 100, 500, 1000, 5000])
# Y = np.array([254234, 263906, 269884, 281908, 285584, 300160])
#
# X, Y = np.meshgrid(X, Y)
# # R = np.sqrt(X**2 + Y**2)
# # Z = np.sin(R)
# Z = np.array([[0.4555, 0.593, 0.734, 0.75, 0.892, 0.972, 0.98, 0.9965],
#               [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#               [0.5115, 0.509, 0.6745, 0.7355, 0.903, 0.97, 0.988, 0.9955],
#               [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#               [0.4455, 0.564, 0.6855, 0.744, 0.915, 0.973, 0.9865, 0.995],
#               [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
# z_mask = np.isfinite(Z)
#
#
# # Plot the surface.
# surf = ax.plot_surface(X[z_mask], Y, Z[z_mask], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()
# plt.savefig('picture.jpg')