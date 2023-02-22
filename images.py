import numpy as np
import clipboard_and_style_sheet as cr
import matplotlib.pyplot as plt
from scipy.integrate import simps
import os

if os.name == 'nt':
    path_folder = r"C:\\Users\\pchan\\SynologyDrive\\Research_Projects" \
                  r"\\Microscope\\Images/"

else:
    # path_folder = r"/Users/peterchang/SynologyDrive/Research_Projects" \
    #               r"/Microscope/Images/"
    path_folder = r"/Volumes/Extreme SSD/Research_Projects/Microscope/Images/"

cr.style_sheet()


def group_3(plot=True):
    path = path_folder + "11-07-2022/"
    s1 = np.load(
        path + "stage1_5120_6480_stage2_8450_8800_step_10_ppifg_74180.npy")
    s2 = np.load(
        path + "stage1_5120_6480_stage2_8810_9060_step_10_ppifg_74180.npy")
    s = np.hstack((s1, s2))
    s = np.transpose(s, axes=[1, 0, 2])
    i = simps(s, axis=-1)
    x = np.arange(5120, 6480 + 10, 10)
    y = np.arange(8450, 9060 + 10, 10)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=np.array([11.2, 6.61]))
        ax.pcolormesh(x, y, i, cmap='gist_heat')
        cr.square()
    return x, y, i, s


def num_4(plot=True):
    path = path_folder + "11-08-2022/"
    s = np.load(
        path + "stage1_5932_6066_stage2_8478_8575p5_step_2p5_ppifg_74180.npy")
    s = np.transpose(s, axes=[1, 0, 2])
    i = simps(s, axis=-1)
    x = np.arange(5932, 6066, 2.5)
    y = np.arange(8478, 8575.5 + 2.5, 2.5)

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.pcolormesh(x, y, i, cmap='gist_heat')
        cr.square()
    return x, y, i, s


def smallest_bar(plot=True):
    path = path_folder + "11-08-2022/"
    s = np.load(
        path + "stage1_6274_6460_stage2_8593_8883_step_2p5_ppifg_74180.npy")
    s = np.transpose(s, axes=[1, 0, 2])
    i = simps(s, axis=-1)
    x = np.arange(6274, 6460, 2.5)
    y = np.arange(8593, 8883 + 2.5, 2.5)

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.pcolormesh(x, y, i, cmap='gist_heat')
        cr.square()
    return x, y, i, s

# x_group3, y_group3, i_group3, s_group3 = group_3(False)
# x_num4, y_num4, i_num4, s_num4 = num_4(False)
# x_bar, y_bar, i_bar, s_bar = smallest_bar(False)
