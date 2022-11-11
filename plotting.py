import numpy as np
import clipboard_and_style_sheet as cr
import matplotlib.pyplot as plt
from scipy.integrate import simps
from images import *
import os

if os.name == 'posix':
    path = r"/home/peterchang/SynologyDrive/Research_Projects/Microscope/CLEO_2023/data_to_plot/"

cr.style_sheet()

x_group3, y_group3, i_group3, s_group3 = group_3(False)
x_num4, y_num4, i_num4, s_num4 = num_4(False)
x_bar, y_bar, i_bar, s_bar = smallest_bar(False)

ind_inbw_patt = [57, 35]
ind_inbw_bar = [30, 27]
ind_on_bar = [24, 19]

snr_bckgnd = np.load(path + "snr_bckgnd.npy")
snr_su8 = np.load(path + "snr_su8.npy")
avg_bckgnd = np.load(path + "avg_bckgnd.npy")
avg_su8 = np.load(path + "avg_su8.npy")
