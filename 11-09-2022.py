import numpy as np
import matplotlib.pyplot as plt
import digital_phase_correction as dpc
import phase_correction as pc
import clipboard_and_style_sheet

path = r"D:\Microscope\11-09-2022/"
# data = np.load(path + "stage1_5116_stage2_8500_53856x74180.npy") # bckgnd
data = np.load(path + "stage1_5300_stage2_8970_53856x74180.npy")  # absorption on SU-8 bar
N_ifgs = len(data) - 1
ppifg = len(data[0])
center = ppifg // 2

data = data.flatten()
data = data[center:-center]
data.resize((N_ifgs, ppifg))

mean = np.mean(data, axis=1)
data = (data.T - mean).T

ll_freq = .1155
ul_freq = 0.17
pdiff = dpc.get_pdiff(data, ll_freq, ul_freq, 200)

h = 0
step = 250
while h < len(data):
    dpc.apply_t0_and_phi0_shift(pdiff[h:h + step], data[h:h + step])
    h += step
    print(len(data) - h)
