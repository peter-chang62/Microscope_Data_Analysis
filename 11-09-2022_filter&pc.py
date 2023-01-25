import sys

sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import td_phase_correct as td
from tqdm import tqdm
import os

cr.style_sheet()


def filt(freq, ll, ul, x, type="bp"):
    if type == "bp":
        return np.where(np.logical_and(freq > ll, freq < ul), x, 0.0)

    elif type == "notch":
        return np.where(np.logical_or(freq < ll, freq > ul), x, 0.0)


if os.name == 'posix':
    path = "/Users/peterchang/SynologyDrive/Research_Projects/" \
           "Microscope/FreeRunningSpectra/11-09-2022/"
else:
    path = r"C:/Users/pchan/SynologyDrive/Research_Projects/Microscope" \
           r"/FreeRunningSpectra/11-09-2022/"

data = np.load(
    path + "stage1_5300_stage2_8970_53856x74180.npy",
    mmap_mode='r')

ppifg = len(data[0])
center = ppifg // 2
N_IFG = len(data)

data.resize((ppifg * N_IFG,))
data = data[center:-center]
data.resize((N_IFG - 1, ppifg))
rfreq = np.fft.rfftfreq(len(data[0]), 1e-9) * 1e-6  # 0 -> 500 MHz

# %% __________________________________________________________________________
# filter the data first (using the 1 GHz resolution)
data_filt = np.load("data/phase_corrected/" +
                    "stage1_5300_stage2_8970_53856x74180_phase_corrected.npy",
                    mmap_mode='r+')

# taken from looking at fft of one interferogram
# list_filter = np.array([[0, 2],
#                         [249.5, 250.5],
#                         [280, 284],
#                         [337, 338],
#                         [436, 437],
#                         [499, 500]])

# taken from looking at fft of average of all full interferograms (1 ghz res)
# list_filter = np.array([[0, 1],
#                         [31.5, 32.5],
#                         [249.5, 250.5],
#                         [281, 283],
#                         [336, 339.5],
#                         [499, 500]])

# together
list_filter = np.array([[0, 2],
                        [31, 32.8],
                        [249.5, 250.5],
                        [280, 284],
                        [337, 338],
                        [436, 437],
                        [499, 500]])

step = len(data) // 8
chunks = np.arange(0, len(data), step)
chunks[-1] = len(data)
start = chunks[:-1]
end = chunks[1:]
console = 7

h = 0
for n in tqdm(range(start[console], end[console])):
    ft = np.fft.rfft(data[n])  # make sure to load from data

    for f in list_filter:
        ft = filt(rfreq, *f, ft, "notch")

    t_filt = np.fft.irfft(ft)
    data_filt[n] = t_filt
    # print(end[console] - start[console] - h - 1) # using tqdm now
    h += 1

# %% __________________________________________________________________________
# now phase correct the filtered data (1 GHz resolution)
opt = td.Optimize(data_filt[:, center - 50:center + 50])
opt.phase_correct(data_filt,
                  start_index=start[console],
                  end_index=end[console],
                  method='Powell')
