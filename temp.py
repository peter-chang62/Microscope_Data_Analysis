import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import scipy.constants as sc
import os
import td_phase_correct as td

cr.style_sheet()

# % ___________________________________________________________________________
# h = 0
# step = 5
# n = 0
# while h < 100:
#     data = np.genfromtxt("data/KC/Data.txt",
#                          skip_header=h,  # start at hth row
#                          max_rows=step  # load step rows each time
#                          )
#     np.save("data/KC/temp/" + f'{n}.npy', data)  # save
#     n += 1  # increment file name by 1
#     h += step  # increment h by step
#     print(h)  # print current progress

# % ___________________________________________________________________________
# names = [i.name for i in os.scandir("data/KC/temp/")]
# [names.remove(i) for i in names if not (".npy" in i)]
# key = lambda s: float(s.split(".npy")[0])
#
# data = np.zeros((100, int(2e6)))
# step = 5
# for n, i in enumerate(names):
#     data[step * n:step * n + step] = np.load("data/KC/temp/" + i)

# % ___________________________________________________________________________
data = np.load("data/KC/Data.npy")
ppifg = len(data[0])
center = ppifg // 2

opt = td.Optimize(data[:, center - 412:center + 264])
opt.phase_correct(data)


def plot_apod(apod):
    avg = np.mean(data, 0)
    a = avg[center - apod // 2:center + apod // 2]
    ft = np.fft.rfft(a)
    freq = np.fft.rfftfreq(len(a), 1 / 100e6) * 2e7
    wl = sc.c * 1e9 / freq

    plt.plot(wl, abs(ft))
    plt.xlim(380, 395)
