import numpy as np
import clipboard_and_style_sheet as cr
import matplotlib.pyplot as plt

path = r"/Users/peterchang/Resilio Sync/OvarianFTIR/"
data = np.fromfile(path + "D1", "<f")

data.shape = (394, 1280, 1280)
wnum = np.genfromtxt(path + "D1.hdr", skip_header=18, skip_footer=1, delimiter=",")
wl = 1e4 / wnum

plt.figure()
plt.plot(wl, data[:, 1280 // 2, 1280 // 2])
plt.xlim(3, 5)
