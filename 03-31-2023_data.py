import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from scipy.integrate import simpson
from tqdm import tqdm


path = r"/Volumes/Peter SSD/Research_Projects/Microscope/Images/03-31-2023/"

s_1 = np.load(path + "img1.npy", mmap_mode="r")
s_2 = np.load(path + "img2.npy", mmap_mode="r")

# %% --------------------------------------------------------------------------
# i_1 = simpson(s_1, axis=-1)
# i_2 = simpson(s_2, axis=-1)

i_1 = np.load("temp/i_1.npy")
i_2 = np.load("temp/i_2.npy")

i_1 = i_1[:, :30]
i_total = np.hstack((i_1, i_2))

# s_1 = s_1[:, :30]
# s_total = np.hstack((s_1, s_2))

s_total = np.load("temp/s_copy.npy", mmap_mode="r+")

# %% --------------------------------------------------------------------------
f_MHz = np.fft.rfftfreq(77760, d=1e-9) * 1e-6

# %% --------------------------------------------------------------------------
s_1 = s_total[:, :30]

plt.figure()
plt.plot(f_MHz, s_1[0, 0])
plt.plot(f_MHz, s_1[-1, -1])
plt.ylim(0, 36e3)

# filt_s1 = np.array(
#     [[31.150, 31.350], [249.6, 250.6], [278, 286], [333, 336.5], [436, 438.5]]
# )

# [[plt.axvline(i, color="r") for i in e] for e in filt_s1]

# for n, f in enumerate(tqdm(filt_s1)):
#     f_ll, f_ul = f
#     (ind,) = np.logical_and(f_ll < f_MHz, f_MHz < f_ul).nonzero()
#     s_1[:, :, ind] = 0

# plt.plot(f_MHz, s_1[0, 0])

# %% --------------------------------------------------------------------------
s_2 = s_total[:, 30:]

# %%
plt.figure()
plt.plot(f_MHz, s_2[0, 0], label="multiple pulses")
plt.plot(f_MHz, s_1[0, 0], label="expected")
plt.plot(f_MHz, s_2[-1, -1], label="low signal")
plt.ylim(0, 36e3)
plt.legend(loc='best')
