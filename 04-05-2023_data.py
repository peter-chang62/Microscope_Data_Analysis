import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from scipy.integrate import simpson
from tqdm import tqdm


s_1 = np.load("temp/img1.npy", mmap_mode="r+")
s_2 = np.load("temp/img2.npy", mmap_mode="r+")

s_1 = s_1[1:]
s_2 = s_2[1:]

x = np.arange(0, s_1.shape[1] * 5, 5)
y = np.arange(0, s_1.shape[0] * 5, 5)

filt_s1 = np.array(
    [
        [19400, 19480],
        [21500, 21750],
        [25850, 26025],
        [34480, 34560],
        [0, 20],
        [38860, 38900],
    ]
)


def apply_filter(s, filt):
    for f in filt_s1:
        s[f[0] : f[1]] = 0


s_1.resize(s_1.shape[0] * s_1.shape[1], s_1.shape[2])

for n, s in enumerate(tqdm(s_1)):
    apply_filter(s, filt_s1)
