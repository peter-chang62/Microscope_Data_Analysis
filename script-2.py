import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import td_phase_correct as td
from tqdm import tqdm
import os

# path = r"D:\230105_1G113 DATA/"
path = r"/Volumes/Extreme SSD/Research_Projects/1G113_Kristina_Chang" \
       r"/230105_1G113 DATA/"
names = [i.name for i in os.scandir(path)]
[names.remove(i) for i in names if not "pc" in i]
[names.remove(i) for i in names if not "pc" in i]

key = lambda s: int(s.split("_")[0])
names.sort(key=key)
names = [path + i for i in names]

ppifg = int(2e7)
center = ppifg // 2
data = np.zeros((len(names), ppifg))
for n, i in enumerate(tqdm(names)):
    data[n] = np.load(i)

opt = td.Optimize(data[:, center - 118:center + 160])
opt.phase_correct(data)

avg = np.mean(data, 0)
ft = np.fft.rfft(avg)
