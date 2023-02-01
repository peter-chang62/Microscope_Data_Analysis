import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import td_phase_correct as td
import os

path = r'C:\Users\fastdaq\Desktop\230105_1G113 DATA/'
names = [i.name for i in os.scandir(path)]
[names.remove(i) for i in names if not ".npy" in i]
[names.remove(i) for i in names if not ".npy" in i]
[names.remove(i) for i in names if "corravg" in i]
[names.remove(i) for i in names if "corravg" in i]
[names.remove(i) for i in names if "pc" in i]
[names.remove(i) for i in names if "pc" in i]
key = lambda s: int(s.split(".npy")[0])
names.sort(key=key)

ppifg = int(2e7)
center = ppifg // 2

# index = 4
for index in range(15, 20):
    x = np.load(path + names[index])
    start = np.argmax(x[:2].flatten())
    x.resize(x.size)
    x = x[start + center:]
    N_ifgs = len(x) // ppifg
    x = x[:N_ifgs * ppifg]
    x.resize((N_ifgs, ppifg))

    for n, i in enumerate(x):
        x[n] = np.roll(i, center - np.argmax(i))

    opt = td.Optimize(x[:, center - 118:center + 160])
    opt.phase_correct(x)

    np.save(path + f'{key(names[index])}_pc.npy', np.mean(x, 0))
