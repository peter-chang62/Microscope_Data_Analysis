import sys

sys.path.append("include/")
import numpy as np
from tqdm import tqdm
import os
import clipboard_and_style_sheet as cr
import td_phase_correct as td
import matplotlib.pyplot as plt
import digital_phase_correction as dpc

if os.name == "posix":
    path = r"/Volumes/Extreme SSD/Research_Projects/Shocktube" \
           r"/DATA_MATT_PATRICK_TRIP_2/06-30-2022" \
           r"/Vacuum_Background_end_of_experiment/"
else:
    path = r"D:\DATA_MATT_PATRICK_TRIP_2\06-30-2022" \
           r"\Vacuum_Background_end_of_experiment/"
path_h2co = path + "3.5um_filter_114204x17506.bin"
path_co = path + "4.5um_filter_114204x17506.bin"

ppifg = 17506
center = ppifg // 2

h2co = np.fromfile(path_h2co, '<h')
h2co = h2co[np.argmax(h2co[:ppifg]) + center:]
h2co = h2co[:(len(h2co) // ppifg) * ppifg]
h2co.resize((len(h2co) // ppifg, ppifg))

step = 250
ft_h2co = np.zeros((len(h2co), len(np.fft.rfftfreq(len(h2co[0])))), dtype=complex)
h = 0
for _ in tqdm(range(0, len(h2co), step)):
    ft_h2co[h: h + step] = np.fft.rfft(h2co[h: h + step], axis=1)
    h += step
