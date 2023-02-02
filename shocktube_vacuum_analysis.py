import sys

sys.path.append("include/")
import numpy as np
import os
import clipboard_and_style_sheet as cr
import td_phase_correct as td
import matplotlib.pyplot as plt
import digital_phase_correction as dpc

path = r"/Volumes/Extreme SSD/Research_Projects/Shocktube" \
       r"/DATA_MATT_PATRICK_TRIP_2/06-30-2022" \
       r"/Vacuum_Background_end_of_experiment/"
path_h2co = path + "3.5um_filter_114204x17506.bin"
path_co = path + "4.5um_filter_114204x17506.bin"

ppifg = 17506
center = ppifg // 2

h2co = np.fromfile(path_h2co, '<h')
h2co = h2co[np.argmax(h2co[:ppifg]) + center:]
h2co = h2co[:(len(h2co) // ppifg) * ppifg]
h2co.resize((len(h2co) // ppifg, ppifg))

ft_h2co = np.fft.rfft(h2co, axis=1)
