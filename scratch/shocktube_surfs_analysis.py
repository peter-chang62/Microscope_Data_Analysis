import sys

sys.path.append("../include/")
import numpy as np
import os
import clipboard_and_style_sheet as cr
import td_phase_correct as td
import matplotlib.pyplot as plt
import digital_phase_correction as dpc

key = lambda s: float(s.split("LoopCount_")[1].split("_Datetime")[0])
ppifg = 17506
center = ppifg // 2

f_MHz = np.fft.rfftfreq(ppifg, d=1e-9) * 1e-6
f_nyq = np.fft.rfftfreq(ppifg)

# ___________________ load path names _________________________________________
path_batt_28 = r"/Volumes/Extreme SSD/Research_Projects/Shocktube" \
               r"/DATA_MATT_PATRICK_TRIP_2/06-30-2022/Battalion_28/"
path_batt_28_co = path_batt_28 + "card1/"
path_batt_28_h2co = path_batt_28 + "card2/"

path_batt_31 = r"/Volumes/Extreme SSD/Research_Projects/Shocktube" \
               r"/DATA_MATT_PATRICK_TRIP_2/06-30-2022/Battalion_31/"
path_batt_31_co = path_batt_31 + "card1/"
path_batt_31_h2co = path_batt_31 + "card2/"

# load co (card 1) path names
names_co_batt_28 = [i.name for i in os.scandir(path_batt_28_co)]
names_co_batt_31 = [i.name for i in os.scandir(path_batt_31_co)]
names_co_batt_28.sort(key=key)
names_co_batt_31.sort(key=key)

names_co_batt_28 = [path_batt_28_co + i for i in names_co_batt_28]
names_co_batt_31 = [path_batt_31_co + i for i in names_co_batt_31]

# load co (card 2) path names
names_h2co_batt_28 = [i.name for i in os.scandir(path_batt_28_h2co)]
names_h2co_batt_31 = [i.name for i in os.scandir(path_batt_31_h2co)]
names_h2co_batt_28.sort(key=key)
names_h2co_batt_31.sort(key=key)

names_h2co_batt_28 = [path_batt_28_h2co + i for i in names_h2co_batt_28]
names_h2co_batt_31 = [path_batt_31_h2co + i for i in names_h2co_batt_31]

names_co = names_co_batt_28 + names_co_batt_31
names_h2co = names_h2co_batt_28 + names_h2co_batt_31

# ____________________ analysis _______________________________________________
IND_SHOCK = np.zeros(len(names_co), dtype=int)
# for loop
for i in range(len(names_co)):
    # load data
    co = np.fromfile(names_co[i], '<h')[:-64]
    co = co / co.max()

    # reshape data
    ind_throw_out = np.argmax(co[:ppifg])
    co = co[ind_throw_out + center:]
    co = co[:(len(co) // ppifg) * ppifg]
    co.resize((len(co) // ppifg, ppifg))

    # load h2co data
    h2co = np.fromfile(names_h2co[0], '<h')[:-64]
    h2co = h2co / h2co.max()

    # reshape h2co data according to co data
    h2co = h2co[ind_throw_out + center:]
    h2co = h2co[:(len(h2co) // ppifg) * ppifg]
    h2co.resize((len(h2co) // ppifg, ppifg))

    # locate shock
    ft_co = np.fft.rfft(co, axis=1)
    ft_filtered = ft_co.copy()
    ft_filtered[:, np.argmin(abs(f_MHz - 5)):] = 0
    bckgnd = np.fft.irfft(ft_filtered, axis=1)
    bckgnd.resize(bckgnd.size)
    ind_rise = np.argmax(bckgnd)
    bckgnd_flipped = bckgnd * - 1
    ind_shock = np.argmax(bckgnd_flipped[ind_rise:]) + ind_rise
    IND_SHOCK[i] = ind_shock

    # filter out fceo
    list_filter = np.array([
        [4350, 4400],
        [4440, 4853],
        [5200, 5400]
    ])
    ft_h2co = np.fft.rfft(h2co, axis=1)
    for filt in list_filter:
        ft_co[:, filt[0]:filt[1]] = 0
        ft_h2co[:, filt[0]:filt[1]] = 0
    co_filt = np.fft.irfft(ft_co)
    h2co_filt = np.fft.irfft(ft_h2co)

    # global reference
    if i == 0:
        co_global_ref = co_filt[0]
        h2co_global_ref = h2co_filt[0]

    if i > 0:
        co_corr = np.vstack([co_global_ref, co_filt])
        h2co_corr = np.vstack([h2co_global_ref, h2co_filt])
    else:
        co_corr = co_filt
        h2co_corr = h2co_filt

    for n in range(len(co_corr)):
        ind_diff_co = center - np.argmax(co_corr[n])
        co_corr[n] = np.roll(co_corr[n], ind_diff_co)

        ind_diff_h2co = center - np.argmax(h2co_corr[n])
        h2co_corr[n] = np.roll(h2co_corr[n], ind_diff_h2co)

    opt_co = td.Optimize(co_corr[:, center - 25:center + 25])
    opt_h2co = td.Optimize(h2co_corr[:, center - 20:center + 20])

    opt_co.phase_correct(co_corr)
    opt_h2co.phase_correct(h2co_corr)

    if i > 0:
        co_corr = co_corr[1:]
        h2co_corr = h2co_corr[1:]

    #  ________ save filtered and phase corrected data ________________________

    print(f'_________________ {len(names_co) - i - 1} _______________________')
