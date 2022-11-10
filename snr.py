import sys

sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import phase_correction as pc

path = r"/home/peterchang/SynologyDrive/Research_Projects/Microscope/FreeRunningSpectra/11-09-2022/"
bckgnd = np.load(path + "stage1_5116_stage2_8500_phase_corrected.npy", mmap_mode='r')
su8 = np.load(path + "stage1_5300_stage2_8970_phase_corrected.npy", mmap_mode='r')
