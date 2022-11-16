import sys

sys.path.append("include/")
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
import phase_correction as pc
import scipy.signal as si
import os
import scipy.constants as sc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import digital_phase_correction as dpc

path = r"C:\Users\pchan\SynologyDrive\Research_Projects\Microscope\FreeRunningSpectra\11-09-2022/" \
       r"stage1_5116_stage2_8500_phase_corrected.npy"
data = np.load(path)

ll, ul = .1181, .1602
pdiff = dpc.get_pdiff()

"""
array looks like:

interferogram1
interferogram2
interferogram3
"""