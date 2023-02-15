# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr

cr.style_sheet()

# %% global variables
if os.name == "nt":
    # this would be something else on the GaGe computer...
    path_folder = (
        r"C:\\Users\\pchan\\SynologyDrive\\Research_Projects\\Microscope\\Images/"
    )

else:
    # linux partition
    # path_folder = r"/Users/peterchang/SynologyDrive/Research_Projects" \
    #               r"/Microscope/Images/"
    path_folder = r"/Volumes/Extreme SSD/Research_Projects/Microscope/Images/"

# %% function defs
def num_4():
    path = path_folder + "11-08-2022/"
    s = np.load(path + "stage1_5932_6066_stage2_8478_8575p5_step_2p5_ppifg_74180.npy")
    s = np.transpose(s, axes=[1, 0, 2])
    x = np.arange(5932, 6066, 2.5)
    y = np.arange(8478, 8575.5 + 2.5, 2.5)

    return x, y, s


def smallest_bar():
    path = path_folder + "11-08-2022/"
    s = np.load(path + "stage1_6274_6460_stage2_8593_8883_step_2p5_ppifg_74180.npy")
    s = np.transpose(s, axes=[1, 0, 2])
    x = np.arange(6274, 6460, 2.5)
    y = np.arange(8593, 8883 + 2.5, 2.5)

    return x, y, s
