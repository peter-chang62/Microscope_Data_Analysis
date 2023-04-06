import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet as cr
from scipy.integrate import simpson

path = r"/Volumes/Peter SSD/Research_Projects/Microscope/Images/03-30-2023/"

s = np.load(path + "image_50um_res_X_6110_10110_Y_372_4372.npy", mmap_mode="r")
img = simpson(s, axis=-1)

plt.figure()
plt.imshow(img)
