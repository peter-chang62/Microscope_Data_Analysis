import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# get colormap
ncolors = 256
color_array = plt.get_cmap("CMRmap_r")(range(ncolors))

# change alpha values
color_array[:, -1] = np.linspace(1.0, 0.0, ncolors)

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name="CMRmap_r_t", colors=color_array)

# register this new colormap with matplotlib
plt.register_cmap(cmap=map_object)

# show some example data
f, ax = plt.subplots()
h = ax.imshow(np.random.rand(100, 100), cmap="CMRmap_r_t")
plt.colorbar(mappable=h)
