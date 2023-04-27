import io
import matplotlib.pyplot as plt

# from PySide2.QtGui import QGuiApplication, QImage
from PyQt5.QtGui import QGuiApplication, QImage, QColor
from PyQt5.Qt import Qt
from PyQt5.QtCore import QByteArray
import numpy as np

# generate an image to copy to clipboard
fig = plt.figure()
plt.plot(np.random.random(1000))

# store the image in a buffer using savefig(), this has the
# advantage of applying all the default savefig parameters
# such as background color; those would be ignored if you simply
# grab the canvas using Qt
temp_path = "fig/clipboard.png"
fig.savefig(temp_path, format="png", dpi=300, transparent=True)
image = QImage(temp_path, format="png")
image = image.convertToFormat(QImage.Format_ARGB32)
QGuiApplication.clipboard().setImage(image)
