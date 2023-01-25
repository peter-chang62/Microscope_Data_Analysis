"""From having worked with scripting for a while now, I think I would
actually like to have an interactive GUI for the purpose of figuring out the
lock window. Specifically, I would like to be able to visualize where the
spectrum would fall inside the nyquist window in both the optical domain,
and in the DCS frequency domain, and be able to see where the f0's would
fall"""

import threading
import numpy as np
import scipy.constants as sc
import PyQt5.QtWidgets as qt
from Error_Window import Ui_Form
import PlotWidgets as pw
import PyQt5.QtGui as qtg
from Gui_DCS_Lockpoints import Ui_MainWindow
import pyqtgraph as pg
import nyquist_bandwidths as nq

min_dfrep = 1


class ErrorWindow(qt.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def set_text(self, text):
        self.textBrowser.setText(text)


def raise_error(error_window, text):
    error_window.set_text(text)
    error_window.show()


class Gui(qt.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()

        self.error_window = ErrorWindow()

        self.plot_window_optical = pw.PlotWindow(self.le_xmin,
                                                 self.le_xmax,
                                                 self.le_ymin,
                                                 self.le_ymax,
                                                 self.gv)
        self.plot_window_rf = pw.PlotWindow(self.le_xmin_2,
                                            self.le_xmax_2,
                                            self.le_ymin_2,
                                            self.le_ymax_2,
                                            self.gv_2)
        self.lr = pg.LinearRegionItem()
        self.lr_nyquist = pg.LinearRegionItem(pen=pg.mkPen(color=qtg.QColor(0, 0, 0, 255), width=1))
        self.lr.setBrush(pg.mkBrush(color=qtg.QColor(255, 0, 0, 255 // 2)))
        self.lr_nyquist.setBrush(pg.mkBrush(color=qtg.QColor(0, 0, 0, 255 // 4)))
        self.lr.setMovable(False)
        self.lr_nyquist.setMovable(False)
        self.region_list = [self.lr_nyquist]

        self.lr_rf = pg.LinearRegionItem()
        self.lr_rf.setMovable(False)
        self.plot_window_rf.plotwidget.addItem(self.lr_rf)

        self.lr_f0 = pg.LinearRegionItem(pen=pg.mkPen(color=qtg.QColor(0, 0, 0, 255), width=1))
        self.lr_f0.setMovable(False)
        self.lr_f0.setBrush(pg.mkBrush(color=qtg.QColor(0, 0, 0, 0)))
        self.plot_window_rf.plotwidget.addItem(self.lr_f0)

        self.plot_window_optical.plotwidget.addItem(self.lr)
        self.plot_window_optical.plotwidget.addItem(self.lr_nyquist)

        self.curve_optical = pw.create_curve()
        self.curve_rf = pw.create_curve()
        self.plot_window_optical.plotwidget.addItem(self.curve_optical)
        self.plot_window_rf.plotwidget.addItem(self.curve_rf)

        self.le_min_wl.setValidator(qtg.QDoubleValidator())
        self.le_max_wl.setValidator(qtg.QDoubleValidator())
        self.le_f01.setValidator(qtg.QDoubleValidator())
        self.le_f02.setValidator(qtg.QDoubleValidator())
        self.le_rep_rate.setValidator(qtg.QDoubleValidator())
        self.le_dfrep.setValidator(qtg.QIntValidator())

        self.lcd.setSegmentStyle(qt.QLCDNumber.Flat)
        self.lcd.setDigitCount(6)

        self.nu_min = sc.c / (float(self.le_max_wl.text()) * 1e-6)
        self.nu_max = sc.c / (float(self.le_min_wl.text()) * 1e-6)
        self.f01 = float(self.le_f01.text()) * 1e6
        self.f02 = float(self.le_f02.text()) * 1e6
        self.frep = float(self.le_rep_rate.text()) * 1e6
        self.dfrep = min_dfrep

        self.connect()

        self.verticalScrollBar.setMinimum(min_dfrep)
        self.verticalScrollBar.setSingleStep(10)

        self.update_wl_max()
        self.update_wl_min()

        self.update_nyquist()
        self.plot_window_rf.update_xmax()

        self.update_f01()
        self.update_f02()

    def connect(self):
        self.le_min_wl.editingFinished.connect(self.update_wl_min)
        self.le_max_wl.editingFinished.connect(self.update_wl_max)
        self.le_f01.editingFinished.connect(self.update_f01)
        self.le_f02.editingFinished.connect(self.update_f02)
        self.le_rep_rate.editingFinished.connect(self.update_frep)
        self.verticalScrollBar.valueChanged.connect(self.update_dfrep_from_lcd)
        self.le_dfrep.editingFinished.connect(self.update_dfrep_from_le)

    @property
    def wl_min(self):
        return sc.c / self.nu_max

    @property
    def wl_max(self):
        return sc.c / self.nu_min

    @wl_min.setter
    def wl_min(self, val):
        self.nu_max = sc.c / val
        self.lr.setRegion([self.nu_min * 1e-12, self.nu_max * 1e-12])

    @wl_max.setter
    def wl_max(self, val):
        self.nu_min = sc.c / val
        self.lr.setRegion([self.nu_min * 1e-12, self.nu_max * 1e-12])

    def set_scrollbar_max(self):
        dnu = self.nu_max - self.nu_min
        max_dfr = nq.bandwidth(self.frep, dnu)
        if max_dfr < 1e5:
            self.verticalScrollBar.setMaximum(int(np.round(max_dfr)))
        else:
            self.verticalScrollBar.setMaximum(int(1e5))
        self.lr.setRegion([self.nu_min * 1e-12, self.nu_max * 1e-12])

    def update_wl_min(self):
        wl_min = float(self.le_min_wl.text())
        if wl_min <= 0:
            raise_error(self.error_window, "min wavelength must be >0")
            return
        elif wl_min >= self.wl_max * 1e6:
            raise_error(self.error_window,
                        "min wavelength must be < max wavelength")
            self.le_min_wl.setText(str(self.wl_min * 1e6))
            return
        self.wl_min = wl_min * 1e-6

        # _____________________ updates based on min wavelength _______________
        self.set_scrollbar_max()
        self.update_rf_plot()

    def update_wl_max(self):
        wl_max = float(self.le_max_wl.text())
        if wl_max <= 0:
            raise_error(self.error_window, "max wavelength must be >0")
            return
        elif wl_max <= self.wl_min * 1e6:
            raise_error(self.error_window,
                        "max wavelength must be > min wavelength")
            self.le_max_wl.setText(str(self.wl_max * 1e6))
            return
        self.wl_max = wl_max * 1e-6

        # __________________________ updates based on max wavelength __________
        self.set_scrollbar_max()
        self.update_rf_plot()

    def update_f01(self):
        f01 = float(self.le_f01.text())
        f01 *= 1e6
        if not (0 <= f01 <= self.frep):
            raise_error(self.error_window, "f01 must be 0 <= f01 <= frep")
            self.le_f01.setText(str(self.f01 * 1e-6))
        self.f01 = f01

        # _____________________ updates to run when updating f01 ______________
        self.lr_f0.setRegion([self.f01 * 1e-6, self.f02 * 1e-6])

    def update_f02(self):
        f02 = float(self.le_f02.text())
        f02 *= 1e6
        if not (0 <= f02 <= self.frep):
            raise_error(self.error_window, "f02 must be 0 <= f02 <= frep")
            self.le_f02.setText(str(self.f02 * 1e-6))
        self.f02 = f02

        # _________________________ updates to run when updating f02 __________
        self.lr_f0.setRegion([self.f01 * 1e-6, self.f02 * 1e-6])

    def update_frep(self):
        frep = float(self.le_rep_rate.text())
        frep *= 1e6
        if any([frep < self.f01, frep < self.f02]):
            raise_error(self.error_window, "frep must be >= f01 and f02")
            self.le_rep_rate.setText(str(self.frep * 1e-6))
            return
        self.frep = frep

        # ________________________ updates based on frep ______________________
        self.set_scrollbar_max()
        self.update_nyquist()

    def update_dfrep_from_lcd(self):
        self.lcd.display(self.verticalScrollBar.value())
        self.dfrep = self.verticalScrollBar.value()

        # _____________________ updates based on delta frep____________________
        self.set_scrollbar_max()
        self.update_nyquist()

    def update_dfrep_from_le(self):
        dfrep = int(self.le_dfrep.text())
        if not min_dfrep <= dfrep <= 1e5:
            raise_error(self.error_window,
                        f"delta frep must be {min_dfrep} <= dfrep <= 100,000")
            self.le_dfrep.setText(str(self.dfrep))
            return
        self.dfrep = dfrep
        self.verticalScrollBar.setValue(self.dfrep)

        # __________________ updates based on delta frep_______________________
        self.update_nyquist()

    def update_nyquist(self):
        bandwidth = nq.bandwidth(self.frep, self.dfrep)
        bandwidth_THz = bandwidth * 1e-12

        N = np.ceil(self.nu_max / bandwidth)
        N_diff = int(N - len(self.region_list))
        if N_diff > 0:  # N_diff is positive -> add region items
            for n in range(N_diff):
                lr = pg.LinearRegionItem(pen=pg.mkPen(color=qtg.QColor(0, 0, 0, 255), width=1))
                lr.setBrush(self.lr_nyquist.brush)
                lr.setMovable(False)

                self.plot_window_optical.plotwidget.addItem(lr)
                self.region_list.append(lr)
        elif N_diff < 0:  # N_diff is negative -> remove region items
            for n in range(N_diff, 0):
                self.plot_window_optical.plotwidget.removeItem(self.region_list[n])
            self.region_list = self.region_list[:N_diff]

        for n, lr in enumerate(self.region_list):
            lr.setRegion(np.array([0, bandwidth_THz]) + bandwidth_THz * n)

        # _______________ updates to run whenever running update_nyquist ______
        self.update_rf_plot()

    def update_rf_plot(self):
        last_window = np.array(self.region_list[-1].getRegion()) * 1e12
        alias = self.nu_min < last_window[0]
        if alias:
            self.lr_rf.setBrush(pg.mkBrush(color=qtg.QColor(255, 0, 0)))
            self.lr.setBrush(pg.mkBrush(color=qtg.QColor(255, 0, 0, 255 // 2)))
        else:
            self.lr_rf.setBrush(pg.mkBrush(color=qtg.QColor(0, 255, 0)))
            self.lr.setBrush(pg.mkBrush(color=qtg.QColor(0, 255, 0, 255 // 2)))

        compression = self.frep / self.dfrep
        vi_rf = self.nu_min / compression
        vf_rf = self.nu_max / compression
        dist = vf_rf - self.frep / 2
        N_trans = np.ceil(dist / self.frep)
        region = np.array([vi_rf, vf_rf]) - N_trans * self.frep
        self.lr_rf.setRegion(abs(region * 1e-6))


if __name__ == '__main__':
    app = qt.QApplication([])
    gui = Gui()
    app.exec()
