#from PyQt5 import QtGui, QtCore

from pycxdgui import pyqtgraph as pg
import numpy as np

from .colormaps import makeALBULACmap, makeALBULALUT
from .guitools import findLowHigh
from PyQt5 import QtCore

from ..tools.smooth import smooth2Dgauss

class ImageAnalWindow(pg.GraphicsLayoutWidget):
    def __init__(self,data,levels=None, maxcts=65535, iso=True, parent=None):
        '''
            iso : plot an iso curve
        '''
        super(ImageAnalWindow, self).__init__(parent)

        if data is None:
            print("Error, must supply data")

        if data.ndim != 2:
            print("Error, cannot continue, expected a 2D image")

        imgdata = data.T

        # Find levels
        if levels is None:
            low, high = findLowHigh(imgdata,maxcts=maxcts)
        else:
            low,high = levels

        self.setWindowTitle("Image with slice plotting")

        # viewbox + axes
        #self.p3 = self.addPlot(col=1,colspan=1,rowspan=1)

        # item for image data
        self.p1 = self.addPlot(col=0,rowspan=4,colspan=4)
        if iso:
            self.isoLine_cross = pg.InfiniteLine(angle=0, movable=True, pen='b')
            self.isoLine_cross.setValue(10)

        self.img = pg.ImageItem()
        self.p1.addItem(self.img)
        if iso:
            self.p1.addItem(self.isoLine_cross)
            self.isoLine_cross.setZValue(10)
        self.p1.invertY()

        self.roi1 = pg.ROI([-data.shape[1],20],[3*data.shape[1],30])
        self.roi1.addScaleHandle([.5, 1], [.5, .5])
        self.roi1.addScaleHandle([0, .5], [.5, .5])
        #self.roi1.setMouseEnabled(x=False)
        self.p1.addItem(self.roi1)
        self.roi1.setZValue(10) #bring to front

        # Isocurve
        if iso:
            self.iso = pg.IsocurveItem(level=.8, pen='g')
            self.iso.setParentItem(self.img)
            self.iso.setZValue(5)

        # Color Control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.addItem(self.hist,rowspan=4)

        # draggable line for isocurve
        if iso:
            self.isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
            self.hist.vb.addItem(self.isoLine)
            self.isoLine.setValue(.8)
            self.isoLine.setZValue(1000)
        self.hist.vb.setMouseEnabled(x=False)
        self.hist.setHistogramRange(low,high)

        self.nextRow()
        self.nextRow()
        self.nextRow()
        self.nextRow()
        self.p2 = self.addPlot(colspan=4)
        self.p2.setMaximumHeight(250)
        self.resize(800,800)




        self.show()

        self.p1.autoRange()
        self.p1.setXRange(0, data.shape[1])
        self.p1.setYRange(0, data.shape[0])


        self.imgdata = imgdata
        self.img.setImage(self.imgdata,levels=(low,high))
        self.img.setLookupTable(makeALBULALUT())
        #self.img.setLevels(low,high)
        #self.hist.setLevels(low, high)

        # smooth image before getting iso
        if iso:
            self.iso.setData(smooth2Dgauss(self.imgdata.astype(float), sigma=2))
            #self.iso.setData(pg.gaussianFilter(self.imgdata, (2,2)))

        # position and scale of image
        #self.img.scale(.2, .2)
        #self.img.translate(-50, 0)


        self.roi1.sigRegionChanged.connect(self.updatePlot)
        self.updatePlot()

        if iso:
            self.isoLine.sigDragged.connect(self.updateIsocurve)
            self.isoLine_cross.sigDragged.connect(self.updateIsocurve2)

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == (QtCore.Qt.Key_Control | QtCore.Qt.Key_W):
            print("Pressed Ctrl + W")
        elif key == (QtCore.Qt.Key_A)|(QtCore.Qt.Key_B):
            print("Pressed A and B")

    def updatePlot(self):
        selected, coords = self.roi1.getArrayRegion(self.imgdata, self.img, returnMappedCoords=True)
        #w = np.where((coords[0] >=0)*(coords[0] < self.imgdata.shape[0])*(coords[0] >=0)*(coords[0] < self.imgdata.shape[1]))
        #datselected = self.imgdata[w]
        #self.p2.plot(datselected.mean(axis=1), clear=True)
        self.p2.plot(selected.mean(axis=1), clear=True)


    def updateIsocurve(self):
        self.iso.setLevel(self.isoLine.value())

    def updateIsocurve2(self):
        self.iso.setLevel(self.isoLine.value())

