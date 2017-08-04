import pyqtgraph as pg
from PyQt5 import QtGui
from .guitools import findLowHigh

from .colormaps import makeALBULACmap

class ImageMainWindow(QtGui.QMainWindow):
    ''' This is a window on its own'''
    def __init__(self,*args,img=None, maxcts=None, levels=None, lockAspect=True, parent=None, **kwargs):
        super(ImageMainWindow, self).__init__(parent)
        self.imgv = ImageWindow(**kwargs)
        self.setCentralWidget(self.imgv)
        if levels is None:
            low, high = findLowHigh(img,maxcts=maxcts)
        else:
            low,high = levels
        self.imgv.setImage(img.T,levels=levels)
        self.imgv.setLevels(low,high)
        self.imgv.setHistogramRange(low, high)
        self.imgv.setColorMap(makeALBULACmap())
        self.imgv.view.setAspectLocked(lockAspect)
        self.imgv.show()
        self.setGeometry(200,200,800,800)
        #self.setMinimumWidth(1200)
        #self.setMinimumHeight(800)
        self.statusBar().showMessage('Ready')
        self.show()

    def setImage(self,img,levels=(0,1)):
        self.imgv.setImage(img,levels=levels)

    def setAspectLock(self, val):
        self.imgv.view.setAspectLocked(val)


class ImageWindow(pg.ImageView):
    def __init__(self, *args, **kwargs):
        super(ImageWindow,self).__init__(*args, **kwargs)
        #proxy = pg.SignalProxy(self.scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

        # add horizontal lines and label
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.label = pg.LabelItem(justify='right')

        self.lbl = self.addItem(self.label)
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)


        self.scene.sigMouseMoved.connect(self.mouseMoved)

    def mouseMoved(self, evt):
        pos = evt.x(), evt.y()  ## using signal proxy turns original arguments into a tuple
        #print(pos)
        if self.view.contains(evt.toPoint()):
            mousePoint = self.view.mapSceneToView(evt.toPoint())
            index = int(mousePoint.x()), int(mousePoint.y())
            # Access parent widget's status bar
            parent1 = self.parentWidget()
            parent2 = self.parentWidget().parentWidget()
            if hasattr(parent1, "statusBar"):
                stbar = parent1.statusBar()
            elif hasattr(parent2, "statusBar"):
                stbar = parent2.statusBar()
            #if index > 0 and index < len(data1):
            if self.image is not None and index[0] > 0 and index[1] > 0 and index[0] < self.image.shape[0] and index[1] < self.image.shape[1]:
                datpt = self.image[int(mousePoint.x()), int(mousePoint.y())]
                stbar.showMessage("({:.1f}, {:.1f}) [{:0.1f}]".format(mousePoint.x(), mousePoint.y(), datpt))
            else:
                stbar.showMessage("({:.1f}, {:.1f})".format(mousePoint.x(), mousePoint.y()))
                pass
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def setColorMap(self, *args, **kwargs):
        # VERSION 0.9.10 and up
        if hasattr(super(ImageWindow, self), 'setColorMap'):
            super(ImageWindow, self).setColorMap(*args, **kwargs)
        else:
            self.ui.histogram.gradient.setColorMap(*args, **kwargs)

