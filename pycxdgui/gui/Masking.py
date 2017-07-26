#Mask Polygon Classes
import numpy as np
import pyqtgraph as pg
import sys
from PyQt5 import QtGui, QtCore
import pdb
import h5py
import os

from PIL import Image

from classes.Mask import Mask

from gui.guitools import findLowHigh

class MPoly(QtGui.QMainWindow):
    def __init__(self, img, datatable=None, mask=None, imgwidget=None):
        ''' Initialize the polygon for mask.
                This object handles the graphics as well as the mask
                creation (it might be a good idea to separate them later
                if need be).
                There are four types of mask creation
                possible: 1. polygon 2. Rectangle 3. Circle 4. Ellipse.
                A value of zero here means do not include and 1 include.
                (Note this is the inverse of how our analysis routines in yorick work).
                Works on an xray img set

                Note: There is deprecated code in here of how we used to
                compute the pixels in a polygon (swath and polymask).
                See their respective comments.
        '''
        super(MPoly, self).__init__()
        # TODO : remove and add error checking
        if img is None:
            img = np.ones((100,100))
        # pointer to imgwidget
        self.imgwidget = imgwidget

        self.blemish = np.ones_like(img)
        self.setimgdata(img.astype(float))
        self.points = []
        self.plotw = pg.GraphicsLayoutWidget()#plot widget
        self.setCentralWidget(self.plotw)
        self.resize(1500,600)
        self.poslabel = pg.LabelItem(justify='right')
        self.poslabel.setText("test")
        self.plotw.addItem(self.poslabel)

        if datatable is None:
            # some defaults
            datatable = dict()
            datatable['SDIR'] = "../storage"

        self.datatable = datatable

        #make a status bar
        self.statusBar().showMessage('Ready')

        #make a toolbar
        self.polymAction = QtGui.QAction(QtGui.QIcon('icons/mpolyicon.png'), 'MakePoly', self)
        self.polymAction.setStatusTip('Make Poly ROI')
        self.polymAction.triggered.connect(self.startPolyCapture)

        self.circlemAction = QtGui.QAction(QtGui.QIcon('icons/roicircleicon.png'), 'MakeCircle', self)
        self.circlemAction.setStatusTip('Make Circle ROI')
        self.circlemAction.triggered.connect(self.makeCircleROI)

        self.ellipsemAction = QtGui.QAction(QtGui.QIcon('icons/roiellipseicon.png'), 'MakeEllipse', self)
        self.ellipsemAction.setStatusTip('Make Ellipse ROI')
        self.ellipsemAction.triggered.connect(self.makeEllipseROI)

        self.rectmAction = QtGui.QAction(QtGui.QIcon('icons/roisquareicon.png'), 'MakeRect', self)
        self.rectmAction.setStatusTip('Make Rectangular ROI')
        self.rectmAction.triggered.connect(self.makeRectROI)

        self.clearROIAction = QtGui.QAction(QtGui.QIcon('icons/roiclearicon.png'), 'ClearROI', self)
        self.clearROIAction.setStatusTip('Clear ROI')
        self.clearROIAction.triggered.connect(self.clearROI)

        self.imaskAction = QtGui.QAction(QtGui.QIcon('icons/imaskicon.png'), 'IMask', self)
        self.imaskAction.setStatusTip('Include the region in mask')
        self.imaskAction.triggered.connect(self.imaskFromROI)

        self.xmaskAction = QtGui.QAction(QtGui.QIcon('icons/xmaskicon.png'), 'XMask', self)
        self.xmaskAction.setStatusTip('Exclude the region in mask')
        self.xmaskAction.triggered.connect(self.xmaskFromROI)

        self.clearMaskAction = QtGui.QAction(QtGui.QIcon('icons/clearmaskicon.png'), 'ClearMask', self)
        self.clearMaskAction.setStatusTip('Clear the mask')
        self.clearMaskAction.triggered.connect(self.clearMask)

        self.saveMaskAction = QtGui.QAction(QtGui.QIcon('icons/savemaskicon.png'), 'Save Mask', self)
        self.saveMaskAction.setStatusTip('Save the mask')
        self.saveMaskAction.triggered.connect(self.saveMask)

        self.openMaskAction = QtGui.QAction(QtGui.QIcon('icons/loadmaskicon.png'), 'Load Mask', self)
        self.openMaskAction.setStatusTip('Load the mask')
        self.openMaskAction.triggered.connect(self.openMask)

        self.loadBlemishAction = QtGui.QAction(QtGui.QIcon('icons/loadblemish.png'), 'Load Blemish', self)
        self.loadBlemishAction.setStatusTip('Load the Blemish file')
        self.loadBlemishAction.triggered.connect(self.loadBlemish)

        self.selectAllAction = QtGui.QAction(QtGui.QIcon('icons/selectall.png'), 'Select All', self)
        self.selectAllAction.setStatusTip('Select All')
        self.selectAllAction.triggered.connect(self.selectAll)

        self.sendMaskAction = QtGui.QAction(QtGui.QIcon('icons/sendmask.png'), 'Send Mask to Data', self)
        self.sendMaskAction.setStatusTip('Send Mask to Data')
        self.sendMaskAction.triggered.connect(self.sendMask)

        self.exitAction = QtGui.QAction(QtGui.QIcon('icons/exit.png'), '&Exit', self)
        self.exitAction.setStatusTip('Close')
        self.exitAction.triggered.connect(self.callExit)

        #Now add toolbar with actions
        self.toolbar = self.addToolBar('Mask Polygon Options')
        self.toolbar.addAction(self.polymAction)
        self.toolbar.addAction(self.circlemAction)
        self.toolbar.addAction(self.ellipsemAction)
        self.toolbar.addAction(self.rectmAction)
        self.toolbar.addAction(self.clearROIAction)
        self.toolbar.addAction(self.imaskAction)
        self.toolbar.addAction(self.xmaskAction)
        self.toolbar.addAction(self.clearMaskAction)
        self.toolbar.addAction(self.saveMaskAction)
        self.toolbar.addAction(self.openMaskAction)
        self.toolbar.addAction(self.loadBlemishAction)
        self.toolbar.addAction(self.selectAllAction)
        self.toolbar.addAction(self.sendMaskAction)
        self.toolbar.addAction(self.exitAction)

        #create a plot (basically has axes)
        self.plot = self.plotw.addPlot()
        self.mplot = self.plotw.addPlot()
        #create an image item (that you'll add to plot later)
        self.img = pg.ImageItem()
        self.mimg = pg.ImageItem()
        low,high= findLowHigh(img,maxcts=10000)
        print("low: {}, high: {}".format(low,high))
        self.plot.addItem(self.img)
        self.mplot.addItem(self.mimg)
        #self.proxy = pg.SignalProxy(self.plotw.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        #create a histogram (just to make cmax cmin choosing easier)
        self.hist = pg.HistogramLUTItem()
        self.mhist = pg.HistogramLUTItem()
        #tie it into the image by default nothing is selected
        #print(self.imgdata.shape)
        if mask is None:
            self._mask = Mask(shape=self.imgdata.shape)
        else:
            self._mask = Mask(mask=mask)

        self.blemish = np.ones(self.imgdata.shape,dtype=float);

        self.img.setImage(self.imgdata, xvals=np.linspace(0., .5, self.imgdata.shape[0]),levels=(low,high))
        self.mimg.setImage(self.mask,autoLevels=False,levels=[0,2])
        self.hist.setImageItem(self.img)
        self.mhist.setImageItem(self.mimg)
        self.mhist.setLevels(0,2)
        self.hist.setLevels(low, high)
        self.hist.setHistogramRange(low,high)
        self.plotw.addItem(self.hist)
        self.plotw.addItem(self.mhist)
        #status of mask (are you making another mask?)
        self.view = self.img.getViewBox()
        self.masking = 0
        #self.startPolyCapture()
        self.show()

    @property
    def mask(self):
        # mask is a callable class
        return self._mask()

    def makeCircleROI(self):
        self.clearROI()
        pos = [self.imgdata.shape[0]/2.,self.imgdata.shape[1]/2.]
        #self.roi = pg.CircleROI(pos,[100,100])
        self.roi = pg.PolyLineROI(pos)
        self.roitype='circle'
        #self.roi.setZValue(10)#make sure it's above image
        self.plot.addItem(self.roi)

    def makeRectROI(self):
        self.clearROI()
        pos = [self.imgdata.shape[0]/2,self.imgdata.shape[1]/2]
        self.roi = pg.RectROI(pos,[100,100])
        self.roitype='rect'
        self.roi.setZValue(10)#make sure it's above image
        self.plot.addItem(self.roi)

    def makeEllipseROI(self):
        self.clearROI()
        pos = [self.imgdata.shape[0]/2,self.imgdata.shape[1]/2]
        self.roi = pg.EllipseROI(pos,[100,100])
        self.roitype='ellipse'
        self.roi.setZValue(10)#make sure it's above image
        self.plot.addItem(self.roi)

    def startPolyCapture(self):
        ''' Start the capturing of a polygon. Basically, replaces
            the mouse capture event to capture events from a polygon
            mouse.
        '''
        self.clearROI()
        self.points = []
        self.masking = 1
        self.mouseClickEventOld = self.img.mouseClickEvent
        self.img.mouseClickEvent = self.mouseClickEvent
        self.line = pg.PlotDataItem(pxMode=True, pen='g')
        self.plot.addItem(self.line)

    def endPolyCapture(self):
        '''End the polygon capture.'''
        self.endMasking()
        if(hasattr(self,'points')):
            self.roi = pg.PolyLineROI(self.points,closed=True)
            self.roitype='poly'
            self.roi.setZValue(10)#make sure it's above image
            self.plot.addItem(self.roi)
            self.plot.removeItem(self.line)
            del self.line
            del self.points
        else:
            print("Error, no points captured. Ignoring mask.")

    def endMasking(self):
        '''Safely end the masking process.'''
        if(self.masking):
            self.masking = 0
        #self.polymask()
            self.img.mouseClickEvent = self.mouseClickEventOld
            del self.mouseClickEventOld

    def clearROI(self):
        '''Clear any ROI that exists.'''
        if(hasattr(self,'roi')):
            self.plot.removeItem(self.roi)
            del self.roi
        if(hasattr(self,'points')):
            del self.points
        if(hasattr(self,'line')):
            del self.line

    def findPixels(self):
        cols,rows = self.imgdata.shape
        m = np.mgrid[:cols,:rows]
        possx = m[0,:,:]+1
        possy = m[1,:,:]+1
        #poss = np.arange(cols*rows)+1
        #poss.shape = cols,rows
        possx.shape = cols,rows
        possy.shape = cols,rows
        try:
            mpossx = self.roi.getArrayRegion(possx,self.img).astype(int)
            mpossx = mpossx[np.nonzero(mpossx)]-1
            mpossy = self.roi.getArrayRegion(possy,self.img).astype(int)
            mpossy = mpossy[np.nonzero(mpossy)]-1
            return (mpossx,mpossy)
        except AttributeError:
            return None

    def imaskFromROI(self):
        '''Include ROI in mask.'''
        #Need to fix this later. getArrayRegion does a bloody interpolation
        px = self.findPixels()
        self._mask.include(px)
        #self.mask[px] = 1*self.blemish[px]
        try:
            self.plot.removeItem(self.roi)
            self.mimg.setImage(self.mask*self.blemish,autoLevels=False,levels=[0,2])
            del self.roi
        except AttributeError:
            pass

    def xmaskFromROI(self):
        '''Exclude ROI in mask.'''
        px = self.findPixels()
        #self.mask[px] *= 0
        self._mask.exclude(px)
        try:
            self.plot.removeItem(self.roi)
            self.mimg.setImage(self.mask*self.blemish,autoLevels=False,levels=[0,2])
            del self.roi
        except AttributeError:
            pass

    def clearMask(self):
        '''Clear the mask. No blemish needed.'''
        self.clearROI()
        #self.mask = self.mask*0
        self._mask.zero()
        self.mimg.setImage(self.mask,autoLevels=False,levels=[0,2])

    def selectAll(self):
        '''Clear the mask.'''
        self.clearROI()
        #self.mask = (self.mask*0 + 1)*self.blemish
        self._mask.one()
        self._mask.exclude(self.blemish==0)
        self.mimg.setImage(self.mask,autoLevels=False,levels=[0,2])

    def setmask(self,mask):
        self.mask = mask

    def getmask(self):
        return self.mask

    def setimgdata(self, imgdata):
        self.imgdata = imgdata

    def openMask(self):
        self.maskfilename= str(QtGui.QFileDialog.getOpenFileName(self, "Open Mask", self.datatable['SDIR'],'Mask File (*mask*.hd5 *mask*.tif *mask*.tiff);;All Files (*)')[0])
        if(os.path.isfile(self.maskfilename)):
            self._mask.load(self.maskfilename)
            #f = h5py.File(self.maskfilename,"r")
            #self.mask = np.array(f['mask'])
            #self.setmask(np.array(f['mask']))
            self.mimg.setImage(self.mask.astype(int),autoLevels=False,levels=[0,2])
            #f.close()
        else:
            msg = "Sorry, {} does not exist. Ignoring Mask Open.".format(self.maskfilename)
            self.statBarMsg.setText(msg)

    def saveMask(self):
        self.maskfilename = str(QtGui.QFileDialog.getSaveFileName(self, "Save Mask", self.datatable['SDIR'])[0])
        self._mask.save(self.maskfilename)
        #f = h5py.File(self.maskfilename,"w")
        #f['mask'] = self.getmask()#self.mask
        #f.close()

    def loadBlemish(self):
        '''Load the blemish file. Note you need to do this first as
        It will erase your mask.'''
        self.blemfilename= str(QtGui.QFileDialog.getOpenFileName(self, "Open Blemish", self.datatable['SDIR'],'Mask File (*blem*.hd5 *blem*.tif *blem*.tiff);;All Files (*)')[0])
        # TODO : add extension checking
        blemfilename = self.blemfilename
        if 'master.h5' in blemfilename:
            f = h5py.File(blemfilename,"r")
            self.blemish = np.array(f['mask'])
            f.close()
        elif 'tif' in blemfilename:
            try:
                #ddir = self.saxsdata.getelem("setup","DDIR")
                self.blemish= np.array(Image.open(blemfilename))
                if self.blemish.ndim == 3:
                    self.blemish = (self.blemish[:,:,0] > 0).astype(int)
            except IOError:
                print("Error could not read file")
                return False



        #self.mask = self.mask*self.blemish
        self._mask = Mask(mask=self.mask*self.blemish)
        self.mimg.setImage(self.mask,autoLevels=False,levels=[0,2])

    def sendMask(self):
        if self.imgwidget is not None:
            self.imgwidget.mask = self._mask.mask
        else:
            print("Error, don't have a reference to image window, cannot send")

    def callExit(self):
        self.close()

    def mouseClickEvent(self,ev):
        '''The mouse click event modification for capturing a polygon.
            Note : This could also be done with a
            QtCore.QEventLoop() object. Just connect the program to
            quit member function and run the exec member function
            of the EventLoop. I decided against this but something
            like this should be implemented (like mouse in yorick)
        '''
        if(self.masking == 1):
            if (ev.button() == QtCore.Qt.RightButton):
                #self.ROI = pg.PolyLineROI((x,y))
                #self.plot.addItem(self.ROI)
                modifiers = QtGui.QApplication.keyboardModifiers()
                if modifiers == QtCore.Qt.ShiftModifier:
                    self.points.append(self.points[0])
                    self.endPolyCapture()
                    #print "got a right click"
            if (ev.button() == QtCore.Qt.LeftButton):
                modifiers = QtGui.QApplication.keyboardModifiers()
                if modifiers == QtCore.Qt.ShiftModifier:
                    #print('Shift+Click')
                    #pt = self.view.mapSceneToView(ev.pos())#map from scene to imageView
                    pt = ev.pos()
                    self.points.append(pt)
                    pt = []
                    #print "got a left click"
            if(hasattr(self,'points')):
                x, y = [], []
                for pt in self.points:
                    x.append(pt.x());
                    y.append(pt.y());

            #self.pROI.setPoints((x,y))
            if(hasattr(self,"line")): #in case this runs once after
                #masking is finished
                self.line.setData(x=x, y=y)

    def polymask(self):
        '''Take img and select regions whose points are within
         polygon.
        NOTE: Polygon should be a closed polygon.
            Polygons are assumed to be N x 2 arrays
            where N is the number of vertices. This was translated
            from Yorick. There exist routines for mask creation
            but none seemed satisfactory.'''
        points = np.array(self.points).astype(int)
        for i in np.arange(0,points.shape[0]-1):
            self.swath(np.array([points[i,],points[(i+1)%points.shape[0],]]))


    def swath(self,seg):
        ''' Takes points to the right of segment and complements
            the values.
        '''
        if(seg[0,1] > seg[1,1]):
            seg = np.roll(seg,2)
        dy = seg[1,1]-seg[0,1]
        if(dy != 0):
            dx = seg[1,0]-seg[0,0]
            ys = (np.arange(seg[0,1],seg[1,1])[np.newaxis,:])
            xs = (np.arange(min(seg[:,0]),self.mask.shape[0])[:,np.newaxis])
            #print xs.shape,ys.shape
            self.mask[np.ix_(xs[:,0],ys[0,:])] ^= (ys*dx < xs*dy+seg[1,0]*seg[0,1]-seg[0,0]*seg[1,1])
            #self.mask[np.ix_(xs[:,0],ys[0,:])] ^= np.cross(
            self.mimg.setImage(self.mask,xvals=np.linspace(0.,.5,self.mask.shape[0]))

        def mouseMoved(self,evt):
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            if self.plotw.sceneBoundingRect().contains(pos):
                mousePoint = vb.mapSceneToView(pos)
                index = int(mousePoint.x())
                self.poslabel.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y=%0.1f</span>" % (mousePoint.x(), mousePoint.y()))
                #vLine.setPos(mousePoint.x())
                #hLine.setPos(mousePoint.y())
