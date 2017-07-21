from PyQt5 import QtGui, QtCore

import pyqtgraph as pg
from tools.smooth import smooth2Dgauss
from gui.ImageWindow import ImageWindow

import numpy as np
from gui.colormaps import makeALBULACmap

from gui.guitools import findLowHigh

MAXCNTS = 10000

class CentralWidget(QtGui.QWidget):
    ''' This is the central widget that contains the image.
        
    '''
    def __init__(self,verbose=False):
        super(CentralWidget, self).__init__()
        self.initUI()
        self.gridding = 1
        self.smoothing = 0
        self.imgdata = None # the image data
        self.imgdata_processed = None # the processed image data
        self.mask = None
        self.verbose = verbose
        self.maxcts = MAXCNTS #hard coded reasonable number for max cnts
        #self.statswin = StatsWindow(self,verbose=verbose)
        #self.statswin.linkItem("gridding",dtype='float')
        #self.statswin.linkItem("smoothing",dtype='int')
        #self.statswin.setGeometry(200,200,400,200)
        #self.statswin.show()
        

    def initUI(self):
        layout_hbox = QtGui.QHBoxLayout()
        layout_spacing = 50 # in pixels
        layout_hbox.addSpacing(layout_spacing)
        #prepare the central image region
        self.image_imv = ImageWindow()
        self.image_imv.setColorMap(makeALBULACmap())
        layout_hbox.addWidget(self.image_imv)
        layout_hbox.addSpacing(layout_spacing)

        text_sigmatext = QtGui.QLabel("smooth sigma: ")
        self.sigmainput = QtGui.QLineEdit()
        self.sigmainput.setMaxLength(10)
        self.sigmainput.setMaximumWidth(50)
        self.sigmainputbutton = QtGui.QPushButton("set")
        self.sigmainputbutton.clicked.connect(self.changeSmoothing)

        gridtext = QtGui.QLabel("smooth grid: ")
        self.gridinput = QtGui.QLineEdit()
        self.gridinput.setMaxLength(10)
        self.gridinput.setMaximumWidth(50)
        self.gridinputbutton = QtGui.QPushButton("set")
        self.gridinputbutton.clicked.connect(self.changeGridding)

        redrawbutton = QtGui.QPushButton("redraw")
        redrawbutton.clicked.connect(self.redrawimg)

        vbox = QtGui.QVBoxLayout()
        vbox.addSpacing(layout_spacing)
        vbox.addLayout(layout_hbox)

        self.imgslider = QtGui.QSlider(0x01)
        hboxslider = QtGui.QHBoxLayout()
        hboxslider.addWidget(self.imgslider)
        vbox.addLayout(hboxslider)

        self.imgslider.setMaximum(0)
        self.imgslider.setMinimum(0)

        gridlbox = QtGui.QGridLayout()
        gridlbox.addWidget(self.imgslider)

        gridlbox.addWidget(text_sigmatext,1,0)
        gridlbox.addWidget(self.sigmainput,1,1)
        gridlbox.addWidget(self.sigmainputbutton,1,2)

        gridlbox.addWidget(gridtext,2,0)
        gridlbox.addWidget(self.gridinput,2,1)
        gridlbox.addWidget(self.gridinputbutton,2,2)

        gridlbox.addWidget(redrawbutton,3,0)

        gridlbox.setColumnStretch(4,1)

        vbox.addLayout(gridlbox)

        vbox.addSpacing(layout_spacing)

        self.setLayout(vbox)

    def changeGridding(self):
        ''' read value from self.gridinput and set gridding to that.'''
        self.setGridding(int(self.gridinput.text()))

    def changeSmoothing(self):
        ''' read value from self.sigmainput and set sigma to that.'''
        self.setSmoothing(float(self.sigmainput.text()))

    def setGridding(self,gridval):
        self.gridding = float(gridval)
        print("Changed griding to {}".format(gridval))

    def setSmoothing(self,smoothval):
        self.smoothing = float(smoothval)
        print("Changed smoothing to {}".format(smoothval))

    def setCentralImage(self, img, levels=None):
        ''' Set the central image.'''
        if levels is None:
            levels = (0,100)
        self.imgdata = img
        self.imgslider.setMaximum(img.shape[0])
        self.redrawimg()

    def setmask(self, mask):
        self.mask = mask.astype(float)
        self.redrawimg()

    def regridimg(self,img):
        ''' regrid image. a quick trick. reshape array into higher dimensions
            and average those dimensions to take advantage of numpy's fast routines.'''
        if img is None:
            print("Sorry can't do anything")
        elif self.gridding is not None:
            # regrid only is the gridding factor is not None
            #y0, y1, x0, x1 where x is fastest varying dimension
            img = self.regrid(img,self.gridding)

        return img

    def regrid(self,img,grd):
        dims = img.shape
        newdims = img.shape[0]//grd, img.shape[1]//grd
        ind = [0, newdims[0]*grd, 0, newdims[1]*grd]
        img = img[ind[0]:ind[1],ind[2]:ind[3]].reshape((newdims[0],grd,newdims[1],grd))
        img = np.average(np.average(img,axis=3),axis=1)
        return img

    def smoothimg(self,img):
        ''' smooth the image.'''
        if img is not None:
            if self.smoothing > 0:
                img_processed = smooth2Dgauss(img.astype(float), mask=self.mask,sigma=self.smoothing)
            else:
                img_processed = img
        else:
            img_processed = None
        return img_processed

    def redrawimg(self):
        self.imgdata_processed = self.smoothimg(self.imgdata)
        self.imgdata_processed = self.regridimg(self.imgdata_processed)
        # for regrid, mask also needs processing
        if self.mask is not None:
            self.mask_processed = self.regridimg(self.mask==0)==0
            self.imgdata_processed *= self.mask_processed
        #self.imgdata_processed = self.imgdata
        if self.imgdata_processed is not None:
            low, high = findLowHigh(self.imgdata_processed,maxcts=self.maxcts)
            levels = (low, high)
            self.image_imv.setImage(self.imgdata_processed,levels=levels)
            self.image_imv.setHistogramRange(low, high)
            self.image_imv.show()
