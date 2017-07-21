from PyQt5 import QtGui, QtCore

import pyqtgraph as pg
from tools.smooth import smooth2Dgauss
from gui.ImageWindow import ImageWindow, ImageMainWindow

import numpy as np
from gui.colormaps import makeALBULACmap

from gui.guitools import findLowHigh

from tools.circavg import circavg
from tools.qphiavg import qphiavg
from tools.qphicorr import deltaphicorr, deltaphicorr_qphivals, normsqcphi

from gui.ImageAnalWindow import ImageAnalWindow

from skbeam.core.roi import circular_average


def mouseMoved(ev):
    print(ev)

class SAXSWidget(QtGui.QWidget):
    ''' This widget allows the display of an average image, as well
            as tweaking some parameters, like smoothing or binning.

        It also contains the necessary routines for averaging images etc.
    '''
    def __init__(self, saxsdata, verbose=False):
        super(SAXSWidget, self).__init__()
        self.saxsdata = saxsdata
        self.initUI()
        self.gridding = 1
        self.smoothing = 0
        self.imgdata = None # the image data
        self.imgdata_processed = None # the processed image data
        self.mask = None
        self.verbose = verbose
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
        pg.SignalProxy(self.image_imv.scene.sigMouseMoved,rateLimit=60,slot=mouseMoved)
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
        self.image_imv.show()

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

    def regridimg(self,img):
        ''' regrid image. a quick trick. reshape array into higher dimensions
            and average those dimensions to take advantage of numpy's fast routines.'''
        if img is None:
            print("Sorry can't do anything")
        elif self.gridding is not None and self.gridding > 1:
            # regrid only is the gridding factor is not None
            #y0, y1, x0, x1 where x is fastest varying dimension
            img = self.regrid(img,self.gridding)

        return img

    def regrid(self,img,grd):
        if grid > 1:
            print("regridding")
            dims = img.shape
            newdims = img.shape[0]//grd, img.shape[1]//grd
            ind = [0, newdims[0]*grd, 0, newdims[1]*grd]
            img = img[ind[0]:ind[1],ind[2]:ind[3]].reshape((newdims[0],grd,newdims[1],grd))
            img = np.average(np.average(img,axis=3),axis=1)
        return img

    def smoothimg(self,img):
        ''' smooth the image.'''
        if img is not None:
            if self.smoothing > 1e-6:
                img_processed = smooth2Dgauss(img.astype(float), mask=self.mask,sigma=self.smoothing)
            else:
                img_processed = img
        else:
            img_processed = None
        return img_processed

    def redrawimg(self):
        ''' Redraw the image, only if the avg_img exists in saxsdata.'''
        if hasattr(self.saxsdata, "avg_img") and self.saxsdata.avg_img is not None:
            self.imgdata_processed = self.smoothimg(self.saxsdata.avg_img)
            self.imgdata_processed = self.regridimg(self.imgdata_processed)
            if self.imgdata_processed is not None:
                # for regrid, mask also needs processing
                if hasattr(self.saxsdata, "mask") and self.saxsdata.mask is not None:
                    if self.saxsdata.mask is not None:
                        self.mask_processed = self.regridimg(self.saxsdata.mask)
                        self.imgdata_processed *= self.mask_processed
                #self.imgdata_processed = self.imgdata
                print("redrawing")
                low, high = findLowHigh(self.imgdata_processed,maxcts=self.saxsdata.maxcts)
                levels = (low, high)
                self.image_imv.setImage(self.imgdata_processed.T,levels=levels)
                self.image_imv.setHistogramRange(low, high)
                self.image_imv.show()


    def removenans(self,data):
        ''' set nan regions to zero'''
        w = np.where(np.isnan(data))
        data[w] = 0

    def removeinfs(self,data):
        ''' set nan regions to zero'''
        w = np.where(np.isinf(data))
        data[w] = 0

    # Analysis and plotting stuff
    def circavg(self):
        ''' calculate and plot circular average.'''
        if hasattr(self.saxsdata,"mask") and self.saxsdata.mask is not None:
            mask = self.saxsdata.mask
        else:
            mask = None
        #sqx, sqy = circavg(self.saxsdata.avg_img, x0=self.saxsdata.getxcen(), y0=self.saxsdata.getycen(),\
                           #noqs=self.saxsdata.getnoqs_circavg(), mask=mask)
        x0 = self.saxsdata.getxcen()
        y0 = self.saxsdata.getycen()
        noqs = self.saxsdata.getnoqs_circavg()
        sqx, sqy = circular_average(self.saxsdata.avg_img,(y0,x0),mask=mask,nx=noqs) 
        p1 = pg.plot(sqx*self.saxsdata.getqperpixel(),sqy)
        p1.getPlotItem().setLogMode(True,True)

    def qphimap(self):
        ''' Makes the qphi map for the data.'''
        saxsdata = self.saxsdata
        if hasattr(saxsdata,"mask") and self.saxsdata.mask is not None:
            mask = saxsdata.mask
        else:
            mask = None
        data = saxsdata.avg_img
        # grab some data from the table
        x0, y0 = saxsdata.getxcen(), saxsdata.getycen()
        noqs,nophis = saxsdata.getnoqs_qphiavg(), saxsdata.getnophis_qphiavg()
        saxsdata.sqphi = qphiavg(data, mask=mask, x0=x0, y0=y0,noqs=noqs,nophis=nophis)
        #p1 = ImageMainWindow(parent=self,img=saxsdata.sqphi, maxcts=self.saxsdata.maxcts,lockAspect=False)
        # need to keep the reference or the window dies
        self.p1 = ImageAnalWindow(saxsdata.sqphi)
        #p1 = ImageAnalWindow(saxsdata.sqphi)
#        low, high = findLowHigh(saxsdata.sqphi,maxcts=self.saxsdata.maxcts)
#        levels = (low, high)
#        p1 = pg.image(saxsdata.sqphi.T,levels=levels)
#        p1.setHistogramRange(low,high)
#        p1.setColorMap(makeALBULACmap())
#        p1.view.setAspectLocked(False)

    def deltaphicorr(self):
        ''' Makes the delta phi correlation for the data.'''
        saxsdata = self.saxsdata
        if hasattr(saxsdata,"mask") and self.saxsdata.mask is not None:
            mask = saxsdata.mask
        else:
            mask = None
        data = saxsdata.avg_img
        # grab some data from the table
        x0, y0 = saxsdata.getxcen(), saxsdata.getycen()
        noqs,nophis = saxsdata.getnoqs_deltaphicorr(), saxsdata.getnophis_deltaphicorr()
        saxsdata.sqcphi = deltaphicorr(data, data, noqs=noqs, nophis=nophis, x0=x0,\
                                    y0=y0, mask=mask, symavg=True)
        saxsdata.sqcphin = normsqcphi(saxsdata.sqcphi)
        self.removenans(saxsdata.sqcphin)
        self.removeinfs(saxsdata.sqcphin)
        #p1 = ImageMainWindow(parent=self,img=saxsdata.sqcphin, maxcts=self.saxsdata.maxcts,levels=(0,1),lockAspect=False)
        #p1 = ImageAnalWindow(saxsdata.sqcphin)
        self.p2 = ImageAnalWindow(saxsdata.sqcphin,levels=(0,1))
        #p1 = pg.image(saxsdata.sqcphin.T,levels=(0.,1.))
        #p1.setColorMap(makeALBULACmap())
        #p1.view.setAspectLocked(False)
        #p1.setHistogramRange(0,1.)
