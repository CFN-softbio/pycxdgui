from PyQt5 import QtGui, QtCore

from .. import pyqtgraph as pg
from ..tools.smooth import smooth2Dgauss
from ..tools.circavg2 import circavg2
from .ImageWindow import ImageWindow, ImageMainWindow

import numpy as np
from .colormaps import makeALBULACmap

from .guitools import findLowHigh

from ..tools.circavg import circavg
from ..tools.qphiavg import qphiavg
from ..tools.qphicorr import deltaphicorr, deltaphicorr_qphivals, normsqcphi

from .ImageAnalWindow import ImageAnalWindow

from .DataTree import calibration_from_datatree


def mouseMoved(ev):
    print(ev)


class SAXSWidget(QtGui.QWidget):
    ''' This is the central widget that contains the image.
        This also does plotting and crude analysis.
        This analysis should be moved to another object eventually.
    '''
    def __init__(self, saxsdata, verbose=False):
        super(SAXSWidget, self).__init__()
        # saxsdata is GUI, saxsdata.saxsdata is param tree
        self.saxsdata = saxsdata
        self.initUI()
        # check that the smoothing and gridding parameters exist in param tree
        gridding = saxsdata.saxsdata.getelem("imganal", "gridding")
        smoothing = saxsdata.saxsdata.getelem("imganal", "gridding")
        if gridding is None:
            raise ValueError("Error, gridding doesn't exist in parameter tree")
        if smoothing is None:
            raise ValueError("Error, smoothin doesn't exist in parameter tree")
        self.imgdata = None # the image data
        self.imgdata_processed = None # the processed image data
        self.mask = None
        self.verbose = verbose

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

        gridlbox.addWidget(redrawbutton,3,0)

        gridlbox.setColumnStretch(4,1)

        vbox.addLayout(gridlbox)

        vbox.addSpacing(layout_spacing)

        self.setLayout(vbox)
        self.image_imv.show()

    def setAspectLock(self, val):
        self.image_imv.view.setAspectLocked(val)

    def changeGridding(self):
        ''' read value from self.gridinput and set gridding to that.'''
        # change in view input box and also parameter tree (saxsdata)
        gridding = int(self.gridinput.text())
        self.setGridding(gridding)

    def changeSmoothing(self):
        ''' read value from self.sigmainput and set sigma to that.'''
        smoothing = float(self.sigmainput.text())
        self.setSmoothing(smoothing)

    def setGridding(self,gridval):
        self.gridding = float(gridval)
        self.saxsdata.saxsdata.setelem("imganal","gridding", gridval)
        print("Changed griding to {}".format(gridval))

    def setSmoothing(self,smoothval):
        self.smoothing = float(smoothval)
        self.saxsdata.saxsdata.setelem("imganal", "smoothing", smoothval)
        print("Changed smoothing to {}".format(smoothval))

    def regridimg(self,img):
        ''' regrid image. a quick trick. reshape array into higher dimensions
            and average those dimensions to take advantage of numpy's fast routines.'''
        gridding = self.saxsdata.saxsdata.getelem("imganal", "gridding")
        if img is None:
            print("Sorry can't do anything")
        elif gridding is not None and gridding > 1:
            # regrid only is the gridding factor is not None
            #y0, y1, x0, x1 where x is fastest varying dimension
            img = self.regrid(img,gridding)

        return img

    def regrid(self, img, grd):
        if grd > 1:
            print("regridding")
            grd = int(grd)
            dims = img.shape
            newdims = img.shape[0]//grd, img.shape[1]//grd
            ind = [0, newdims[0]*grd, 0, newdims[1]*grd]
            img = img[ind[0]:ind[1],ind[2]:ind[3]].reshape((newdims[0],grd,newdims[1],grd))
            img = np.average(np.average(img,axis=3),axis=1)
        return img

    def smoothimg(self,img):
        ''' smooth the image.'''
        if img is not None:
            smoothing = self.saxsdata.saxsdata.getelem("imganal", "smoothing")
            if smoothing > 1e-6:
                img_processed = smooth2Dgauss(img.astype(float), mask=self.saxsdata.mask,sigma=smoothing)
            else:
                img_processed = img
        else:
            img_processed = None
        return img_processed

    def recomputemask(self):
        ''' recompute the mask '''
#        if hasattr(self.saxsdata, "premask") and self.saxsdata.premask is not None:
#            print("Recomputing mask")
#            mask_threshold = int(self.saxsdata.saxsdata.getelem("setup","mask_threshold"))
#            self.saxsdata.mask = (self.saxsdata.premask >= mask_threshold).astype(int)
#            self.mask_processed = self.regridimg(self.saxsdata.mask)
#        else:
#            self.mask_processed = None
        self.mask_processed = self.mask

    def redrawimg(self):
        ''' Redraw the image, only if the avg_img exists in saxsdata.'''
        self.recomputemask()
        if hasattr(self.saxsdata, "avg_img") and self.saxsdata.avg_img is not None:
            self.imgdata_processed = self.smoothimg(self.saxsdata.avg_img)
            self.imgdata_processed = self.regridimg(self.imgdata_processed)
            if self.imgdata_processed is not None:
                # for regrid, mask also needs processing
                if self.mask_processed is not None:
                    if self.imgdata_processed.shape == self.mask_processed.shape:
                        self.imgdata_processed = self.imgdata_processed*self.mask_processed
                    else:
                        print("Warning: mask is incompatible with image (ignoring mask)" + \
                                    "mask shape : {}".format(self.mask_processed.shape) + \
                                    "img shape : {}".format(self.imgdata_processed.shape)\
                            )
                #self.imgdata_processed = self.imgdata
                print("redrawing")
                low, high = findLowHigh(self.imgdata_processed,maxcts=self.saxsdata.maxcts)
                levels = (low, high)
                # funny bug where (0,1) gives errors (probably has to do with
                # lookup table, but I have limited time)
                if levels == (0,1):
                    levels = 0,10
                # get image transformation
                trans = self.saxsdata.saxsdata.getelem("setup","transformation")
                axis1 = np.argmax(np.abs(trans[0]))
                axis2 = np.argmax(np.abs(trans[1]))
                imgp = self.imgdata_processed
                imgp = np.transpose(imgp, axes=(axis1, axis2))
                # should be +/- 1 else will give error
                # quickly written
                imgp = imgp[::trans[0][axis1], ::trans[1][axis2]]
                self.imgdata_processed = imgp

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
            # TODO : move this to mask loading, reorgnaize mask loading
            mask = self.saxsdata.mask
            if mask.ndim == 3:
                mask = mask[:,:,0]
        else:
            mask = None
        #sqx, sqy = circavg(self.saxsdata.avg_img, x0=self.saxsdata.getxcen(), y0=self.saxsdata.getycen(),\
                            #noqs=self.saxsdata.getnoqs_circavg(), mask=mask)
        # REMOVE ME
        det_height, det_width = self.saxsdata.avg_img.shape
        # TODO : make this more permanent
        print("Computing calibration from saxs data")
        self.calibration = calibration_from_datatree(self.saxsdata.saxsdata, det_width, det_height)
        q_map = self.calibration.q_map
        r_map = self.calibration.r_map
        self.saxsdata.saxsdata.getelem("circavg","noqs")
        print("Computing circular average")
        print("qmap size {}".format(q_map.shape))
        if mask is not None:
            print("mask size {}".format(mask.shape))
        else:
            print("Not using mask")
        print("img size {}".format(self.saxsdata.avg_img.shape))
        sqx, sqy, sqxerr, sqyerr = circavg2(self.saxsdata.avg_img, q_map=q_map,
                                            r_map=r_map, mask=mask)
        #def circavg2(image, q_map=None, r_map=None,  bins=None, mask=None):
        #sqx = sqx*self.saxsdata.getqperpixel()
        p1 = pg.plot(sqx,sqy)
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
                                    y0=y0, mask=mask)
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
