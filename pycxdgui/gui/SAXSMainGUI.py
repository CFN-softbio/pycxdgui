from detector.eiger import EigerImages
from PIL import Image
#from tifffile import TiffFile

from tools.average import runningaverage
from tools.mask import openmask
#from tools.SAXS import SAXSObj
#to read descriptor files
from tools.circavg import circavg
from tools.qphiavg import qphiavg

from skbeam.core.roi import circular_average

import tools.mask

from gui.SAXSWidget import SAXSWidget
from gui.Masking import MPoly
from gui.SAXSDataModel import SAXSDataModel

from gui.DataTree import SAXSDataTree

from gui.colormaps import makeALBULACmap
from gui.FileListener import FileListener

import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtCore

import os
import os.path

import pandas as pd
import numpy as np


# some hard coded values
MAXCTS = 65535 # for proper image color scale display, ignore high values

''' Structure of GUI:
    The main GUI has buttons which each call separate widgets that perform
    various functions. They should be independent of the main GUI.

    There is one central widget in focus and it could be eventually changed.

    Various data elements must be connected to the GUI and the other windows.
    So far the list is:
        -SAXSDataTree : SAXS data analysis tree

'''

# the different extensions accepted
_file_filters = {
        '.tif' : 'tif File (*.tif *tiff)',
        'master.h5' : 'EigerImage File (*master.h5)',
        'all' : 'All Files (*)'
        }

class SAXSGUI(QtGui.QMainWindow):
    ''' Uses QMainWindow, has builtin features like menu bar, toolbars
            and dockable toolbars. I will add a central widget to the
            window which will have its own layout.'''

    # metaclassed, needs to be here
    signal_imageupdated = pyqtSignal()#name="imageupdated()")

    def __init__(self, configfile=None, verbose=False):
        ''' This starts the SAXS GUI.'''
        super(SAXSGUI, self).__init__()

        self.imgreader = None
        self.avg_img, self.Ivst = None, None
        self.verbose = verbose
        # data stuff
        self.configfile = configfile
        self.saxsdata = SAXSDataTree(configfile)
        self.load_img()
        self.average_frames()
        self.load_mask()

        # needs to move away from this eventually
        self.maxcts = MAXCTS

        self.verbose = verbose

        self.numentries = 0 # number of data entries analyzed here
        wait_time = float(self.saxsdata.getelem("setup","wait_time"))

        # start a file listener using the specified data directory
        ddir = self.saxsdata.getelem("setup","DDIR")
        extension = self.saxsdata.getelem("setup","extension")
        self.filelistener = FileListener(wait_time=wait_time)
        self.filelistener.listen_for_files(ddir, extension)

        self.listen_for_newfiles = True

        # UI stuff
        self.initUI()
        self.filelistener.signal_newerfile.connect(self.load_newfile)
        self.signal_imageupdated.connect(self.reprocess_and_draw)

    def initUI(self):
        ''' Initialize the user interface.
            Initializes the main window.
        '''
        self.setGeometry(300,200,1200,800)
        self.statusBar().showMessage('Ready')
        menubar = self.menuBar()

        # icon, shortcut, description, action (function)
        loadEAction = self.mkAction('icons/loadE.png', '&Open EIGER File', 
                                    None, 'Open EIGER File', self.openmasterfile)
        loadMAction = self.mkAction('icons/loadM.png', '&Open Mask File', 
                                    None, 'Open Mask File', self.openmaskfile)
        exitAction = self.mkAction('icons/quit.png', '&Exit', 'Ctrl+W', 
                                   'Exit Application', QtWidgets.qApp.quit)
        maskAction = self.mkAction('icons/mpolyicon.png', '&Make Mask', 
                                   None, 'Make new mask', self.startmasking)
        dataTableAction = self.mkAction('icons/datatable.png', 'View Data Table', 
                                   None, 'View Data Table', self.showDataTable)
        circAvgAction = self.mkAction('icons/circavg.png', 'Plot &Circular Average', 
                                   None, 'Plot Circular Average', self.circavg)
        sqphiAction = self.mkAction('icons/sqphi.png', 'Plot Qphi Map', 
                                   None, 'Plot QPhi Map', self.qphimap)
        deltaphicorrAction = self.mkAction('icons/deltaphicorr.png', 'Plot Delta Phi Corr Map', 
                                   None, 'Plot Delta Phi Corr Map', self.deltaphicorr)
        listenToggleAction = self.mkAction('icons/listen.png', 'Toggle listen', 
                                   None, 'Toggle listen', self.toggle_listen_for_newfiles)

        aspectToggleAction = self.mkAction('icons/aspect.png', 'Lock/Unlock Aspect Ratio', 
                                   None, 'Lock/Unlock Aspect Ratio', self.toggle_aspect)

        # menu bar
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(loadEAction)
        filemenu.addAction(loadMAction)
        filemenu.addAction(exitAction)
        filemenu.addAction(maskAction)
        filemenu.addAction(dataTableAction)
        filemenu.addAction(circAvgAction)
        filemenu.addAction(sqphiAction)
        filemenu.addAction(deltaphicorrAction)
        filemenu.addAction(listenToggleAction)
        filemenu.addAction(aspectToggleAction)

        # tool bar
        self.toolbar = self.addToolBar("EIGER quick access")
        self.toolbar.addAction(loadEAction)
        self.toolbar.addAction(loadMAction)
        self.toolbar.addAction(exitAction)
        self.toolbar.addAction(maskAction)
        self.toolbar.addAction(dataTableAction)
        self.toolbar.addAction(circAvgAction)
        self.toolbar.addAction(sqphiAction)
        self.toolbar.addAction(deltaphicorrAction)
        self.toolbar.addAction(listenToggleAction)
        self.toolbar.addAction(aspectToggleAction)

        self.setWindowTitle("EIGER Data Visualizor")

        #self.showDataTable()

        self.imgwidget = SAXSWidget(self) #spawns the ui for the image etc
        self._aspectlock = True
        self.imgwidget.setAspectLock(self._aspectlock)
        self.setCentralWidget(self.imgwidget)
        # initial draw
        self.imgwidget.redrawimg()

        self.show()

    def toggle_aspect(self):
        self._aspectlock = not self._aspectlock
        self.imgwidget.setAspectLock(self._aspectlock)

    def showDataTable(self):
        ''' Show the data table'''
        pass
        #self.saxsdata = SAXSDataTree()
        #self.datatableview = QtGui.QTableView()
        #self.datatablemodel = SAXSDataModel(self.saxsobj)

        #self.datatableview.setModel(self.datatablemodel)

        #self.datatableview.setGeometry(600,400,600,600)
        #self.datatableview.show()

    def startmasking(self):
        if self.mask is None:
            mask = None
        else:
            mask = self.mask.astype(float)
        # TODO : make general fucntion
        if self.avg_img is None:
            self.avg_img = np.ones((100,100))
        self.maskprog = MPoly(self.avg_img.astype(float),mask=None, imgwidget=self.imgwidget)

    def circavg(self):
        self.imgwidget.circavg()

    def qphimap(self):
        self.imgwidget.qphimap()

    def deltaphicorr(self):
        self.imgwidget.deltaphicorr()

    def mkAction(self, icon, txt, shrtcut, stattip, action):
        ''' Quit routine to make actions and add them to this GUI.'''
        if icon is not None:
            myAction = QtGui.QAction(QtGui.QIcon(icon), txt, self)
        else:
            myAction = QtGui.QAction(txt, self)
        if shrtcut is not None:
            myAction.setShortcut(shrtcut)
        if stattip is not None:
            myAction.setStatusTip(stattip)
        if action is not None:
            myAction.triggered.connect(action)
        return myAction

    def openmasterfile(self):
        file_filters = _file_filters.copy()
        # re order filters to have chosen extension first
        filt_string = ""
        extension = self.saxsdata.getelem("setup", "extension")
        # now make filters string (could be sep function)
        first = True
        if extension in file_filters:
            filt_string = filt_string + file_filters.pop(extension)
            first = False

        for key in file_filters:
            if first:
                first = False
            else:
                filt_string = filt_string + ";;"
            filt_string = filt_string + file_filters[key]

        filename, filetype = QtGui.QFileDialog.getOpenFileName(self, 'Open data file', self.getDDIR(), filt_string)
        if(len(filename)):
            print("opening file {}".format(filename))
            self.load_img(filename=filename)
            print("Computing a running average of the images ({} images)...".format(len(self.imgreader)))
            self.average_frames()
            print("done")
            self.imgwidget.redrawimg()

    def openmaskfile(self):
        fname, filetype = QtGui.QFileDialog.getOpenFileName(self, 'Open Mask', self.getSDIR(),
                            "Masks (*mask*.hd5 *tif *tiff);;Blemish Files (*blemish*.hd5);; All Files (*)")
        self.loadmaskfromfile(fname)

    def loadmaskfromfile(self, fname):
        self.load_mask(fname)
        self.imgwidget.redrawimg()

    def start(self):
        self.top.mainloop()

    def toggle_listen_for_newfiles(self):
        if self.listen_for_newfiles is False:
            #reset (should access memeber functions need to check threading)
            self.filelistener.curfilename = None
        self.listen_for_newfiles = not self.listen_for_newfiles
        print("Toggled the file listening, listening is now {}".format(self.listen_for_newfiles))

    def load_newfile(self,filename):
        ''' Wrapper to load new file only if listen is on.'''
        if self.listen_for_newfiles:
            self.load_img(filename)
        # else do nothing

    # the functions for the buttons in the Main GUI
    def load_img(self, filename=None):
        ''' Try to load the data. If the filename is not specified,
            it will look for filename already stored in object.
            If filename is specified, then the object will be updated
            with this filename.
            '''
        if filename is None:
            filename = self.saxsdata.getelem("setup","filename")
        else:
            # the store new name in object
            self.saxsdata.setelem("setup","filename",filename)

        if filename is not None:
            if "master.h5" in filename:#.endswith("master.h5"):
                try:
                    #ddir = self.saxsdata.getelem("setup","DDIR")
                    self.imgreader = EigerImages(filename)
                except IOError:
                    print("Error could not read file")
                    return False
            elif "tif" in filename or "tiff" in filename:
                try:
                    #ddir = self.saxsdata.getelem("setup","DDIR")
                    self.imgreader = np.array(Image.open(filename))
                    #self.imgreader = TiffFile(filename).asarray()
                    if len(self.imgreader.shape) == 3:
                        self.imgreader = np.average(self.imgreader,axis=2)
                    # then make 3D t series (of one image)
                    self.imgreader = np.array([self.imgreader])
                except IOError:
                    print("Error could not read file")
                    return False
            else:
                print("Error, could not read image {} with known extensions (master.h5 or tif supported)".format(filename))
                self.imgreader = None
                return False
        else:
            print("Did not load any images. Make sure filename and parent directory DDIR are set: {}".format(filename))
            return False

        self.signal_imageupdated.emit()
        return True

    def reprocess_and_draw(self):
        self.average_frames()
        self.load_mask()
        self.imgwidget.redrawimg()


    def load_mask(self, mask_name=None):
        if mask_name is None:
            mask_name = self.getmask_name()
        else:
            self.setmask_name(mask_name)

        if mask_name is not None:
            print("Opening mask: {}".format(mask_name))
            mask_threshold = int(self.saxsdata.getelem("setup", "mask_threshold"))
            if mask_name.endswith("h5") or mask_name.endswith("hd5"):
                # premask is mask, and mask is thresholded (useful when using a flatfield)
                self.premask = tools.mask.openmask(mask_name)
            elif mask_name.lower().endswith("tif") or mask_name.lower().endswith("tiff"):
                # save two instances of mask
                self.premask = np.array(Image.open(mask_name))
                self.mask = (self.premask >= mask_threshold).astype(int)
            else:
                print("Warning, unsupported extension, ignoring...")
                self.premask = None

            if self.premask is not None:
                self.mask = (self.premask >= mask_threshold).astype(int)
            else:
                self.mask = None
            if hasattr(self, 'imgwidget'):
                self.imgwidget.mask = self.mask


    def average_frames(self):
        if self.imgreader is not None:
            self.avg_img, self.Ivst = runningaverage(self.imgreader)
            if self.mask is not None:
                self.avg_img = self.avg_img
        else:
            print("Cannot average images, no images loaded")

    def compute_sq(self):
        self.sqx,self.sqy = circavg(self.avg_img,x0=self.getxcen(),\
                            y0=self.getycen(),mask=self.mask)

    def calcq(self):
        '''calculate the q perpixel for the saxs data. '''
        self.qperpixel = calc_q(self.getrdet(),self.getdpix(),self.getwavelength())
        self.saxsdata.setelem("setup","qperpixel",self.qperpixel)

    # convenience functions for data retrieval from saxsdata (should be
    # improved or separated) figure out data typing output later
    def getqperpixel(self):
        return float(self.saxsdata.getelem("setup","qperpixel"))

    def getrdet(self):
        return float(self.saxsdata.getelem("setup","rdet"))

    def getwavelength(self):
        return float(self.saxsdata.getelem("setup","wavelength"))

    def getdpix(self):
        return float(self.saxsdata.getelem("setup","dpix"))

    def getxcen(self):
        return float(self.saxsdata.getelem("setup","xcen"))

    def getycen(self):
        return float(self.saxsdata.getelem("setup","ycen"))

    def getDDIR(self):
        return self.saxsdata.getelem("setup","DDIR")

    def getSDIR(self):
        return self.saxsdata.getelem("setup","SDIR")

    def getfilename(self):
        return self.saxsdata.getelem("setup","filename")

    def getmask_name(self):
        return self.saxsdata.getelem("setup","mask_name")

    def getnoqs_circavg(self):
        return int(self.saxsdata.getelem("circavg","noqs"))

    def getnoqs_qphiavg(self):
        return int(self.saxsdata.getelem("qphiavg","noqs"))

    def getnophis_qphiavg(self):
        return int(self.saxsdata.getelem("qphiavg","nophis"))

    def getnoqs_deltaphicorr(self):
        return int(self.saxsdata.getelem("deltaphicorr","noqs"))

    def getnophis_deltaphicorr(self):
        return int(self.saxsdata.getelem("deltaphicorr","nophis"))

    def setmask_name(self, mask_name):
        return self.saxsdata.setelem("setup","mask_name", mask_name)

