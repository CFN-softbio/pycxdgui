import pyqtgraph as pg
import numpy as np
from PyQt4 import QtGui, QtCore

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree,\
        ParameterItem, registerParameterType

from tools.optics import calc_q

class DataTree(QtGui.QWidget):
    ''' Takes a list of dictionaries (can be nested) of triplets
        which have keys 'name', 'type' and 'value' (or 'children' if 'type'=group
        'value' is the default value
    '''
    def __init__(self, params):
        super(QtGui.QWidget, self).__init__()
        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(QtGui.QLabel("Parameters"), 0,  0, 1, 2)

        self.params = params

        # create the Parameter Tree
        self.p = Parameter.create(name='Parameters', type='group', children=self.params)

        self.p.sigTreeStateChanged.connect(self.change)

        self.t = ParameterTree()
        self.t.setParameters(self.p, showTop=False)
        self.t.setWindowTitle('Parameter Tree')

        self.layout.addWidget(self.t, 1, 0, 1, 1)
        self.show()
        self.resize(800,800)

    def change(self, param, changes):
        pass
        #print("tree changes:")
        #print(param)
        #for param, change, data in changes:
            #path = self.p.childPath(param)
            #if path is not None:
                #childName = '.'.join(path)
            #else:
                #childName = param.name()
            #print('  parameter: %s'% childName)
            #print('  change:    %s'% change)
            #print('  data:      %s'% str(data))
            #print('  ----------')


# this makes the parameters for the SAXS Object
setuplst =  [
        ['xcen', 'float', 0],
        ['ycen', 'float', 0],
        ['extension', 'str', '.tif'],
        ['DDIR', 'str', '.'],
        ['SDIR', 'str', '../storage'],
        ['mask_name', 'str', 'None'],
        ['mask_threshold', 'float', '1'],
        ['wait_time', 'float', '.1'],
        ['rdet', 'float', np.nan],
        ['dpix', 'float', np.nan],
        ['wavelength', 'float', np.nan],
        ['energy', 'float', np.nan],
        ['qperpixel', 'float', np.nan],
        ['filename', 'str', ""],
    ]

intanallst = [
    ['group', 'int', 1], #group frames together
        ]

sqlst = [
        ['noqs', 'int', 1000],
        ['nophis', 'int', 360],
        ]

sqphilst = [
        ['noqs', 'int', 1000],
        ['nophis', 'int', 360],
        ]

deltaphicorrlst = [
        ['noqs', 'int', 1000],
        ['nophis', 'int', 360],
        ]

imganallst = [
        ['smoothing', 'float', 0.],
        ['gridding', 'int', 1]
]

def dictfromparams(paramslst):
    dictlst = list()
    for entry in paramslst:
        dictlst.append({'name' : entry[0], 'type' : entry[1], 'value' : entry[2]})
    return dictlst

params = [
        {'name' : 'setup', 'type' : 'group', 'children' : dictfromparams(setuplst)},
        {'name' : 'intanal', 'type' : 'group', 'children' : dictfromparams(intanallst)},
        {'name' : 'circavg', 'type' : 'group', 'children' : dictfromparams(sqlst)},
        {'name' : 'qphiavg', 'type' : 'group', 'children' : dictfromparams(sqphilst)},
        {'name' : 'deltaphicorr', 'type' : 'group', 'children' : dictfromparams(deltaphicorrlst)},
        {'name' : 'imganal', 'type' : 'group', 'children' : dictfromparams(imganallst)},
        ]

class SAXSDataTree(DataTree):
    def __init__(self,configfile=None):
        super(SAXSDataTree, self).__init__(params)
        self.params = params
        if configfile is not None:
            self.loadfromcsv(configfile)
        # now calculate typical SAXS stuff
        a = float(self.getelem("setup","dpix"))
        L = float(self.getelem("setup","rdet"))
        wv = float(self.getelem("setup","wavelength"))
        if a is not None and L is not None and wv is not None:
            qperpixel = calc_q(L,a,wv)
        self.setelem("setup", "qperpixel", qperpixel)

    # these should be moved to DataTree

    def loadfromcsv(self,configfile):
        ''' Load from a csv config file
            setup, noqs, value
            etc
        '''
        entries = self.file2list(configfile)
        for entry in entries:
            self.setelem(entry[0], entry[1], entry[2])

    def setelem(self, parent, name, value):
        ''' Set element, for example:
                setup, noqs, 1
        '''
        try:
            if hasattr(self.p, "child"):
                # VERSION for backwards compatibility with pyqtgraph version 0.9.9
                par1 = self.p.child(parent)
                par2 = par1.child(name)
                par2.setValue(value)
            else:
                par1 = self.p.param(parent)
                par2 = par1.param(name)
                par2.setValue(value)
        except Exception:
            print("Could not find value. Ignoring")

    def getelem(self,parent,name):
        try:
            if hasattr(self.p, "child"):
                # VERSION for backwards compatibility with pyqtgraph version 0.9.9
                par1 = self.p.child(parent)
                par2 = par1.child(name)
                return par2.value()
            else:
                par1 = self.p.param(parent)
                par2 = par1.param(name)
                return par2.value()
        except Exception:
            print("Could not find value, returning None")
            return None

    def file2list(self,filename):
        ''' Transforms a SAXS config file to a dictionary of dictionaries.
            Main purpose is for parameters here.
            Syntax is basically a csv file, first two columns read, rest ignored
            If line starts with #, line is ignored
        '''
        #  get reference to dict
        entries = list()
        try:
            with open(filename) as f:
                for line in f:
                    if line[0] is not "#":
                        # tokenize = and remove whitespace
                        str1 = line.replace(" ", "").replace("\"","").replace("\n","").split(",")
                        if len(str1) > 2:
                            entries.append([str1[0], str1[1], str1[2]])
        except IOError:
            pass

        return entries

    # these are more SAXS stuff, shortcuts for common commands
    def getqperpixel(self):
        return float(self.getelem("setup","qperpixel"))

    def getrdet(self):
        return float(self.getelem("setup","rdet"))

    def getwavelength(self):
        return float(self.getelem("setup","wavelength"))

    def getdpix(self):
        return float(self.getelem("setup","dpix"))

    def getxcen(self):
        return float(self.getelem("setup","xcen"))

    def getycen(self):
        return float(self.getelem("setup","ycen"))

    def getDDIR(self):
        return self.getelem("setup","DDIR")

    def getSDIR(self):
        return self.getelem("setup","SDIR")

    def getfilename(self):
        return self.getelem("setup","filename")

    def getmask_name(self):
        return self.getelem("setup","mask_name")

    def getnoqs_circavg(self):
        return int(self.getelem("circavg","noqs"))

    def getnoqs_qphiavg(self):
        return int(self.getelem("qphiavg","noqs"))

    def getnophis_qphiavg(self):
        return int(self.getelem("qphiavg","nophis"))

    def getnoqs_deltaphicorr(self):
        return int(self.getelem("deltaphicorr","noqs"))

    def getnophis_deltaphicorr(self):
        return int(self.getelem("deltaphicorr","nophis"))

    def setmask_name(self, mask_name):
        return self.setelem("setup","mask_name", mask_name)

