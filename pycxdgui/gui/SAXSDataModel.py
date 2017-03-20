from PyQt4 import QtGui, QtCore
import sys

#Qt Framework is to make models and let views view models

'''from 
http://stackoverflow.com/questions/31475965/fastest-way-to-populate-qtableview-from-pandas-data-frame
and it's a pandas dataframe
'''

class SAXSDataModel(QtCore.QAbstractTableModel):
    def __init__(self, saxsobj, parent=None):
        ''' data should be the SAXS object
        '''
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._saxsobj = saxsobj

    def rowCount(self, parent=None):
        return self._saxsobj.rowCount()

    def columnCount(self, parent=None):
        # always two columns for dict
        return 2

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                row = self._saxsobj.getrow(index.row())
                #print(str(row[1]))
                if index.column() == 0:
                    return str(row[0])
                elif index.column() == 1:
                    if row[1] is not None:
                        return str(row[1])
        # if none of these, return None
        return None

    def flags(self, index):
        #return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        return QtCore.QAbstractTableModel.flags(self, index) | QtCore.Qt.ItemIsEditable 

    def setData(self, index, val, role=QtCore.Qt.EditRole):
        if index.isValid():
            row = self._saxsobj.getrow(index.row())
            if index.column() == 1:
                self._saxsobj.set_param(row[0], val)
                return True
            else:
                print("Error, cannot edit first column, not changing value")
                
        return False
