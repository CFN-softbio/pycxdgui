from PyQt5 import QtGui, QtCore
import sys
import os


if __name__ == "__main__":
    # Necessary for a pyQt application to work
    print("starting GUI")
    app = QtGui.QApplication(sys.argv)
    splitter = QtGui.QSplitter()
    model = QtGui.QFileSystemModel()
    model.setRootPath(os.getcwd())

    tree = QtGui.QTreeView(splitter)
    tree.setModel(model)
    tree.setRootIndex(model.index(QtCore.QDir.currentPath()))

    lst = QtGui.QListView(splitter)
    lst.setModel(model)
    lst.setRootIndex(model.index(QtCore.QDir.currentPath()))

    splitter.setWindowTitle("File Dialog")
    splitter.show()

    sys.exit(app.exec_())
