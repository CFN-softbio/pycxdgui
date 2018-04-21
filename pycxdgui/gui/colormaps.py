from pycxdgui import pyqtgraph as pg
import numpy as np

def makeALBULACmap():
    ''' Make ALBULA like color map for
        pyqtgraph.'''
    RGB = np.zeros((5,3))
    # first give white
    RGB[0,0] = 255;
    RGB[0,1] = 255;
    RGB[0,2] = 255;
    # then black
    # already zero
    # then red 
    #RGB[:255,0] = np.arange(255);
    RGB[2,0] = 0;
    RGB[2,0] = 255;
    # next keep red at 255 increase green
    #RGB[255:255*2,0] = 255
    #RGB[255:255*2,1] = np.arange(255)
    RGB[3,0] = 255
    RGB[3,1] = 255
    # last row
    RGB[4,0] = 255
    RGB[4,1] = 255
    RGB[4,2] = 255
    #RGB[255*2:255*3,0] = 255
    #RGB[255*2:255*3,1] = 255
    #RGB[255*2:255*3,2] = np.arange(255)

    RGB= RGB/255. #make float 0 to 1

    #pos = np.linspace(0, 1, 255*3)
    pos = np.linspace(0, 1, 4)
    cmap = pg.ColorMap(pos,RGB)

    return cmap

def makeALBULALUT():
    return makeALBULACmap().getLookupTable(alpha=False)


