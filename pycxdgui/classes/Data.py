import h5py
from PIL import Image
import numpy as np

def mask_open_hdf5(filename):
    f = h5py.File(filename, "r")
    mask = np.copy(f['mask'])
    f.close()
    return mask

def mask_save_hdf5(filename, mask):
    f = h5py.File(filename, "w")
    f['mask'] = mask
    f.close()

def mask_open_tiff(filename):
    mask = np.copy(Image.open(filename))
    return mask

def mask_save_tiff(filename, mask):
    mask = Image.fromarray(mask.astype(np.uint8))
    mask.save(filename)


class Data2D:
    ''' initialize a 2D contiguous data set. Either specify with shape
        (initializes to zeros) for give a mask.

        Assumes a 2D array format

        ftype : file type

        Note: if a mask is supplied, it makes a copy (to not overwrite given mask)
    '''
    # all possible extensions
    extensions = {
            'hd5' : {   'name' : 'hdf5',
                        'fmtstring' : '*mask*.hd5',
                        'reader' : mask_open_hdf5,
                        'writer' : mask_save_hdf5,
                        },
            'tif' : {   'name' : 'tiff',
                        'fmtstring' : '*mask*.tif',
                        'reader' : mask_open_tiff,
                        'writer' : mask_save_tiff,
                        },
    }

    def __init__(self,shape=None, mask=None, ftype="hd5"):
        if shape is None and mask is None:
            raise ValueError("Error, either shape or mask must be specified")
        # now we know at least one is set, default to using mask over shape
        if shape is not None:
            self.mask = np.zeros(shape, dtype=int)
        if mask is not None:
            self.mask = np.copy(mask, dtype=int)

        self.ftype = ftype

    def __call__(self):
        return self.mask

    def fill(self, val):
        self.mask[:,:] = val

    def zero(self):
        self.fill(0)

    def one(self):
        self.fill(1)

    def exclude(self, pixels):
        self.mask[pixels] = 0

    def include(self, pixels):
        self.mask[pixels] = 1

    def load(self, maskfilename, ftype=None):
        ''' loads from maskfilename.
            Tries to guess the ftype is not specified.
        '''
        if ftype is None:
            if 'tif' in maskfilename:
                ftype = 'tif'
            elif 'hd5' in maskfilename or 'h5' in maskfilename:
                ftype = 'hd5'
            else:
                # default to tif
                ftype = 'tif'
        self.ftype = ftype
        reader = self.extensions[self.ftype]['reader']
        self.mask = reader(maskfilename)

    def save(self, maskfilename):
        writer = self.extensions[self.ftype]['writer']
        writer(maskfilename, self.mask)
