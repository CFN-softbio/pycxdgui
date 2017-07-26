import h5py
from PIL import Image
#from tifffile import TiffFile
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
    mask = np.array(Image.open(filename))
    #mask = np.copy(TiffFile(filename).asarray())
    return mask

def mask_save_tiff(filename, mask):
    mask = Image.fromarray(mask.astype(np.uint8))
    mask.save(filename, format="tiff")


class Mask:
    ''' initialize a mask. Either specify with shape (initializes to zeros)
        for give a mask.

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
            self.mask = np.copy(mask).astype(int)

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
        if pixels is not None:
            self.mask[pixels] = 0

    def include(self, pixels):
        if pixels is not None:
            self.mask[pixels] = 1

    def get_ftype_from_filename(self, filename):
        if 'tif' in filename.lower():
            ftype = 'tif'
        elif 'hd5' in filename or 'h5' in filename:
            ftype = 'hd5'
        else:
            # default to current
            ftype = self.ftype

        return ftype

    def load(self, maskfilename, ftype=None):
        ''' loads from maskfilename.
            Tries to guess the ftype is not specified.
        '''
        if ftype is None:
            ftype = self.get_ftype_from_filename(maskfilename)
        self.ftype = ftype
        reader = self.extensions[self.ftype]['reader']
        self.mask = reader(maskfilename)

    def post_process(self, mask, ftype):
        ''' post process image for outputting to a specific file
            type.
            For example, for TIFF, 1 should be 255 so it is seen easier.
        '''
        if ftype is 'tif':
            mask[mask > .5] = 255
        return mask

    def save(self, maskfilename, ftype=None):
        print(maskfilename)
        if ftype is None:
            ftype = self.get_ftype_from_filename(maskfilename)
        self.ftype = ftype
        writer = self.extensions[self.ftype]['writer']
        mask_out = self.post_process(self.mask, ftype)
        writer(maskfilename, mask_out)
