from .tiff import TIFFReader
from .eiger import EigerImages
from .npy import NPYReader
from .hdf5 import HDF5Reader
from collections import OrderedDict


# insertion order matters
extension_dict = OrderedDict()
extension_dict[".tiff"] =  TIFFReader
extension_dict[".TIFF"] =  TIFFReader
extension_dict[".tif"] =  TIFFReader
extension_dict[".TIF"] =  TIFFReader
extension_dict["_master.h5"] = EigerImages
extension_dict[".npy"] = NPYReader
extension_dict[".h5"] = HDF5Reader
extension_dict[".hdf5"] = HDF5Reader
