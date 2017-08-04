from pycxdgui.readers.readers_base import ReaderRegistry
from collections import OrderedDict
from pycxdgui.readers import main_reader
from pycxdgui.readers.tiff import TIFFReader
from pycxdgui.readers.eiger import EigerImages
from pycxdgui.readers.npy import NPYReader
from pycxdgui.readers.hdf5 import HDF5Reader

def test_readermultiplexer():

    reg = OrderedDict({'foo' : 'some handler',
                       'bar' : 'some other handler',
                       'oo' : 'should not get here'})

    reader = ReaderRegistry(extension_dict=reg)
    reader.register('abc', 'another reader')
    assert reader.extension_dict['foo'] == 'some handler'
    assert reader.extension_dict['abc'] == 'another reader'

def test_main_reader():
    ''' test the main reader'''
    reader = main_reader._get_reader_from_filename("foo.tiff")
    assert reader is TIFFReader
    reader = main_reader._get_reader_from_filename("foo.master.h5")
    assert reader is EigerImages
    reader = main_reader._get_reader_from_filename("foo.npy")
    assert reader is NPYReader
    reader = main_reader._get_reader_from_filename("foo.h5")
    assert reader is HDF5Reader
