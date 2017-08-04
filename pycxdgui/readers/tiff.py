# this is the TIFF image reader
# readers are implied to be image readers. T
from .readers_base import ReaderBase
from PIL import Image


class TIFFReader(ReaderBase):
    description = "TIFF File"
    def __init__(self, fpath):
        # can raise FileNotFound
        im = Image.open(fpath)
        self.image = np.array(img).astype(float)
        # if it's 3D, collapse to 1D
        # it's assumed 3rd axis is the channel (color, alpha)
        if self.image.ndims == 3:
            self.image = np.average(self.image, axis=2)

    def to_array(self):
        return self.image
