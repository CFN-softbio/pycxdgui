import numpy as np


class NPYReader:
    description = "NPY File"
    def __init__(self, filename):
        self.image = np.load(filename)


