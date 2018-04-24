#!/usr/bin/env python
import numpy as np

#from distutils.core import setup
import setuptools

setuptools.setup(name='pycxdgui',
    version='1.0',
    author='Julien Lhermitte',
    description="Coherent Xray Diffraction GUI",
    include_dirs=[np.get_include()],
    packages=setuptools.find_packages(),
    author_email='lhermitte@bnl.gov',
    install_requires=['tifffile', 'pims', 'scikit-beam', 'numpy'],  # essential deps only
#    install_requires=['six', 'numpy'],  # essential deps only
    keywords='Image Processing Analysis',
    license='BSD',
     )
