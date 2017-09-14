# directory to save the gui to
SOFTWAREDIR=~/software

conda create -n pycxdgui python=3.5 qt=5 pyqtgraph numpy h5py pip
source activate pycxdgui
pip install pims pillow tqdm scipy scikit-image pandas cython

# install scikit-beam to some directory
# make some directory for software
mkdir -p $SOFTWAREDIR

pushd $SOFTWAREDIR
git clone https://www.github.com/scikit-beam/scikit-beam.git
cd scikit-beam
python setup.py develop
cd ..
# now install pycxdGUI
git clone https://www.github.com/CFN-softbio/pycxdgui.git
cd pycxdgui
python setup.py develop

popd
