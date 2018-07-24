# Simple GUI for xray image viewing/analysis

See buildenv.sh for a sample script on how to setup. Requires anaconda.


## Running

Running is as simple as running this within python:
```py3
    from pycxdgui.startGUI import run
    run(configfile=config_filename)
```
where `config_filename` is the location to your configuration file (see below
for more details).


## Dependencies

- python3.5 PyQt4 numpy h5py pip
- pims pillow tqdm scipy scikit-image pandas cython
- pyqtgraph. This has been copied from the
  [pyqgraph](https://github.com/pyqtgraph/pyqtgraph) repository from commit
  4752b777921f54b3a23dfc2952697ddf11922112.
  To retrieve this code, run the following:

  ```bash
  git clone https://github.com/pyqtgraph/pyqtgraph
  cd pyqtgraph
  git checkout 4752b777921f54b3a23dfc2952697ddf11922112
  ```

## Configuration

A configuration file may be used to specify initial configuration setings of
the GUI. A sample configuration file is as follows (see comments for the
descriptions):

```yaml
# yaml file for configuration
# this is conventional yaml format:
# http://www.yaml.org/start.html
# comment out rows you don't want
setup:
    # default extension to look for
    extension: ".imm"
    # the parent directory
    DDIR: "." 
    # the storage directory
    SDIR: "."
    # beam center
    xcen: 1313.39
    ycen: 1267.81
    # wait time for file listening
    wait_time: .1
    # the initial mask name
    mask_name: "/path/to/mask.hd5"
    # the values that define masked and unmasked region >=
    mask_threshold: 1
    # sample detector distance
    rdet: 3.800 # in m
    energy: 7.44137  # in keV
    wavelength: 1.6660910558136472  # in angstroms
    dpix: 20 # pixel size in um
    # first filename to load
    filename: "/path/to/filename.tiff"
    transformation: [ [1, 0],
                      [0, 1]
                    ]
# parameters for circavg computations
circavg:
    noqs: 800 #partitioning for circular average/qphi maps
# parameters for qphi avg computations
qphiavg:
    noqs: 800 #partitioning for circular average/qphi maps
    nophis: 360 #partitioning for qphimaps in phi
deltaphicorr:
    noqs: 800 #partitioning for circular average/qphi maps
    nophis: 360 #partitioning for qphimaps in phi
```

Credits:
Some of the icons were provided for free online by various authors. 
See icons/source-list.txt for the original source link for each icon

* Load Image Icon: (icons/load_image_icon.jpg) made by [Ico Moon](https://www.flaticon.com/authors/icomoon) from www.flaticon.com
* Load Mask Icon (icons/load_mask_icon.jpg) made by [Egor Rumyantsev](https://www.flaticon.com/authors/egor-rumyantsev) from www.flaticon.com
* Data tableIcon (icons/datatable_icon.png) made by [Freepik](https://www.flaticon.com/authors/freepik) from www.flaticon.com
* Circular Average Icon (icons/circavg_icon.png) made by [Freepik](https://www.flaticon.com/authors/freepik) from www.flaticon.com
* Q-Phi map Icon (icons/sqphi_icon.png) made by [Silviu Runceanu](https://www.flaticon.com/authors/silviu-runceanu) from www.flaticon.com
* File listenenr Icon (icons/listen_icon.png) made by [Freepik](https://www.flaticon.com/authors/freepik) from www.flaticon.com
* Lock Aspect Ratio Icon (icons/lock_aspect_icon.png) made by [Picol](https://www.flaticon.com/authors/picol) from www.flaticon.com
* Exit Icon (icons/exit_icon.png) made by [Freepik](https://www.flaticon.com/authors/freepik) from www.flaticon.com
