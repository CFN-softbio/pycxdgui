# Simple GUI for xray image viewing/analysis

See buildenv.sh for a sample script on how to setup. Requires anaconda.


Running:
    python startGUI.py

Dependencies:

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
