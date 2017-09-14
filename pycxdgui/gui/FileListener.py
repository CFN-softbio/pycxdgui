from PyQt5.QtCore import QThread, pyqtSignal

from time import sleep
import os

import numpy as np


class FileListener(QThread):
    ''' This is meant to listen for files.
        Emits a "newerfile" signal if finds file.
        To change the directory or extension, just modify
        the self.ddir and self.extension members
            (is this thread safe? I am uncertain but works for now a better
            solution is use a signal and slot)

        Note : The new file is only returned when we are certain no one is
        touching it anymore (i.e. the file transfer is still not in progress).
        To check for this, we just check for the file size twice at some
        interval specified by wait_mtime.

        This function should listen on some specified directory. It will display the newest file.
            Special cases:
                - if the newest file is deleted, then it will find last newest
                - if directory is deleted, it won't emit any signals anymore
    '''
    signal_newerfile = pyqtSignal(str)

    def __init__(self, parent=None, wait_time=1):
        super(QThread, self).__init__(parent)
        # default just wait 1 second
        self.wait_time = wait_time
        # time to wait between checking if a file is modified
        self.wait_mtime = wait_time

    def listen_for_files(self, ddir, extension):
        self.ddir = ddir
        self.extension = extension
        self.curfilename = None
        self.cur_mod_time = None
        self.start()

    def run(self):
        while(True):
            sleep(self.wait_time)
            result = self.check_newest_file()
            #print("checking file state: {}".format(result))
            #print("current file : {}".format(self.curfilename))
            if result is not None:
                # send signal that file changed
                print("Found new file: {}".format(result))
                # TODO : fix file listener (signal passing args problem)
                print("ignoring for now")
                self.signal_newerfile.emit(result)

    def check_newest_file(self):
        ''' Check for latest image in directory
            This will check the timestamp of last "new" file.
            If time stamp is updated, returns the new file.
        '''
        extension = self.extension
        # get data directory from params
        ddir = self.ddir
        # get the current filename
        filename = self.curfilename
        cur_mod_time = self.cur_mod_time
        # now find newest file
        newest, new_mod_time = self.find_newest(ddir, extension)
        # if filename is new and a file was found
        #if newest != filename and newest is not None:
        if new_mod_time is not None \
           and (cur_mod_time is None or (new_mod_time > cur_mod_time)):
            # then update
            self.curfilename = newest
            self.cur_mod_time = new_mod_time
            return newest
        else:
            return None
        # else do nothing (so if no file found, ignore)

    def find_newest(self, ddir, ext):
        ''' find newest file in directory ddir with extension ext
        '''
        newest = None
        mod_time = None
        try:
            filelist = os.listdir(ddir)
            filelist = list(filter(lambda x: not os.path.isdir(x) and x.endswith(ext), filelist))

            if len(filelist) > 0:
                newest = max(filelist, key=lambda x: os.stat(ddir + "/" + x).st_mtime)
                if newest is not None:
                    newest = ddir + "/" + newest
                # now check file is not touched after a certain interval
                mtime1 = os.path.getmtime(newest)
                sleep(self.wait_mtime) #seconds
                mtime2 = os.path.getmtime(newest)
                dmtime = mtime2-mtime1
                # if being modified, ignore this newest
                if np.abs(dmtime) > 1e-6:
                    newest = None
                mod_time = mtime2
        except FileNotFoundError:
            newest = None

        return newest, mod_time
