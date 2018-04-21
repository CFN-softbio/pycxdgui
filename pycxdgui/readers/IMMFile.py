import numpy as np

class IMMFile:
    def __init__(self, filename, ndarks=10,PF=True):
        ''' Initialize IMM file.

            ndark : number of darks in file (negative number searches from beg)

            IMM file is a sequence of 1024 headers followed by an image.
                The size of the image is determined by the header.

            Note: Mark's code looked for a "magic" number.
                I could not find any number that looked like either
                one: [25992,39019912]
                So I just use little endian format.

                - I have not checked the member variables, they could be wrong

            numpy's memmap allows one to point to file
                without reading it in

        '''
        self.ndarks = ndarks
        self.PF = PF

        rawdata64 = np.memmap(filename,dtype='<i8')
        rawdata32 = np.memmap(filename,dtype='<i4')
        rawdata16 = np.memmap(filename,dtype='<i2')
        rawdata8 = np.memmap(filename,dtype='<i1')
        self.rawdata8 = rawdata8
        self.rawdata16 = rawdata16
        self.rawdata32 = rawdata32
        self.rawdata64 = rawdata64

        self.headersize = 1024

        self.rows,self.cols = None, None
        self.readheader(0)

        self.noframes = rawdata32[56//4]
        self.noframes = int((rawdata16.shape[0])/(self.rows*self.cols+self.headersize//2))
        self.noimgs = self.noframes-self.ndarks

        # set up positions
        self.ndarks = ndarks
        self.firstdark = 0
        self.first = 0
        if ndarks > 0:
            self.first += ndarks
        self.last = ndarks + self.noimgs
        if ndarks < 0:
            self.last -= ndarks
            self.firstdark += self.last


        self.darkimg = None

    def readheader(self, nh):
        ''' read the nth header. '''
        if self.rows is None:
            h0 = 0
            self.rows = self.rawdata32[(h0 + 108)//4] # at byte position 108
            self.cols = self.rawdata32[(h0 + 112)//4]
        h0 = nh*(self.headersize + self.rows*self.cols)
        self.mode = self.rawdata32[h0//4]
        self.compression = self.rawdata32[(h0+4)//4]
        self.date = self.rawdata8[h0+8:h0+8+32].tostring()
        self.prefix = self.rawdata8[h0+40:h0+40+16].tostring()
        self.suffix = self.rawdata8[h0+60:h0+60+16].tostring()
        self.monitor = self.rawdata32[(h0 + 76)//4]
        self.shutter = self.rawdata32[(h0 + 80)//4]
        self.row_beg = self.rawdata32[(h0 + 84)//4]
        self.row_end = self.rawdata32[(h0 + 88)//4]
        self.col_beg = self.rawdata32[(h0 + 92)//4]
        self.col_end = self.rawdata32[(h0 + 96)//4]
        self.row_bin = self.rawdata32[(h0 + 100)//4]
        self.col_bin = self.rawdata32[(h0 + 104)//4]
        self.rows = self.rawdata32[(h0 + 108)//4] # at byte position 108
        self.cols = self.rawdata32[(h0 + 112)//4]
        self.kinetics = self.rawdata32[(h0 + 120)//4]
        self.kinwinsize = self.rawdata32[(h0 + 124)//4]
        self.elapsed = self.rawdata64[(h0 + 128)//8].astype(float)
        self.preset = self.rawdata64[(h0 + 136)//8].astype(float)


    def __getitem__(self, n):
        hdr0 = self.headersize//2*(n+1) + n*self.rows*self.cols
        return np.copy(self.rawdata16[hdr0:hdr0+self.rows*self.cols]).reshape(
                    (self.rows,self.cols))

    def __len__(self):
        return self.last-self.first

    def dkanal(self):
        ''' dark analysis.'''
        self.darkimg = np.zeros((self.rows, self.cols),dtype=float)
        for i in range(self.firstdark, self.firstdark+self.ndarks):
            if self.PF:
                print("Dark Analysis: {} of {}".format(i-self.firstdark+1,self.ndarks))
            self.darkimg += self[i]
        self.darkimg /= self.ndarks

    def intanal(self):
        ''' int analysis.'''
        if self.darkimg is None:
            print("Sorry can't run intensity analysis. Please run dark analysis first")
            return -1
        self.avgimg = np.zeros((self.rows, self.cols),dtype=float)
        for i in range(self.first, self.last):
            if self.PF:
                print("Intensity Analysis: {} of {}".format(i-self.first+1,self.noimgs))
            self.avgimg += self[i]-self.darkimg
        self.avgimg /= self.noimgs
