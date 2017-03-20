import numpy as np

def findLowHigh(img, maxcts=None):
    ''' Find the reasonable low and high values of an image
            based on its histogram.
            Ignore the zeros
    '''
    if maxcts is None:
        maxcts = 65536
    w = np.where((~np.isnan(img.ravel()))*(~np.isinf(img.ravel())))
    hh,bb = np.histogram(img.ravel()[w], bins=maxcts, range=(1,maxcts))
    hhs = np.cumsum(hh)
    hhs = hhs/np.sum(hh)
    wlow = np.where(hhs > .15)[0] #5%
    whigh = np.where(hhs < .85)[0] #95%
    if len(wlow):
        low = wlow[0]
    else:
        low = 0
    if len(whigh):
        high = whigh[-1]
    else:
        high = maxcts
    if high <= low:
        high = low + 1
    # debugging
    #print("low: {}, high : {}".format(low, high))
    return low, high
