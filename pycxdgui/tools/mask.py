import h5py
import numpy as np
#from tools.coordinates import mkpolar
from skbeam.core.utils import radial_grid, angle_grid

def findnexthigh(img,mask,n=1):
    ''' Find the next high point in image. Use this to successively mask out 
        pixels from a mask.
        n : find the n successives points
    '''
    imgtmp = img*mask
    imgwidth = imgtmp.shape[1]
    pxlsx = list()
    pxlsy = list()
    for i in range(n):
        pos = np.argmax(imgtmp)
        posx, posy = pos%imgwidth, pos//imgwidth
        print("next highest pixel : ({},{}), value: {} cts".format(posx, posy, img[posy, posx]))
        imgtmp[pos//imgwidth, pos%imgwidth] *= 0
        pxlsx.append(posx)
        pxlsy.append(posy)
    return np.array(pxlsy),np.array(pxlsx)

def openmask(maskfilename):
    ''' Open a mask with file name. Just a quick shortcut to using hdf5 etc'''
    if maskfilename is not None and len(maskfilename) > 0:
        try:
            f = h5py.File(maskfilename,"r")
            mask = np.copy(f['mask'])
            f.close()
        except IOError or ValueError:
            mask = None
    else:
        mask = None
    return mask

def savemask(maskfilename,mask):
    ''' Save a mask.'''
    f = h5py.File(maskfilename, "w")
    f['mask'] = mask
    f.close()

def addmaskwedge(mask, r0, qlim=None, philim=None):
    ''' add a wedge to mask centered at x0, y0
        note : need to fix. right now phi (hopefully)
            goes from -pi to pi but it could change
            if anything changes in mkpolar
    '''
    Q, PHI = mkpolar(mask,x0=r0[0], y0=r0[1])
    Q = radial_grid((r0[0], r0[1]), mask.shape)
    PHI = angle_grid((r0[0], r0[1]), mask.shape)
    PHI = PHI%(2*np.pi)
    philim = np.array(philim)%(2*np.pi)

    expression = np.ones_like(Q,dtype=bool)

    if qlim is not None:
        expression *= (Q >= qlim[0])*(Q < qlim[1])
    if philim is not None:
        if (philim[1] >= philim[0]):
            expression *= (PHI >= philim[0])*(PHI < philim[1])
        else:
            # special case where philim straddles the wrap around boundary
            expression *= (PHI >= philim[0])|(PHI < philim[1])

    w = np.where(expression)

    mask[w] *= 0 

def makenfoldmask(mask,n,ddphi=None,r0=None):
    ''' make a mask with n fold symmetry. 
        ddphi - mask wedge thickness (in angle)
    '''
    if r0 is None:
        r0 = mask.shape[1]/2, mask.shape[0]/2
    masksym = n
    dphi = 2*np.pi/masksym
    if ddphi is None:
        ddphi = .1
    nphi = masksym

    phi0 = 0

    for i in range(nphi):
        addmaskwedge(mask,r0,qlim=None, philim = [phi0 + dphi*i - ddphi/2., phi0 + dphi*i + ddphi/2.])

    return mask
