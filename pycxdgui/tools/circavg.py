''' Note: circavg has been replaced by a new version. It's less portable
    but more flexible.
    If you need to copy and paste quick circavg code, use the older code which
        has only a numpy dependency.
    I noticed the new circavg appears 2-3 times slower. I'll keep it though,
        since we don't really perform tons of circular averages (other than
        SAXS, but that's usually a one time thing that requires no
        thought/interaction)
'''
import numpy as np
from .coordinates import mkpolar
from .partitions import linplist, partition1D, partition2D
from .partitions import partition_avg, partition_sum
from .interpolations import fillin1d

def circavg(img,x0=None,y0=None,mask=None,SIMG=None,noqs=None,xrdata=None):
    ''' Compute a circular average of the data. 
        x0 : x center
        y0 : y center
        mask : the mask
        If SIMG is not null, put the data into this image.
        noqs : the number of qs to partition into. Default is 
            the number of pixels approximately.
    '''
    if xrdata is not None:
        x0, y0 = xrdata.x0, xrdata.y0
        if hasattr(xrdata,"mask_name"):
            mask = openmask(SDIR + "/" + xrdata.mask_name)
    dimy, dimx = img.shape
    if(mask is None):
        pxlst = np.arange(dimx*dimy)
    else:
        pxlst = np.where(mask.ravel() == 1)[0]

    if(x0 is None):
        x0 = dimx/2
    if(y0 is None):
        y0 = dimy/2
    
    QS,PHIS = mkpolar(img,x0=x0,y0=y0)
    qs,phis = QS.ravel()[pxlst],PHIS.ravel()[pxlst]

    data = img.ravel()[pxlst]

    
    if(noqs is None):
        noqs = (np.max(qs)-np.min(qs)).astype(int)
    qlist_m1 = linplist(np.min(qs),np.max(qs),noqs)
    #philist_m1 = linplist(np.min(phis),np.max(phis),1)

    qid,pselect = partition1D(qlist_m1,qs)

    sqy = partition_avg(data[pselect],qid)
    #sqx = partition_avg(qs[pselect],qid)
    sqx = (qlist_m1[0::2] + qlist_m1[1::2])/2.
    if(len(sqy) < len(sqx)):
        sqy = np.concatenate((sqy,np.zeros(len(sqx)-len(sqy))))

    if(SIMG is not None):
        SIMGtmp =  0*SIMG
        SIMGtmp = np.interp(QS,sqx,sqy)
        np.copyto(SIMG,SIMGtmp)
    return sqx, sqy

def circsum(img,x0=None,y0=None,mask=None,SIMG=None,noqs=None):
    ''' Compute a circular sum of the data. 
        x0 : x center
        y0 : y center
        mask : the mask
        If SIMG is not null, put the data into this image.
        noqs : the number of qs to partition into. Default is 
            the number of pixels approximately.
    '''
    dimy, dimx = img.shape
    if(mask is None):
        pxlst = np.arange(dimx*dimy)
    else:
        pxlst = np.where(mask.ravel() == 1)[0]

    if(x0 is None):
        x0 = dimx/2
    if(y0 is None):
        y0 = dimy/2
    
    QS,PHIS = mkpolar(img,x0=x0,y0=y0)
    qs,phis = QS.ravel()[pxlst],PHIS.ravel()[pxlst]

    data = img.ravel()[pxlst]

    
    if(noqs is None):
        noqs = (np.max(qs)-np.min(qs)).astype(int)
    qlist_m1 = linplist(np.min(qs),np.max(qs),noqs)
    #philist_m1 = linplist(np.min(phis),np.max(phis),1)

    qid,pselect = partition1D(qlist_m1,qs)

    sqy = partition_sum(data[pselect],qid)
    #sqx = partition_avg(qs[pselect],qid)
    sqx = (qlist_m1[0::2] + qlist_m1[1::2])/2.

    if(SIMG is not None):
        SIMGtmp =  0*SIMG
        SIMGtmp = np.interp(QS,sqx,sqy)
        np.copyto(SIMG,SIMGtmp)
    return sqx, sqy


def circavg2(img1,img2,x0=None,y0=None,mask=None,SIMG=None):
    ''' Compute a circular average of the second moment of the data. 
        If SIMG is not null, put the data into this image.
    '''
    if(mask is None):
        mask = np.ones(img1.shape)
    dimy,dimx = img1.shape
    if(x0 is None):
        x0 = dimx/2
    if(y0 is None):
        y0 = dimy/2
    imgprod = img1.ravel()*img2.ravel()

    pixellist = np.where(mask.ravel()==1)

    x = np.arange(dimx) - x0
    y = np.arange(dimy) - y0
    X,Y = np.meshgrid(x,y)
    R = np.sqrt(X**2 + Y**2).ravel()
    Rd = (R+.5).astype(int).ravel()
    
    noperR = np.bincount(Rd.ravel()[pixellist]).astype(float)
    w = np.where(noperR != 0)

    Rvals = np.bincount(Rd.ravel()[pixellist],weights=R.ravel()[pixellist])[w]/noperR[w]
    Ivals = np.bincount(Rd.ravel()[pixellist],weights=imgprod[pixellist])[w]/noperR[w]
    if(SIMG is not None):
        np.copyto(SIMG.ravel(),np.interp(R,Rvals,Ivals))
    return Rvals, Ivals

#------ the old circavg if new circavg is slow, try this one it has been robustly tested -------
def circavg_old(img,x0=None,y0=None,mask=None,SIMG=None):
    ''' Compute a circular average of the data. 
        If SIMG is not null, put the data into this image.
    '''
    if(mask is None):
        mask = np.ones(img.shape)
    dimy,dimx = img.shape
    if(x0 is None):
        x0 = dimx/2
    if(y0 is None):
        y0 = dimy/2
    img = img.ravel()

    pixellist = np.where(mask.ravel()==1)

    x = np.arange(dimx) - x0
    y = np.arange(dimy) - y0
    X,Y = np.meshgrid(x,y)
    R = np.sqrt(X**2 + Y**2).ravel()
    Rd = (R+.5).astype(int).ravel()
    
    noperR = np.bincount(Rd.ravel()[pixellist]).astype(float)
    w = np.where(noperR != 0)

    Rvals = np.bincount(Rd.ravel()[pixellist],weights=R.ravel()[pixellist])[w]/noperR[w]
    Ivals = np.bincount(Rd.ravel()[pixellist],weights=img.ravel()[pixellist])[w]/noperR[w]
    if(SIMG is not None):
        np.copyto(SIMG.ravel(),np.interp(R,Rvals,Ivals))
    return Rvals, Ivals
