# qphi average routines
import numpy as np
from tools.coordinates import mkpolar
from tools.partitions import linplist, partition1D, partition2D
from tools.partitions import partition_avg, partition_sum
from tools.interpolations import fillin1d

def qphiavg_getlists(img,x0=None,y0=None,mask=None,SIMG=None,qlist=None,philist=None,noqs=None,nophis=None,interp=None):
    ''' get lists from data.'''
    dimy, dimx = img.shape
    if(mask is None):
        pxlst = np.arange(dimx*dimy)
    else:
        pxlst = np.where(mask.ravel() == 1)[0]

    if(x0 is None):
        x0 = dimx/2
    if(y0 is None):
        y0 = dimy/2

    if(qlist is None):
        if(noqs is None):
            noqs = 800
        qlist_m1 = None
    elif(len(qlist) == 1):
        noqs = qlist
        qlist_m1 = None
    else:
        qlist_m1 = np.array(qlist)
        noqs = qlist_m1.shape[0]//2

    if(philist is None):
        if(nophis is None):
            nophis = 360 
        philist_m1 = None
    elif(len(philist) == 1):
        noqs = philist
        philist_m1 = None
    else:
        philist_m1 = np.array(philist)
        nophis = philist_m1.shape[0]//2
    
    QS,PHIS = mkpolar(img,x0=x0,y0=y0)
    qs,phis = QS.ravel()[pxlst],PHIS.ravel()[pxlst]

    data = img.ravel()[pxlst]
    
    if(noqs is None):
        noqs = (np.max(qs)-np.min(qs))
    if(nophis is None):
        nophis = 12

    nophis = int(nophis);noqs = int(noqs);
    if(qlist_m1 is None):
        qlist_m1 = linplist(np.min(qs),np.max(qs),noqs)
    if(philist_m1 is None):
        philist_m1 = linplist(np.min(phis),np.max(phis),nophis)
    return qlist_m1, philist_m1

def qphiavg_getvals(img,x0=None,y0=None,mask=None,SIMG=None,qlist=None,philist=None,noqs=None,nophis=None,interp=None):
    ''' get lists from data.'''
    dimy, dimx = img.shape
    if(mask is None):
        pxlst = np.arange(dimx*dimy)
    else:
        pxlst = np.where(mask.ravel() == 1)[0]

    if(x0 is None):
        x0 = dimx/2
    if(y0 is None):
        y0 = dimy/2
    if(qlist is None):
        if(noqs is None):
            noqs = 800
        qlist_m1 = None
    elif(len(qlist) == 1):
        noqs = qlist
        qlist_m1 = None
    else:
        qlist_m1 = np.array(qlist)
        noqs = qlist_m1.shape[0]//2

    if(philist is None):
        if(nophis is None):
            nophis = 360 
        philist_m1 = None
    elif(len(philist) == 1):
        noqs = philist
        philist_m1 = None
    else:
        philist_m1 = np.array(philist)
        nophis = philist_m1.shape[0]//2
    
    QS,PHIS = mkpolar(img,x0=x0,y0=y0)
    qs,phis = QS.ravel()[pxlst],PHIS.ravel()[pxlst]

    data = img.ravel()[pxlst]
    
    if(noqs is None):
        noqs = (np.max(qs)-np.min(qs))
    if(nophis is None):
        nophis = 12

    nophis = int(nophis);noqs = int(noqs);
    if(qlist_m1 is None):
        qlist_m1 = linplist(np.min(qs),np.max(qs),noqs)
    if(philist_m1 is None):
        philist_m1 = linplist(np.min(phis),np.max(phis),nophis)
    qvals = (qlist_m1[::2] + qlist_m1[1::2])*.5
    phivals = (philist_m1[::2] + philist_m1[1::2])*.5
    return qvals, phivals

def qphiavg(img,x0=None,y0=None,mask=None,SIMG=None,qlist=None,philist=None,noqs=None,nophis=None,interp=None):
    ''' Compute a qphi average of the data.
        x0 : x center
        y0 : y center
        mask : the mask
        If SIMG is not null, put the data into this image.
        noqs : the number of qs to partition into. Default is
            the number of pixels approximately.
        interp : interpolation methods:
            None (default) : no interpolation
            1 : interpolate in phi only (loop over q's)
            ... so far no other methods
    '''
    if mask is not None:
        if img.shape != mask.shape:
            errormsg = "Error, image and mask not same shape\n"
            errormsg += "Image shape : {}".format(img.shape)
            errormsg += "Mask shape : {}".format(mask.shape)
            raise ValueError(errormsg)
    dimy, dimx = img.shape
    if(mask is None):
        pxlst = np.arange(dimx*dimy)
    else:
        pxlst = np.where(mask.ravel() == 1)[0]

    if(x0 is None):
        x0 = dimx/2
    if(y0 is None):
        y0 = dimy/2

    if(qlist is None):
        if(noqs is None):
            noqs = 800
        qlist_m1 = None
    elif(len(qlist) == 1):
        noqs = qlist
        qlist_m1 = None
    else:
        qlist_m1 = np.array(qlist)
        noqs = qlist_m1.shape[0]//2

    if(philist is None):
        if(nophis is None):
            nophis = 360 
        philist_m1 = None
    elif(len(philist) == 1):
        noqs = philist
        philist_m1 = None
    else:
        philist_m1 = np.array(philist)
        nophis = philist_m1.shape[0]//2

    QS,PHIS = mkpolar(img,x0=x0,y0=y0)
    qs,phis = QS.ravel()[pxlst], PHIS.ravel()[pxlst]

    data = img.ravel()[pxlst]
    if(noqs is None):
        noqs = (np.max(qs)-np.min(qs))
    if(nophis is None):
        nophis = 12

    nophis = int(nophis);noqs = int(noqs);
    if(qlist_m1 is None):
        qlist_m1 = linplist(np.min(qs),np.max(qs),noqs)
    if(philist_m1 is None):
        philist_m1 = linplist(np.min(phis),np.max(phis),nophis)

    qid,pid,pselect = partition2D(qlist_m1,philist_m1,qs,phis)
    bid = (qid*nophis + pid).astype(int)

    sqphi = partition_avg(data[pselect],bid)

    sqphitot = np.zeros((noqs,nophis))
    maxid = np.max(bid)
    sqphitot.ravel()[:maxid+1] = sqphi

    phis_m1 = (philist_m1[0::2] + philist_m1[1::2])/2.
    qs_m1 = (qlist_m1[0::2] + qlist_m1[1::2])/2.

    if(interp is not None):
        sqmask = np.zeros((noqs,nophis))
        sqmask.ravel()[:maxid + 1] = partition_avg(pselect*0+1,bid)
        if((interp == 1) or (interp == 2)):
            for i in range(sqphitot.shape[0]):
                # won't work if you have numpy version < 1.10
                sqphitot[i] = fillin1d(phis_m1, sqphitot[i], sqmask[i],period=2*np.pi)
                # best suited for phi range -pi/2 to pi/2, might need to change
                # for diff versions
        if interp == 2:
            # second interp method also uses point symmetry
            # reverse philist, and rebin
            # assume the point of reflection is about zero, if not, maybe this
            # commented code could be tried (not tested)
            # first, find the point of reflection (is it about zero, -pi or pi?)
            #avgphi = np.average(phis_m1)
            #if(avgphi > -np.pi/2. and avgphi < np.pi/2.):
                #const = 0. #do nothing
            #elif(avgphi > np.pi/2. and avgphi < 3*np.pi/2.):
                #const = np.pi
            #elif(avgphi > -3*np.pi/2. and avgphi < -np.pi/2.):
                #const = np.pi
            const = 0.
            # now reflect philist
            philist_rev = const - philist_m1[::-1]
            qidr, pidr, pselectr = partition2D(qlist_m1, philist_rev, qs, phis)
            bidr = (qidr*nophis + pidr).astype(int)
            maxidr = np.max(bidr)
            sqphitotr = np.zeros((noqs,nophis))
            sqphitotr.ravel()[:maxidr+1] = partition_avg(data[pselectr],bidr)
            sqphitotr = sqphitotr[:,::-1]
            sqmaskr = np.zeros((noqs,nophis))
            sqmaskr.ravel()[:maxidr + 1] = partition_avg(pselectr*0 + 1,bidr)
            sqmaskr = sqmaskr[:,::-1]
            # now fill in values
            # just fill in the points, don't interp
            w = np.where((sqmask == 0)*(sqmaskr == 1))
            sqphitot[w] = sqphitotr[w]
    
    if(SIMG is not None):
        SIMG.ravel()[pxlst[pselect]] = sqphitot.ravel()[bid]

    return sqphitot

def qphisum(img,x0=None,y0=None,mask=None,SIMG=None,qlist=None,philist=None,noqs=None,nophis=None,interp=None):
    ''' Compute a qphi average of the data. 
        x0 : x center
        y0 : y center
        mask : the mask
        If SIMG is not null, put the data into this image.
        noqs : the number of qs to partition into. Default is 
            the number of pixels approximately.
        interp : interpolation methods:
            None (default) : no interpolation
            1 : interpolate in phi only (loop over q's)
            ... so far no other methods
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
    if(qlist is None):
        if(noqs is None):
            noqs = 800
        qlist_m1 = None
    elif(len(qlist) == 1):
        noqs = qlist
        qlist_m1 = None
    else:
        qlist_m1 = np.array(qlist)
        noqs = qlist_m1.shape[0]//2

    if(philist is None):
        if(nophis is None):
            nophis = 360 
        philist_m1 = None
    elif(len(philist) == 1):
        noqs = philist
        philist_m1 = None
    else:
        philist_m1 = np.array(philist)
        nophis = philist_m1.shape[0]//2
    
    QS,PHIS = mkpolar(img,x0=x0,y0=y0)
    qs,phis = QS.ravel()[pxlst],PHIS.ravel()[pxlst]

    data = img.ravel()[pxlst]
    
    if(noqs is None):
        noqs = (np.max(qs)-np.min(qs))
    if(nophis is None):
        nophis = 12

    nophis = int(nophis);noqs = int(noqs);
    if(qlist_m1 is None):
        qlist_m1 = linplist(np.min(qs),np.max(qs),noqs)
    if(philist_m1 is None):
        philist_m1 = linplist(np.min(phis),np.max(phis),nophis)

    qid,pid,pselect = partition2D(qlist_m1,philist_m1,qs,phis)
    bid = (qid*nophis + pid).astype(int)

    sqphi = partition_sum(data[pselect],bid)

    sqphitot = np.zeros((noqs,nophis))
    maxid = np.max(bid)
    sqphitot.ravel()[:maxid+1] = sqphi

    phis_m1 = (philist_m1[0::2] + philist_m1[1::2])/2.
    qs_m1 = (qlist_m1[0::2] + qlist_m1[1::2])/2.

    if(interp is not None):
        sqmask = np.zeros((noqs,nophis))
        sqmask.ravel()[:maxid + 1] = partition_avg(pselect*0+1,bid)
        if((interp == 1) or (interp == 2)):
            for i in range(sqphitot.shape[0]):
                # won't work if you have numpy version < 1.10
                sqphitot[i] = fillin1d(phis_m1, sqphitot[i], sqmask[i],period=2*np.pi)
                # best suited for phi range -pi/2 to pi/2, might need to change
                # for diff versions
        if interp == 2:
            # second interp method also uses point symmetry
            # reverse philist, and rebin
            # assume the point of reflection is about zero, if not, maybe this
            # commented code could be tried (not tested)
            # first, find the point of reflection (is it about zero, -pi or pi?)
            #avgphi = np.average(phis_m1)
            #if(avgphi > -np.pi/2. and avgphi < np.pi/2.):
                #const = 0. #do nothing
            #elif(avgphi > np.pi/2. and avgphi < 3*np.pi/2.):
                #const = np.pi
            #elif(avgphi > -3*np.pi/2. and avgphi < -np.pi/2.):
                #const = np.pi
            const = 0.
            # now reflect philist
            philist_rev = const - philist_m1[::-1]
            qidr, pidr, pselectr = partition2D(qlist_m1, philist_rev, qs, phis)
            bidr = (qidr*nophis + pidr).astype(int)
            maxidr = np.max(bidr)
            sqphitotr = np.zeros((noqs,nophis))
            sqphitotr.ravel()[:maxidr+1] = partition_avg(data[pselectr],bidr)
            sqphitotr = sqphitotr[:,::-1]
            sqmaskr = np.zeros((noqs,nophis))
            sqmaskr.ravel()[:maxidr + 1] = partition_avg(pselectr*0 + 1,bidr)
            sqmaskr = sqmaskr[:,::-1]
            # now fill in values
            # just fill in the points, don't interp
            w = np.where((sqmask == 0)*(sqmaskr == 1))
            sqphitot[w] = sqphitotr[w]
    
    if(SIMG is not None):
        SIMG.ravel()[pxlst[pselect]] = sqphitot.ravel()[bid]

    return sqphitot

def read_qphimaps(filenames, img_reader, mask, x0, y0, noqs, nophis,PF=None,fileskip=None):
    ''' read qphi maps from filenames with a mask,for noqs nophis
        img_reader must be a function
        x0, y0 are the beam center
    '''
    if fileskip is None:
        fileskip = 1
    Nfiles = len(filenames)//fileskip
    sqphis = np.zeros((Nfiles, noqs, nophis))

    if PF is not None:
        print("read_qphimaps : reading qphimaps")

    for j in range(Nfiles):
        for i in range(fileskip):
            filename = filenames[j*fileskip + i]
            if PF is not None:
                print("iteration {} of {} (file {} of {})".format(j+1,Nfiles,i+1,fileskip))
            img = img_reader(filename,PF=PF)
            if j == 0 and i == 0:
                imgtot = np.zeros_like(img,dtype=float)
            sqphis[j] += qphiavg(img, mask=mask,x0=x0,y0=y0,noqs=noqs,nophis=nophis)
            imgtot += img

    imgtot /= Nfiles

    return sqphis

