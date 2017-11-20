import pickle
from astropy.table import Table
from xdgmm import XDGMM
import astropy.coordinates as coord
import numpy as np
import astropy.units as u
import sys 

def matrixize(data, err):
    """
    vectorize the 2 pieces of data into a 2D mean and 2D covariance matrix
    """
    X = np.vstack(data).T
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([e**2. for e in err]).T
    return X, Xerr

ngauss = int(sys.argv[1])
xdgmmFilename1 = 'gaiasdss.radecpmrapmdec.ngauss' + str(ngauss)
#xdgmmFilename2 = 'gaiasdss.radeclogpmrapmdec.ngauss' + str(ngauss)
xdgmmFilename3 = 'gaiasdss.lbpmlpmb.ngauss' + str(ngauss)
#xdgmmFilename4 = 'gaiasdss.lblogpmlpmb.ngauss' + str(ngauss)

cache_file = 'gaiasdssHaloNew.pkl'

with open(cache_file, "rb") as f:
            res = pickle.load(f)

sdsstbl = Table(res)
rakey = 'ra'
deckey = 'dec'


c = coord.ICRS(ra=sdsstbl[rakey]*u.deg, dec=sdsstbl[deckey]*u.deg,
               pm_ra_cosdec=sdsstbl['pmra_new']*u.mas/u.yr, pm_dec=sdsstbl['pmdec_new']*u.mas/u.yr)
cGal = c.transform_to(coord.Galactic)
pmerr = np.zeros(len(sdsstbl)) + 2.

poserr = np.zeros(len(sdsstbl)) + 4./3600.

#X, Xerr = matrixize([sdsstbl[rakey], sdsstbl[deckey], sdsstbl['pmra_new'], sdsstbl['pmdec_new']],
#                    [sdsstbl['ra_error']/3600., sdsstbl['dec_error']/3600., pmerr, pmerr])
X, Xerr = matrixize([sdsstbl[rakey], sdsstbl[deckey], sdsstbl['pmra_new'], sdsstbl['pmdec_new']],
                    [poserr, poserr, pmerr, pmerr])

xdgmm = XDGMM(method='Bovy')
xdgmm.n_components = ngauss
xdgmm = xdgmm.fit(X, Xerr)
xdgmm.save_model(xdgmmFilename1)


n = 10
nstars = len(sdsstbl)
ras    = np.random.normal(loc=sdsstbl['s_ra1'], scale=sdsstbl['ra_error']/3600., size=[n, nstars])
decs   = np.random.normal(loc=sdsstbl['s_dec1'], scale=sdsstbl['dec_error']/3600., size=[n, nstars])
pmras  = np.random.normal(loc=sdsstbl['pmra_new'], scale=pmerr, size=[n, nstars])
pmdecs = np.random.normal(loc=sdsstbl['pmdec_new'], scale=pmerr, size=[n, nstars])


cGalTest = coord.ICRS(ra=ras*u.deg, dec=decs*u.deg,
                     pm_ra_cosdec=pmras*u.mas/u.yr, pm_dec=pmdecs*u.mas/u.yr)
def std(x, n):
    return np.sqrt(np.sum((x - np.mean(x, axis=0))**2., axis=0)/(n - 1))

std_ls = std(cGalTest.l, n)
std_bs = std(cGalTest.b, n)
std_pml = std(cGalTest.pm_l_cosb, n)
std_pmb = std(cGalTest.pm_b, n)

X, Xerr = matrixize([cGal.l, cGal.b, cGal.pm_l_cosb, cGal.pm_b],
                    [std_ls, std_bs, std_pml, std_pml])

xdgmm = XDGMM(method='Bovy')
xdgmm.n_components = ngauss
xdgmm = xdgmm.fit(X, Xerr)
xdgmm.save_model(xdgmmFilename3)
