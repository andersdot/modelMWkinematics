impot pickle
from astropy.table import Table
from xdgmm import XDGMM
import astropy.coordinates as coord

def matrixize(data, err):
    """
    vectorize the 2 pieces of data into a 2D mean and 2D covariance matrix
    """
    X = np.vstack(data).T
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([e**2. for e in err]).T
    return X, Xerr

ngauss = 512
xdgmmFilename1 = 'gaiasdss.radecpmrapmdec.ngauss' + str(ngauss)
#xdgmmFilename2 = 'gaiasdss.radeclogpmrapmdec.ngauss' + str(ngauss)
xdgmmFilename3 = 'gaiasdss.lbpmlpmb.ngauss' + str(ngauss)
#xdgmmFilename4 = 'gaiasdss.lblogpmlpmb.ngauss' + str(ngauss)

cache_file = 'gaiasdssHaloNew.pkl'

with open(cache_file, "rb") as f:
            res = pickle.load(f)

sdsstbl = Table(res)

c = coord.ICRS(ra=sdsstbl['s_ra1']*u.deg, dec=sdsstbl['s_dec1']*u.deg,
               pm_ra_cosdec=sdsstbl['pmra_new']*u.mas/u.yr, pm_dec=sdsstbl['pmdec_new']*u.mas/u.yr)
cGal = c.transform_to(coord.Galactic)
pmerr = np.zeros(len(sdsstbl)) + 2.


X, Xerr = matrixize([sdsstbl['s_ra1'], sdsstbl['s_dec1'], sdsstbl['pmra_new'], sdsstbl['pmdec_new']],
                    [sdsstbl['ra_error']/3600., sdsstbl['dec_error']/3600., pmerr, pmerr])

xdgmm = XDGMM(method='Bovy')
xdgmm.n_components = ngauss
xdgmm = xdgmm.fit(X, Xerr)
xdgmm.save_model(xdgmmFilename1)


X, Xerr = matrixize(cGal.l, cGal.b, cGal.pm_l_cosb, cGal.pm_b],
                    [sdsstbl['ra_error']/3600., sdsstbl['dec_error']/3600., pmerr, pmerr])

xdgmm = XDGMM(method='Bovy')
xdgmm.n_components = ngauss
xdgmm = xdgmm.fit(X, Xerr)
xdgmm.save_model(xdgmmFilename3)
