import pickle
from astropy.table import Table
from xdgmm import XDGMM
import astropy.coordinates as coord
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plotvector(mean, var, step=0.001):
    """
    mean, var should be *projected* to the 2-d space in which plotting is about to occur
    """
    assert mean.shape == (2,)
    assert var.shape ==	(2, 2)
    ts = np.arange(0, 2. * np.pi, step) #magic
    w, v = np.linalg.eigh(var)
    print w, v
    ps = np.sqrt(w[0]) * (v[:, 0])[:,None] * (np.cos(ts))[None, :] + \
         np.sqrt(w[1]) * (v[:, 1])[:,None] * (np.sin(ts))[None, :] + \
	     mean[:, None]
    return ps

def logNegative(x):
    return np.sign(x)*np.log10(np.abs(x) + 1.)

def plotGMM(xdgmm, ax=None, c='k', lw=1, labels='prior', indices=[[0,1]]):
    for gg in range(xdgmm.n_components):
        if xdgmm.weights_[gg] == np.max(xdgmm.weights_):
            label = 'gaussian mixture model'
        else:
            label = None
        for axis, ind, l in zip(ax, indices, labels):
            mean = xdgmm.means_[gg][ind]
            var = xdgmm.covariances_[gg][ind[0]:ind[1]+1,ind[0]:ind[1]+1]
            weight = xdgmm.weights_[gg]
            points = plotvector(mean, var)
            if l == 'pm':
                x = points[0,:]
                y = points[1,:]
                if axis: axis.plot(logNegative(x), logNegative(y), c, lw=lw,
                        alpha=weight/np.max(xdgmm.weights_), label=label, rasterized=True)
                else: plt.plot(logNegative(x), logNegative(y), c, lw=lw,
                        alpha=weight/np.max(xdgmm.weights_), label=label, rasterized=True)

            else:
                if axis: axis.plot(points[0,:], points[1,:], c, lw=lw,
                alpha=weight/np.max(xdgmm.weights_), label=label, rasterized=True)
                else: plt.plot(points[0,:], points[1,:], c, lw=lw,
                alpha=weight/np.max(xdgmm.weights_), label=label, rasterized=True)

def matrixize(data, err):
    """
    vectorize the 2 pieces of data into a 2D mean and 2D covariance matrix
    """
    X = np.vstack(data).T
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([e**2. for e in err]).T
    return X, Xerr

def std(x, n):
    return np.sqrt(np.sum((x - np.mean(x, axis=0))**2., axis=0)/(n - 1))

if __name__ == '__main__':
    ngauss = 128
    xdgmmFilename1 = 'gaiasdss.radecpmrapmdec.ngauss' + str(ngauss)
    #xdgmmFilename2 = 'gaiasdss.radeclogpmrapmdec.ngauss' + str(ngauss)
    xdgmmFilename3 = 'gaiasdss.lbpmlpmb.ngauss' + str(ngauss)
    #xdgmmFilename4 = 'gaiasdss.lblogpmlpmb.ngauss' + str(ngauss)

    xdgmmfilename = xdgmmFilename1

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes = axes.flatten()
    cache_file = 'gaiasdssHaloNew.pkl'
    with open(cache_file, "rb") as f:
            res = pickle.load(f)
    pmlim = 2
    nbins = 500
    bins = [np.linspace(-pmlim, pmlim, nbins), np.linspace(-pmlim, pmlim, nbins)]
    sdsstbl = Table(res)
    axes[0].hist2d(sdsstbl['ra'], sdsstbl['dec'], bins=100, norm=mpl.colors.LogNorm())
    axes[1].hist2d(logNegative(sdsstbl['pmra_new']), logNegative(sdsstbl['pmdec_new']), bins=bins, norm=mpl.colors.LogNorm())
    try:
        xdgmm = XDGMM(filename=xdgmmfilename)
        plotGMM(xdgmm, ax = axes, indices=[[0,1],[2,3]], labels=['pos', 'pm'])
        axes[0].set_xlabel('ra')
        axes[0].set_ylabel('dec')
        axes[1].set_xlabel('pmra')
        axes[1].set_ylabel('pmdec')
        axes[1].set_xlim(-2, 2)
        axes[1].set_ylim(-2, 2)
        fig.savefig('gmmradec.pdf')
    except IOError:
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


        n = 10
        nstars = len(sdsstbl)
        ras    = np.random.normal(loc=sdsstbl['s_ra1'], scale=sdsstbl['ra_error']/3600., size=[n, nstars])
        decs   = np.random.normal(loc=sdsstbl['s_dec1'], scale=sdsstbl['dec_error']/3600., size=[n, nstars])
        pmras  = np.random.normal(loc=sdsstbl['pmra_new'], scale=pmerr, size=[n, nstars])
        pmdecs = np.random.normal(loc=sdsstbl['pmdec_new'], scale=pmerr, size=[n, nstars])


        cGalTest = coord.ICRS(ra=ras*u.deg, dec=decs*u.deg,
                         pm_ra_cosdec=pmras*u.mas/u.yr, pm_dec=pmdecs*u.mas/u.yr)

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
