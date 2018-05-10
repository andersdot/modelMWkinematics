import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_product, matrix_transpose, rotation_matrix
from astropy.table import Table
from astropy.io import fits
import pickle
import scipy.interpolate as scint
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import gala.coordinates as gc
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from astropy_healpix import HEALPix
import time
import os.path
import sys
import csv
import schwimmbad
from astropy.io import ascii
import h5py

class ArbitraryPoleFrame(coord.BaseCoordinateFrame):

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'phi1'),
            coord.RepresentationMapping('lat', 'phi2'),
            coord.RepresentationMapping('distance', 'distance')],
        coord.SphericalCosLatDifferential:[
            coord.RepresentationMapping('d_lon_coslat', 'pm_phi1_cosphi2'),
            coord.RepresentationMapping('d_lat', 'pm_phi2'),
            coord.RepresentationMapping('d_distance', 'radial_velocity')],
        coord.SphericalDifferential: [
            coord.RepresentationMapping('d_lon', 'pm_phi1'),
            coord.RepresentationMapping('d_lat', 'pm_phi2'),
            coord.RepresentationMapping('d_distance', 'radial_velocity')]
    }

    pole = coord.CoordinateAttribute(frame=coord.ICRS)
    roll = coord.QuantityAttribute(default=0*u.degree)


@frame_transform_graph.transform(coord.DynamicMatrixTransform, coord.ICRS, ArbitraryPoleFrame)
def icrs_to_arbpole(icrs_coord, arbpole_frame):

    roll = arbpole_frame.roll
    pole = arbpole_frame.pole

    # Align z(new) with direction to M31
    mat1 = rotation_matrix(-pole.dec, 'y')
    mat2 = rotation_matrix(pole.ra, 'z')
    mat3 = rotation_matrix(roll, 'z')
    mat4 = rotation_matrix(90*u.degree, 'y')
    R = matrix_product(mat4, mat1, mat2, mat3)

    return R


@frame_transform_graph.transform(coord.DynamicMatrixTransform, ArbitraryPoleFrame, coord.ICRS)
def arbpole_to_icrs(arbpole_coord, icrs_frame):
    return matrix_transpose(icrs_to_arbpole(None, arbpole_coord))


def plotFootprint(x, y, filename, nbins=250):
    fig2, axes = plt.subplots()#1, 3, figsize=(15, 5))
    H, xedges, yedges = np.histogram2d(x, y, bins=(nbins, nbins))
    axes.pcolormesh(xedges, yedges, H.T, norm=mpl.colors.LogNorm(vmin=10, vmax=1000))
    plt.ylabel('phi2')
    plt.xlabel('phi1')
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(10,10)
    fig.savefig(filename, rasterized=True)

def signal2noise(phi1, phi2, pmphi1, pmphi2, p1, deltaPhi1 = 5.*u.deg, deltaPhi2 = 0.5*u.deg,
                 deltaPMphi1 = 15.*u.mas/u.yr, deltaPMphi2 = 5.*u.mas/u.yr,
                 phi2MaxBackground = 5.0*u.deg, phi2MinBackground = 0.5*u.deg,
                 histBinPM = 1.0*u.mas/u.yr, histBinPhi = 0.1*u.deg,
                 minstars=1000., minFracBackground = 0.7, plotSNthreshold = 15.,
                 detectionThreshold = 3., filename='sag_pm'):
    phi1min = p1 - deltaPhi1 #80*u.deg #-60.*u.deg #105*u.deg
    phi1max = p1 + deltaPhi1 #120*u.deg #-20.*u.deg #115*u.deg
    phi2min = -deltaPhi2
    phi2max = deltaPhi2

    phi2max_b = phi2MaxBackground
    phi2min_b = phi2MinBackground

    dphi1 = 2.*deltaPhi1
    dphi2 = 2.*deltaPhi2
    dphi2_b = phi2max_b - phi2min_b


    dpm1 = deltaPMphi1.value#*u.mas/u.yr
    dpm2 = deltaPMphi2.value#*u.mas/u.yr
    deltapm = histBinPM.value#*u.mas/u.yr

    #histograms for determining area of background to area of signal
    deltaphi = histBinPhi.value

    signal_edges_pm1 = [-dphi1/2., dphi1/2.]
    signal_edges_pm2 = [-dphi2/2., dphi2/2.]
    background_edges_pm1 = [-dphi1/2., dphi1/2.]
    background_edges_pm2 = [-(dphi2_b + dphi2)/2., -dphi2/2.]
    backgtound_edges_pm2 = [dphi2/2., (dphi2_b+dphi2)/2.]

    signal_indices = (phi1 >= phi1min) & (phi1 <= phi1max) & (phi2 >= phi2min) & (phi2 <= phi2max) #& colorCut & magCut

    #check that there are more then minstars in the signal sample
    if np.sum(signal_indices) < minstars:
        return None, None, 0
    else:
        background_indices = (phi1 >= phi1min) & (phi1 <= phi1max) & \
                             np.logical_or((phi2 > phi2min_b) & (phi2 <= phi2max_b),
                                           (phi2 >= -phi2max_b) & (phi2 <= -phi2min_b)) #& colorCut & magCut

        phi1_edges = np.arange(phi1min.value, phi1max.value+000.1, deltaphi)
        phi2_edges = np.arange(-phi2max_b.value, phi2max_b.value+000.1, deltaphi)


        histBack_pos, xe, ye = np.histogram2d(phi1[background_indices].value,
                                              phi2[background_indices].value,
                                              bins=[phi1_edges, phi2_edges])
        frac_back_in_footprint = np.sum(histBack_pos > 0)/np.float(np.sum(histBack_pos >= 0))

        #check that at least some fraction of the background is in the footprint
        if frac_back_in_footprint < minFracBackground:
            return None, None, 0
        else:
            pm1_edges = np.arange(-dpm1, dpm1+0.001, deltapm)
            pm2_edges = np.arange(-dpm2, dpm2+0.001, deltapm)

            H, xedges, yedges = np.histogram2d(pmphi1[signal_indices].value,
                                               pmphi2[signal_indices].value,
                                               bins=[pm1_edges, pm2_edges])
            Hback, xeback, yeback     = np.histogram2d(pmphi1[background_indices].value,
                                               pmphi2[background_indices].value,
                                               bins=[pm1_edges, pm2_edges])

            histSig_pos, xe, ye  = np.histogram2d(phi1[signal_indices].value,
                                                          phi2[signal_indices].value,
                                                          bins=[phi1_edges, phi2_edges])

            ycenters = (yedges[1:] + yedges[:-1])*0.5
            xcenters = (xedges[1:] + xedges[:-1])*0.5
            ybackcenters = (yeback[1:] + yeback[:-1])*0.5


            areaNorm = np.sum(histSig_pos > 0)/np.float(np.sum(histBack_pos > 0))
            finalhist = gaussian_filter(H - Hback*areaNorm, sigma=1.5)
            #finalhist = H - Hback*areaNorm

            H1D = np.sum(H, axis=1)
            Hback1D = np.sum(Hback*areaNorm, axis=1)

            Hdiff = H1D - Hback1D
            Signal_to_noise = Hdiff/np.sqrt(Hback1D)

            #if np.max(Signal_to_noise) > plotSNthreshold:
            #    plotPMdiff(xcenters, Signal_to_noise, dpm1, xedges, yedges, finalhist, phi1,
            #                   signal_indices, phi2, phi1_edges, phi2_edges, background_indices,
            #                   pmphi1, pmphi2, pm1_edges, pm2_edges, filename, frac_back_in_footprint)
            detected = Signal_to_noise >= detectionThreshold

            n = np.sum(detected)
            if not n: n = 0
            detected = Signal_to_noise == np.max(Signal_to_noise)
            return xcenters[detected], Signal_to_noise[detected], n

def plotPMdiff(xcenters, Signal_to_noise, dpm1, xedges, yedges, finalhist, phi1,
               signal_indices, phi2, phi1_edges, phi2_edges, background_indices,
               pmphi1, pmphi2, pm1_edges, pm2_edges, filename, frac_back_in_footprint):

    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid((4, 2), (0, 0))
    ax2 = plt.subplot2grid((4, 2), (0, 1))
    ax3 = plt.subplot2grid((4, 2), (1, 0))
    ax4 = plt.subplot2grid((4, 2), (1, 1))
    ax5 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
    ax6 = plt.subplot2grid((4, 2), (3, 0), colspan=2)
    plt.suptitle('Fraction of Background in Footprint: {0:0.2f}'.format(frac_back_in_footprint))

    ax6.plot(xcenters, Signal_to_noise, label='S/N')
    ax6.axhline(5, color='black', linestyle='--')
    ax6.set_xlim(-dpm1, dpm1)
    ax6.set_ylim(0,)
    plt.legend()

    blah = ax5.pcolormesh(xedges, yedges, finalhist.T, vmin=0) #, norm=mpl.colors.LogNorm(vmin=1, vmax=10))
    ax5.set_xlabel('pm1')
    ax5.set_ylabel('pm2')
    ax5.axhline(0.0)

    ax1.hist2d(phi1[signal_indices], phi2[signal_indices], bins=[phi1_edges, phi2_edges])
    ax3.hist2d(phi1[background_indices], phi2[background_indices], bins=[phi1_edges, phi2_edges])

    (counts, xedges, yedges, axis1) = ax2.hist2d(pmphi1[signal_indices], pmphi2[signal_indices],
                                                    bins=[pm1_edges, pm2_edges], norm=mpl.colors.LogNorm())
    (counts, xedges, yedges, axis2) = ax4.hist2d(pmphi1[background_indices], pmphi2[background_indices],
                                                    bins=[pm1_edges, pm2_edges], norm=mpl.colors.LogNorm())

    ax2.set_title('signal')
    ax4.set_title('background')
    fig.colorbar(axis1, ax=ax2)
    fig.colorbar(axis2, ax=ax4)

    for axis in [ax1, ax3]:
        axis.set_xlabel('phi1')
        axis.set_ylabel('phi2')
    for axis in [ax2, ax4]:
        axis.set_xlabel('pm1')
        axis.set_ylabel('pm2')

    plt.tight_layout()
    plt.savefig(filename + '.png', figsize=(10, 15))
    plt.close('all')

def sphereSelection(x, y, cenx, ceny, radius=0.5*u.deg):
    indices = np.zeros(len(x))
    for cx, cy in zip(cenx, ceny):
        tempArray = (x-cx)**2. + (y-cy)**2. <= radius**2.
        indices =  np.any([tempArray, indices], axis=0)
    return indices

def reflex(c):
    c = coord.SkyCoord(c)
    v_sun = coord.Galactocentric.galcen_v_sun
    observed = c.transform_to(coord.Galactic)
    rep = observed.cartesian.without_differentials()
    rep = rep.with_differentials(observed.cartesian.differentials['s'] + v_sun)
    return coord.Galactic(rep).transform_to(c.frame)


class Worker(object):
    #set output path for file written by all processes
    def __init__(self, output_path, distance):
        self.output_path = output_path
        self.distance = distance

    #tell processes to write the file
    def callback(self, result):
        lpole, bpole, p1, muphi1, signal_to_noise, n = result
        if muphi1 is not None:
            with open(self.output_path, 'a') as f:
                for l, b, p, mu, s2n, num in zip(lpole, bpole, p1, muphi1, signal_to_noise, n):

                    f.write(','.join(['{:10.6f}'.format(x) for x in [l, b, p, mu, s2n, num]]) + '\n')

    #when called, do the work
    def __call__(self, task):
        print('number of tasks on a worker: ', np.shape(task))
        return self.work(task)

    #work for each process
    def work(self, task):
        #lpole = centers.l
        #bpole = centers.b

        cmCut = False
        #define phi1 array will be the search centers along the great circle
        phiShift = 2.5*u.deg #how much to shift each window by
        phi1_search_array = np.arange(0, 360+0.0001, phiShift.value)*u.deg
        deltaPhi1 = 5.*u.deg #width of window/2
        deltaPhi2 = 0.125*u.deg #height of window/2
        starttime = time.clock()
        data = Table()
        with h5py.File('nearbyStars.hdf5') as f:
            #columns = list(f['table/columns'])
            columns = ['ra', 'dec', 'pmra', 'pmdec', 'parallax']
            for c in columns: data[c] = f['table/columns/{0}/data'.format(c)][:]

        c = coord.SkyCoord(ra=data['ra']*u.deg, dec=data['dec']*u.deg,
                   pm_ra_cosdec = data['pmra']*u.mas/u.yr, pm_dec= data['pmdec']*u.mas/u.yr,
                   distance=1./data['parallax']*u.kpc) #1./ds.parallax.values*u.kpc)
        c = reflex(c)


        thistime = time.clock()
        print('time to read in data :', thistime - starttime)

        xkey = 'ra'
        ykey = 'dec'
        pmxkey = 'pmra'
        pmykey = 'pmdec'
        filename_pre = 'test'

        #remove clusters, radius 6 arcminutes
        #filename = 'CompiledSatCatalogv2_gabriel.csv'
        #clusterdata = ascii.read(filename)
        #ind = sphereSelection(data['ra'], data['dec'], clusterdata['ra'], clusterdata['dec'], radius=0.5)

        #for a given pole, set up lists of possible phi1s that have detections
        phi1_set = []
        muphi1_set = []
        signal_to_noise_set = []
        n_set = []
        lpole_set = []
        bpole_set = []
        npoles = 0
        for t in task:
            begLoop = time.clock()
            lpole = t.l
            bpole = t.b
            #define the pole where the equator is the great circle
            pole = coord.Galactic(l=lpole, b=bpole)
            frame = ArbitraryPoleFrame(pole=pole)
            #align frame with pole
            newframe = c.transform_to(frame)

            pmphi1 = newframe.pm_phi1_cosphi2
            pmphi2 = newframe.pm_phi2
            phi1 = newframe.phi1 #.wrap_at(180*u.deg) #.wrap_at(180*u.deg)
            phi2 = newframe.phi2

            for p1 in phi1_search_array:
                filename = filename_pre+'_lpole{0:0.2f}_bpole{1:0.2f}_phi1{2:03d}'.format(lpole.value, bpole.value, int(p1.value))
                muphi1, signal_to_noise, number_of_detections = signal2noise(phi1, phi2, pmphi1, pmphi2, p1,
                                                                             filename=filename, deltaPhi1 = deltaPhi1, deltaPhi2=deltaPhi2)
                if number_of_detections > 0:
                    phi1_set.append(p1.value)
                    muphi1_set.extend(muphi1)
                    signal_to_noise_set.extend(signal_to_noise)
                    n_set.append(number_of_detections)
                    lpole_set.append(lpole.value)
                    bpole_set.append(bpole.value)
                #if number_of_detections > 1:
                #    phi1_set.extend(p1.value)
                #    muphi1_set.extend(muphi1)
                #    signal_to_noise_set.extend(signal_to_noise)
                #    n_set.extend(number_of_detections)
                #    lpole_set.extend(lpole.value)
                #    bpole_set.extend(bpole.value)
            endLoop = time.clock()
            npoles += 1
            if (npoles % 100) == 0:
                print('time for each pole: ', endLoop - begLoop)
                print('number of poles searched: ', npoles)

        return lpole_set, bpole_set, phi1_set, muphi1_set, signal_to_noise_set, n_set

def main(pool, distance=20, filename='output_file.txt', nside=64):

    #calculate pole centers
    hp = HEALPix(nside=nside, order='ring', frame=coord.Galactic())
    centers = hp.healpix_to_skycoord(np.arange(0, hp.npix))
    #only keep poles above equator to not redo calculate with opposite pole
    centers = centers[centers.b >= 0.0*u.deg]
    #pad centers to match to number of processors
    ncenters = centers.shape[0]
    nprocs = pool.size
    if nprocs == 0.:
        nprocs = 1
    ncenters_per_proc = np.ceil(ncenters/float(nprocs))
    npads = nprocs*ncenters_per_proc - ncenters
    skypad = coord.SkyCoord(np.zeros(int(npads)), np.zeros(int(npads)), frame='galactic', unit=u.deg)
    centers = coord.concatenate((centers, skypad))
    #reshape centers so each worker gets a block of them
    centers = centers.reshape(nprocs, int(ncenters_per_proc))
    print( 'number of workers: ', nprocs)
    print( 'number of poles: ', ncenters)
    print( 'number of poles per worker: ', ncenters_per_proc)
    print( 'number of poles added as padding: ', npads)
    print( 'shape of poles array: ', np.shape(centers))
    #instantiate worker with filename results will be saved to
    worker = Worker(filename, args.dist)
    #you better work !
    for r in pool.map(worker, list(centers), callback=worker.callback):
        pass

if __name__ == '__main__':

    import schwimmbad
    from argparse import ArgumentParser
    parser = ArgumentParser(description = 'Search for cold streams in stellar density maps.')
    parser.add_argument('--distance', dest='dist', default=20, type=int, help='distance to stream in kpc')
    parser.add_argument('--nside', dest='nside', default=64, type=int, help='determines number of poles: 12*nside^2')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ncores', dest='n_cores', default=1, type=int, help='number of processes (uses multiprocessing).')
    group.add_argument('--mpi', dest='mpi', default=False, action='store_true', help='run with mpi.')
    args = parser.parse_args()
    print(args)
    #intelligently choose pool based on command line arguments passed

    with schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores) as pool:
        filename = 'detections_distance{0:02d}_nside{1:03d}_SN3_dphi2_0.25_nocmcut.txt'.format(args.dist, args.nside)
        #start running
        main(pool, distance=args.dist, filename=filename, nside=args.nside)
    print('finished')
