import sqlutil
from astropy.table import Table
from os import path
import pickle
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
import astropy.units as units

def dust(l, b):
    c = SkyCoord(l, b, frame='galactic')
    sfd = SFDQuery()
    dust = sfd(c)
    return dust

dustCoeff = {'B': 3.626,
             'V': 2.742,
             'g': 3.303,
             'r': 2.285,
             'i': 1.698,
             'J': 0.709,
             'H': 0.449,
             'K': 0.302,
             'W1':0.18,
             'W2':0.16,
             'G': 2.55}

def queryDatabase(query, columns, post_query, cache_file='db.pickle'):

    #cache_file = "reducedPM2.pickle"

    if not path.exists(cache_file):
        res = sqlutil.get(query.format(','.join(columns), post_query),
                          db='wsdb', host='cappc127.ast.cam.ac.uk',
                          user='gaia_sprint', password='NOT@Gaia_sprint',
                          asDict=True)

        with open(cache_file, "wb") as f:
            pickle.dump(res, f)

    else:
        with open(cache_file, "rb") as f:
            res = pickle.load(f)
    return res



columns = ['treated_as_pointsource', 'pointsource', 'w1_mag', 'w1_mag_err',
          'w2_mag', 'w2_mag_err',

           'g.source_id', 'g.ra', 'g.dec', 'g.parallax', 'g.parallax_error',
           'g.phot_g_mean_flux', 'g.phot_g_mean_flux_error', 'g.l', 'g.b',

           'j_m', 'k_m', 'h_m', 'j_cmsig', 'k_cmsig', 'h_cmsig']


#SELECT * FROM sdss AS s
#    INNER JOIN tmass as t
#        ON s.source_id = t.source_id
#    INNER JOIN unwise as w
#        ON s.obj_id = w.obj_id
#WHERE t.j - t.k > 0.5

query = """select {0} from 
           gaia_dr1_aux.gaia_source_sdssdr9_xm_new as g
           inner join unwise.sdss_forced as u
           on g.objid = u.objid
           inner join gaia_dr1_aux.gaia_source_2mass_xm as t
           on g.source_id = t.source_id
           inner join gaia_dr1.tgas_source as tgas
           on g.source_id = tgas.source_id
           {1}"""

cuts = """where t.j_m - t.k_m > -0.5;"""

res = queryDatabase(query, columns, cuts, cache_file='tgas2massunwise.pkl')

tbl = Table(res)

dustEBV = dust(tbl['l']*units.deg, tbl['b']*units.deg)

jkc = (tbl['j_m'] - dustCoeff['J']*dustEBV) - (tbl['k_m'] - dustCoeff['K']*dustEBV)
w12c = (tbl['w1_mag'] - dustCoeff['W1']*dustEBV) - (tbl['w2_mag'] - dustCoeff['W2']*dustEBV)

good = ~np.isnan(tbl['w1_mag']-tbl['w2_mag']) #& (np.abs(tbl['w1_mag'] - tbl['w2_mag']) < 0.5) & (tbl['j_m'] - tbl['k_m'] < 2.)


plt.hist2d(tbl['w1_mag'][good] - tbl['w2_mag'][good], tbl['j_m'][good] - tbl['k_m'][good], bins=2500, norm=mpl.colors.LogNorm(), cmap='Greys', rasterized=True)
plt.xlim(-0.5, 0.5)
plt.ylim(0.5, 2)
plt.ylabel('tmass j-k')
plt.xlabel('wise 1-2')
plt.savefig('tmasswiseCC_tgas.png')

plt.clf()

plt.hist2d(w12c[good], jkc[good], bins=2500, norm=mpl.colors.LogNorm(), cmap='Greys', rasterized=True)
plt.xlim(-0.5, 0.5)
plt.ylim(0.5, 2)
plt.ylabel('tmass j-k')
plt.xlabel('wise 1-2')
plt.savefig('tmasswiseCC_dustCorrected_tgas.png')


