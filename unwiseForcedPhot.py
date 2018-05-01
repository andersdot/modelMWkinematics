import sqlutil
from astropy.table import Table
from os import path
import pickle
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt



def queryDatabase(query, columns, post_query, cache_file='db.pickle'):

    #cache_file = "reducedPM2.pickle"

    if not path.exists(cache_file):
        res = sqlutil.get(query.format(','.join(columns), post_query),
                          db='wsdb', host='cappc127.ast.cam.ac.uk',
                          user=user, password=password,
                          asDict=True)

        with open(cache_file, "wb") as f:
            pickle.dump(res, f)

    else:
        with open(cache_file, "rb") as f:
            res = pickle.load(f)
    return res



columns = ['treated_as_pointsource', 'pointsource', 'w1_mag', 'w1_mag_err',
          'w2_mag', 'w2_mag_err', 'w3_mag', 'w3_mag_err', ' w4_mag', 'w4_mag_err',
          'has_wise_phot',

           'g.source_id', 'g.ra', 'g.dec', 'g.parallax', 'g.parallax_error',
           'g.phot_g_mean_flux', 'g.phot_g_mean_flux_error', 'g.l', 'g.b',

           'psfmag_g', 'psfmag_r', 'psfmag_i', 'psfmagerr_g', 'psfmagerr_r',
           'psfmagerr_i', 'pmra_new', 'pmdec_new']

           #'j_m', 'k_m', 'h_m', 'j_cmsig', 'k_cmsig', 'h_cmsig']

query = """select {0} from unwise.sdss_forced as u,
           gaia_dr1_aux.gaia_source_sdssdr9_xm_new as g
           {1}"""


#query = """select {0} from unwise.sdss_forced as u,
#           gaia_dr1_aux.gaia_source_sdssdr9_xm_new as g,
#           gaia_dr1_aux.gaia_source_2mass_xm as t
#           {1}"""

cuts = """where u.objid = g.objid and
          g.psfmag_g - g.psfmag_i > 0.5;"""

res = queryDatabase(query, columns, cuts, cache_file='gaiaunwise.pkl')

columns = ['g.source_id', 'j_m', 'k_m', 'h_m', 'j_cmsig', 'k_cmsig', 'h_cmsig', 'psfmag_g', 'psfmag_r', 'psfmag_i', 'pmra_new', 'pmdec_new']

query = """select {0} from gaia_dr1_aux.gaia_source_2mass_xm as t,
           gaia_dr1_aux.gaia_source_sdssdr9_xm_new as g
           {1}"""
cuts="""where t.source_id = g.source_id and
        g.psfmag_g - g.psfmag_i > 0.5;"""

res2 = queryDatabase(query, columns, cuts, cache_file='gaiasdssRed.pkl')


tblunwise = Table(res)
tbl2mass = Table(res2)
"""
tblunwise = tblunwise[np.argsort(tblunwise['source_id'])]
tbl2mass = tbl2mass[np.argsort(tbl2mass['source_id'])]

ind = np.in1d(tbl2mass['source_id'], tblunwise['source_id'])
"""

plt.hist2d(tblunwise['psfmag_g'] - tblunwise['psfmag_i'], tblunwise['w1_mag'] - tblunwise['w2_mag'], bins=250, norm=mpl.Color.LogNorm(), cmap='Greys', rasterized=True)
plt.xlabel('sdss g-i')
plt.ylabel('wise 1-2')
plt.savefig('sdssWiseCC.png')

plt.hist2d(tbl2mass['psfmag_g'] - tbl2mass['psfmag_i'], tbl2mass['j_m'] - tbl2mass['k_m'], bins=250, norm=mpl.Color.LogNorm(), cmap='Greys', rasterized=True)
plt.xlabel('sdss g-i')
plt.ylabel('2mass j-k')
plt.savefig('sdss2massCC.png')

