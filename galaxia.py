import astropy.coordinates as coord
import astropy.units as u
import ebf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys

data = ebf.read('sdssgalaxy.ebf', '/')

c = coord.Galactic(u=data['px']*u.kpc, v=data['py']*u.kpc, w=data['pz']*u.kpc,
                 U=data['vx']*u.km/u.s, V=data['vy']*u.km/u.s, W=data['vz']*u.km/u.s,
                 representation=coord.CartesianRepresentation, differential_cls=coord.CartesianDifferential)
c.set_representation_cls(coord.SphericalRepresentation, s=coord.SphericalCosLatDifferential)


appg = data['sdss_g'] + 5.*np.log10(100.*data['rad'])
appi = data['sdss_i'] + 5.*np.log10(100.*data['rad'])

pm = np.sqrt(c.pm_b**2. + c.pm_l_cosb**2.)
redpm_I = appi + 5.*np.log10(pm.to_value(u.arcsec/u.yr)) + 5.

halo = data['popid'] == 8
bulge = data['popid'] == 9
thickdisk = data['popid'] == 7
thindisk = data['popid'] < 7
highb = np.abs(data['glat']) > 30.

colorGminusI = data['sdss_g'] - data['sdss_i']
colorRedclump = (colorGminusI < 1.5) & (colorGminusI > 1.0)
#indexRedClump = (data['smass'] > data['mtip']) & (data['sdss_i']>0.0) & (data['smass'] > data['mtip']) &
indexRedClump = (data['sdss_i']>0.0) & (data['smass'] > data['mtip']) & (data['sdss_i'] < 0.5)
indices = [halo, bulge, thickdisk, thindisk, highb]
labels = ['halo', 'bulge', 'thickdisk', 'thindisk', 'highb']
color = ['red', 'orange', 'green', 'blue', 'black']


plt.clf()
for i, l, c in zip(indices, labels, color):
    plt.hist(np.log10(pm[i & ~indexRedClump & colorRedclump].value), color=c, bins=100, label=l, log=True, histtype='step', linewidth=2)
    plt.hist(np.log10(pm[i & indexRedClump & colorRedclump].value), color=c, bins=100, linestyle='--', log=True, histtype='step', linewidth=2)
plt.xlabel('log10 proper motion')
plt.legend(loc='best')
plt.ylim(1, 3e5)
plt.savefig('properMotionsHistPopulations.pdf')

plt.clf()
for i, l, c in zip(indices, labels, color):
    plt.hist(redpm_I[i & ~indexRedClump & colorRedclump], color=c, bins=100, label=l, log=True, histtype='step', linewidth=2)
    plt.hist(redpm_I[i & indexRedClump & colorRedclump], color=c, bins=100, linestyle='--', log=True, histtype='step', linewidth=2)
plt.xlabel('reduced proper motion SDSS I')
plt.legend(loc='best')
plt.ylim(1, 3e5)
plt.savefig('reducedProperMotionsHistPopulations.pdf')



xlim = [-1, 3]
ylim = [18, 0]

plt.clf()
plt.hist2d(data['sdss_g'] - data['sdss_i'], data['sdss_i'], bins=250, cmap='Greys', norm=mpl.colors.LogNorm())
#plt.scatter(data['sdss_g'][indexRedClump] - data['sdss_i'][indexRedClump], data['sdss_i'][indexRedClump], s=1, c='blue', lw=0)
#plt.gca().invert_yaxis()
plt.xlabel('sdss g-i')
plt.ylabel('sdss i')
plt.xlim(xlim)
plt.ylim(12.5, -5)
plt.savefig('SDSSCMD_GII.pdf', rasterized=True)


plt.hist2d(data['sdss_g'][~indexRedClump & colorRedclump]- data['sdss_i'][~indexRedClump & colorRedclump], data['sdss_i'][~indexRedClump & colorRedclump], cmap='Greens', bins=100, norm=mpl.colors.LogNorm())
plt.xlabel('sdss g-i')
plt.ylabel('sdss i')
plt.xlim(xlim)
plt.ylim(12.5, -5)
plt.savefig('SDSSCMD_GII_NOTRC.pdf', rasterized=True)

plt.hist2d(data['sdss_g'][indexRedClump & colorRedclump]- data['sdss_i'][indexRedClump & colorRedclump], data['sdss_i'][indexRedClump & colorRedclump], cmap='Reds', bins=100, norm=mpl.colors.LogNorm())
plt.xlabel('sdss g-i')
plt.ylabel('sdss i')
plt.xlim(xlim)
plt.ylim(12.5, -5)
plt.savefig('SDSSCMD_GII_RC.pdf', rasterized=True)
#sys.exit()

plt.clf()
plt.hist2d(data['sdss_g'] - data['sdss_i'], redpm_I, bins=250, cmap='Greys', norm=mpl.colors.LogNorm())
####plt.gca().invert_yaxis()
plt.xlabel('sdss g-i')
plt.ylabel('reduced pm sdss i')
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig('reducedPM_SDSS.pdf')

sdsscolor = data['sdss_g'] - data['sdss_i']
print xlim, ylim
print np.min(sdsscolor), np.max(sdsscolor)
good = (sdsscolor >= xlim[0]) & (sdsscolor <= xlim[1]) & (redpm_I >= ylim[1]) & (redpm_I <= ylim[0])
#fig, ax = plt.subplots()
for i, l, c in zip(indices, labels, color):
    plt.clf()
    #current_x = sdsscolor[i]
    #current_y = redpm_I[i]
    #print len(current_x)
    #good = (current_x > xlim[0]) & (current_x < xlim[1]) & (current_y < ylim[0]) & (current_y > ylim[1])
    #print np.sum(good)
    #joint_kws = {'bins':10}
    #marginal_kws = {'bins':100}
    plt.hist2d(data['sdss_g'][i] - data['sdss_i'][i], redpm_I[i], bins=250, cmap='Greys', norm=mpl.colors.LogNorm())
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('sdss g-i')
    plt.ylabel('reduced pm sdss i')
    #g = sns.jointplot(data['sdss_g'][i] - data['sdss_i'][i], redpm_I[i], kind='hex', color=c, xlim=xlim, ylim=ylim, joint_kws=joint_kws, marginal_kws=marginal_kws)
    #g.plot_joint(plt.hist2d, bins=250, cmap='Greys', norm=mpl.colors.LogNorm(), zorder=-10)
    #g.set_axis_labels('sdss g-i', 'reduced pm sdss i')
    #plt.gca().invert_yaxis()
    #g.set_axis_limits(xlim, ylim)
    plt.savefig('reducedPM_SDSS_' + l + '.pdf', rasterized=True)
