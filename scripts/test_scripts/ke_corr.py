"""
Plots the kinetic energy of the total system along with cold, hot and beam electrons.
Appropriate files must be chosen using 'file_name' and 'path'.
Use %reset -f to clear the python workspace.
Data File Invoked: ke_1024.txt
Run as:
"""
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib as mp
import os.path
from os.path import join as pjoin
from scipy.constants import value as constants

file_name = 'ke_1024.txt'
path = '../../data/data_test/'
path_fig = '../../data/data_test/figs'
# -------------------------- Comments ------------------------------------------
# input parameters specific file
# path to the data folder is ../data/data002_vd_20/files/ke_1024.txt for vd=20
# path to the data folder is ../data/data001_vd_80/files/ke_1024.txt for vd=80
# vd is an input parameter to run this file and that must be provided along with others.
#++++++++++++++++++++ UNPACK ++++++++++++++++++++++++++++++++++++++++++++++++++
if os.path.exists(pjoin(path,file_name)):
    t,kei,kec,keh,keb = np.loadtxt(pjoin(path,file_name),unpack=True)
else:
    print('No data')
    exit()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -------- NORMALIZE THE INDIVIDUAL KINETIC ENERGIES W.R.T THE CODE------------
"""
# In the code velocities were normalized while the mass was not normalized.
# Further, the kinetic energy was converted into eV unit. Hence, here the data
# is deconverted by multiplying by e and divided by me to be a normalized one.
"""
eps0 = constants('electric constant')
e = constants('elementary charge')
me = constants('electron mass')
#-----------------------------------------------------------------------------
# Choose the parameters from the appropriate input file
Tec = 1*e;
n0 = 1E10;
nec0 = n0;
#-----------------------------------------------------------------------------
wpec = np.sqrt((nec0*e**2)/(eps0*me)) # Cold electron Plasma Frequency
LDC = np.sqrt(eps0*Tec/(nec0*e**2)) # Cold electron Debye Length

# Total thermal energy of all the cold electrons
vth_ec = wpec*LDC # Thermal velocity of the cold electrons
nParticlesE = 5000000 # Total number of simulation particles of cold electrons
electron_cold_spwt = (nec0*1*1024*LDC)/(nParticlesE); # Cold electron specific weight (spwt = density*DX*(domain.xl)*LDC)/(nParticlesE)
# Total Thermal Energy = Thermal Energy of a single real electron
#                        *Number of real particles represented by a single pseudo electron (or simulation electron)
#                        *Number Psudo Particles
Th_ec = Tec*electron_cold_spwt*nParticlesE
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# By multiplying (vth_ec**2) we unnormalize the normalized velocity in the calculation of KE
# Then, by multiplying e we de-convert it eV to Joules unit.
# Then, divide by the total thermal energy of cold electrons to once again normalize KE, overall.
# Thus, the KE is now normalized by Total Thermal Energy of cold electrons
kei = kei*((vth_ec)**2)*(e/Th_ec)
kec = kec*((vth_ec)**2)*(e/Th_ec)
keh = keh*((vth_ec)**2)*(e/Th_ec)
keb = keb*((vth_ec)**2)*(e/Th_ec)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ke = kec+keh+keb # Total kinetic energy of the electrons
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#print('beam to cold eletron correlation: ', np.corrcoef(keb, kec))
#rint('beam to hot eletron correlation: ', np.corrcoef(keb, keh))
#print('cold to hot eletron correlation: ', np.corrcoef(kec, keh))
corr_bc = np.corrcoef(keb, kec)
corr_bh = np.corrcoef(keb, keh)
corr_ch = np.corrcoef(kec, keh)
#print('b-c correlation coefficient: {:.4f}',format(corr))

figsize = np.array([200,200/1.618]) #Figure size in mm (FOR SINGLE FIGURE)
dpi = 1200                        #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Plot the figure
fig, ax = plt.subplots(1,3,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)

ax[0].scatter(keb,kec, c='blue',marker='.')
ax[0].set_xlabel('$KE_{b}$')
ax[0].set_ylabel('$KE_{c}$')
#ax1.legend(loc='upper right',framealpha=0.5)

ax[1].scatter(keb,keh, c='red',marker='.')
ax[1].set_xlabel('$KE_{b}$')
ax[1].set_ylabel('$KE_{h}$')

ax[2].scatter(kec,keh, c='red',marker='.')
ax[2].set_xlabel('$KE_{c}$')
ax[2].set_ylabel('$KE_{h}$')
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.savefig(pjoin(path_fig,'ke_corr.png'),dpi=dpi)

#plt.figure(2)
#plt.semilogx(t,ke,'b',linestyle='-',linewidth=1.0)

plt.show()
