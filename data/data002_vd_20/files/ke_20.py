"""
The code plots the kinetic energy of the three electron system 
containing cold, hot and beam electron species. Appropriate files 
must be chosen using file_name and path.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import os.path
from os.path import join as pjoin
from scipy.constants import value as constants

file_name = 'ke_1024.txt'
path = './'

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
#kei = kei*(e/me) 
#kec = kec*(e/(me)) 
#keh = keh*(e/(me)) 
#keb = keb*(e/(me)) 
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
fig,ax = plt.subplots(2,2,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)

ax[0][0].plot(t,ke,'r',linestyle='-',linewidth=1.0,label='KE')
ax[0][0].set_xlabel('$\omega_{pe}t$')
ax[0][0].set_ylabel('KE')
ax[0][0].legend(loc='upper right',framealpha=0.5)

ax[0][1].plot(t,kec,'g',linestyle='-',linewidth=1.0, label='$KE_{C}$')
ax[0][1].set_xlabel('$\omega_{pe}t$')
ax[0][1].set_ylabel('$KE_{C}$')
ax[0][1].legend(loc='upper right',framealpha=0.5)

ax[1][0].plot(t,keh,'b',linestyle='-',linewidth=1.0, label='$KE_{H}$')
ax[1][0].set_xlabel('$\omega_{pe}t$')
ax[1][0].set_ylabel('$KE_{H}$')
ax[1][0].legend(loc='upper right',framealpha=0.5)

ax[1][1].plot(t,keb,'k',linestyle='-',linewidth=1.0, label='$KE_{B}$')
ax[1][1].set_xlabel('$\omega_{pe}t$')
ax[1][1].set_ylabel('$KE_{B}$')
ax[1][1].legend(loc='upper right',framealpha=0.5)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.savefig(pjoin(path,'ke_20.png'),dpi=dpi)

#plt.figure(2)
#plt.semilogx(t,ke,'b',linestyle='-',linewidth=1.0)

plt.show()