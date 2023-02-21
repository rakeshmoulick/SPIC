"""
Plots the dispersion graph of from the electric field data
Use %reset -f to clear the python workspace
Data File Invoked: processed_results_all.npz
Run as: 
"""
# TO RUN: python3 dispersion_npz.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from scipy.constants import value as constants
import os.path
from os.path import join as pjoin
import sys

file_name_pe = 'pe_20.txt'
file_name_ke = 'ke_20.txt'
path = sys.argv[1]
# ------------------ Comments -------------------------------------------------
# input parameters specific file
# path to the data folder is ../data/data002_vd_20/files for vd=20
# path to the data folder is ../data/data001_vd_80/files for vd=80
#------------------------------------------------------------------------------
# Constants
eps0 = constants('electric constant')
kb = constants('Boltzmann constant')
me = constants('electron mass')
AMU = constants('atomic mass constant')
e = constants('elementary charge')
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
NUM_TS = 4600
write_interval = 5
DT_coeff = 0.01
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
n0 = 1E10
Tec = 1*e   # cold electron temperature in joule
Teh = 100*e # hot electron temperature in joule
Teb = 1*e   # beam electron temperature in joule
Ti = 0.026*e
mi = 40*AMU
#-------------------------------------------------------
alp = 1.0
beta = 0.04
#------------------------------------------------------
# SIM Vars
NC = 1024
Time = 0
dx = 1
#-----------------------------------------------------
ni0 = n0
nec0 = n0/(1+alp+beta)
neh0 = alp*nec0
neb0 = beta*nec0
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
DATA_TS = int(NUM_TS/write_interval) + 1

LDH = np.sqrt(eps0*Teh/(neh0*e**2)) # Hot electron Debye Length
LDC = np.sqrt(eps0*Tec/(nec0*e**2)) # Cold electron Debye Length
LDB = np.sqrt(eps0*Teb/(neb0*e**2)) # Beam electron Debye Length
LD = np.sqrt(eps0*Tec/(n0*e**2)) # Characteristic Debye length

wpi = np.sqrt(ni0*e**2/(eps0*mi))
wp  = np.sqrt(n0*e**2/(eps0*me)) # Total Plasma Frequency

wpec = np.sqrt((nec0*e**2)/(eps0*me))
wpeh = np.sqrt(neh0*e**2/(eps0*me))
wpeb = np.sqrt(neb0*e**2/(eps0*me))

#ud = vd*np.sqrt(Tec/me);
nParticlesE = 20000
nParticlesI = nParticlesE 
xl = NC*dx

electron_cold_spwt = (nec0*xl*LD)/(nParticlesE)
electron_hot_spwt = (neh0*xl*LD)/(nParticlesE)
electron_beam_spwt = (neb0*xl*LD)/(nParticlesE)
ion_spwt = (ni0*xl*LD)/(nParticlesI)

DT = DT_coeff*(1.0/wp)
#++++++++++++++++++++ UNPACK ++++++++++++++++++++++++++++++++++++++++++++++++++
if os.path.exists(pjoin(path,file_name_pe)):
    ts,pe = np.loadtxt(pjoin(path,file_name_pe),unpack=True)    
else:
    print('No data')
    exit()   
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# un-normalize the pe
pe *= (Tec/e)     
#Total_cold_electrons = nParticlesE*(electron_cold_spwt) # nec0*xl*LD
Total_cold_electrons = nec0*xl*LD

pe /= (Total_cold_electrons)*Tec
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -------------------------- Comments ------------------------------------------
# input parameters specific file
# path to the data folder is ../data/data002_vd_20/files/ke_1024.txt for vd=20
# path to the data folder is ../data/data001_vd_80/files/ke_1024.txt for vd=80
# vd is an input parameter to run this file and that must be provided along with others.
#++++++++++++++++++++ UNPACK ++++++++++++++++++++++++++++++++++++++++++++++++++
if os.path.exists(pjoin(path,file_name_ke)):
    t,kei,kec,keh,keb = np.loadtxt(pjoin(path,file_name_ke),unpack=True)
else:
    print('No data')
    exit()

ke = kec+keh+keb
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
figsize = np.array([80,80/1.618]) #Figure size in mm (FOR SINGLE FIGURE)
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

#fig,ax = plt.subplots(3,1, figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
fig,ax = plt.subplots(3,1)
ax[0].plot(ts,pe)
ax[0].set_xlabel('$\omega_{pe}t$')
ax[0].set_ylabel('$Potential Energy$')

ax[1].plot(t,ke)
ax[1].set_xlabel('$\omega_{pe}t$')
ax[1].set_ylabel('$Kinetic Energy$')

ax[2].plot(ts,ke+pe)
ax[2].set_xlabel('$\omega_{pe}t$')
ax[2].set_ylabel('$Total Energy$')

plt.savefig(pjoin(path,'pe.png'),dpi=dpi)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.show()
