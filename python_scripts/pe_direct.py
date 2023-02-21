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

file_name_pe = 'pe_80.txt'
file_name_ke = 'ke_80.txt'
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

#++++++++++++++++++++ UNPACK ++++++++++++++++++++++++++++++++++++++++++++++++++
if os.path.exists(pjoin(path,file_name_pe)):
    ts,pe = np.loadtxt(pjoin(path,file_name_pe),unpack=True)    
else:
    print('No data')
    exit()   
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
