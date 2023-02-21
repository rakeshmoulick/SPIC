"""
Plots the IRI data set
Use %reset -f to clear the python workspace.
Data File Invoked: iono_data_Svalbard_2014_Feb1 & iono_data_Svalbard_2020_Feb1
Run as:
"""
file_name1 = 'iono_data_Svalbard_2014_Feb1.txt'
file_name2 = 'iono_data_Svalbard_2020_Feb1.txt'
path = './'
# --------------------- Comments --------------------------------------------
# Path to the data directory is ../iri_dataset/
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import os.path
import matplotlib as mp
from os.path import join as pjoin
import matplotlib.pyplot as plt

if os.path.exists(pjoin(path, file_name1)):
    x1,Ne1,Tn1,Ti1,Te1 = np.loadtxt(pjoin(path,file_name1), unpack=True)
else:
    print('No Data')
    exit()

if os.path.exists(pjoin(path, file_name2)):
    x2,Ne2,Tn2,Ti2,Te2 = np.loadtxt(pjoin(path,file_name2), unpack=True)
else:
    print('No Data')
    exit()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
figsize = np.array([200,200/1.618]) #Figure size in mm (FOR SINGLE FIGURE)
dpi = 1200                        #Print resolution
ppi = np.sqrt(1920**2+1200**2)/20 #Screen resolution

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=figsize/25, constrained_layout=True, dpi=ppi)

ax1.plot(x1,Ne1, label='$N_{e}, 2014$')
ax1.plot(x1,Ne2, label='$N_{e}, 2020$')
ax1.set_xlabel('$x$ (km)')
ax1.set_ylabel('$N_{e}(m^{-3})$')
ax1.legend(loc='upper right',framealpha=0.5)
ax1.grid(True)

ax2.plot(x1,Te1, label='$T_{e}, 2014$')
ax2.plot(x1,Te2, label='$T_{e}, 2020$')
ax2.set_xlabel('$x$ (km)')
ax2.set_ylabel('$T_{e} (K)$')
ax2.legend(loc='upper left',framealpha=0.5)
ax2.grid(True)

"""
fig, ax = plt.subplots(1, 2, figsize=figsize/20, constrained_layout=False, dpi=ppi)
ax[0][0].plot(x1,Ne1, label='$N_{e}, 2014$')
ax[0][0].plot(x1,Ne2, label='$N_{e}, 2020$')
ax[0][0].set_xlabel('$x$ (km)')
ax[0][0].set_ylabel('$N_{e}(m^{-3})$')
ax[0][0].legend(loc='upper right',framealpha=0.5)
ax[0][0].grid(True)

ax[0][1].plot(x1,Te1, label='$T_{e}, 2014$')
ax[0][1].plot(x1,Te2, label='$T_{e}, 2020$')
ax[0][1].set_xlabel('$x$ (km)')
ax[0][1].set_ylabel('$T_{e} (K)$')
ax[0][1].legend(loc='upper left',framealpha=0.5)
ax[0][1].grid(True)

ax[1][0].plot(x1,Ti1, label='$T_{i}, 2014$')
ax[1][0].plot(x1,Ti2, label='$T_{i}, 2020$')
ax[1][0].set_xlabel('$X$ (km)')
ax[1][0].set_ylabel('$T_{i} (K)$')
ax[1][0].legend(loc='upper left',framealpha=0.5)
ax[1][0].grid(True)

ax[1][1].plot(x1,Tn1, label='$T_{n}, 2014$')
ax[1][1].plot(x1,Tn2, label='$T_{n}, 2020$')
ax[1][1].set_xlabel('$x$ (km)')
ax[1][1].set_ylabel('$T_{n} (K)$')
ax[1][1].legend(loc=0,framealpha=0.5)
ax[1][1].grid(True)
"""
plt.savefig(pjoin(path,"iono.png"), dpi=dpi)
plt.show()
