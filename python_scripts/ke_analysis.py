"""
Plots the kinetic energy of the total system along with cold, hot and beam electrons.
Appropriate files must be chosen using 'file_name' and 'path'.
Use %reset -f to clear the python workspace.
Data File Invoked: ke_1024.txt
Run as:
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import os.path
from os.path import join as pjoin
from scipy.constants import value as constants
from scipy.signal import find_peaks
import sys
vd = int(input('Enter vd:'))
file_name = 'ke_%d.txt'%(vd)
path = sys.argv[1]
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
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#keb /= np.max(keb)
peaks, _ = find_peaks(keb, prominence=0.1)
#peaks, _ = find_peaks(keb, distance=100)
print(peaks)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dkeb = keb[0] - keb[peaks[0]]
print("Normalized keb diff:",dkeb/np.max(keb))
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
fig,ax = plt.subplots(figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
ax.axvline(peaks[0], color='red',linestyle='--',linewidth=2)
ax.axhline(keb[peaks[0]], color='red',linestyle='--',linewidth=2)
ax.plot(peaks,keb[peaks],'ob')
ax.plot(keb,'k',linestyle='-',linewidth=1.0, label='$KE_{B}$')
ax.set_xlabel('No. of samples')
#ax.set_xlabel('$\omega_{pe}t$')
ax.set_ylabel('$KE_{B}$')
ax.legend(loc='upper right',framealpha=0.5)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.savefig(pjoin(path,'ke_analysis_%d.png'%(vd)),dpi=dpi)

plt.show()
