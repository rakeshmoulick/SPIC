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
import sys
file_name = 'pe_20.txt'
#path = './'
path = sys.argv[1]
# -------------------------- Comments ------------------------------------------
# input parameters specific file
# path to the data folder is ../data/data002_vd_20/files/ke_1024.txt for vd=20
# path to the data folder is ../data/data001_vd_80/files/ke_1024.txt for vd=80
# vd is an input parameter to run this file and that must be provided along with others.
#++++++++++++++++++++ UNPACK ++++++++++++++++++++++++++++++++++++++++++++++++++
if os.path.exists(pjoin(path,file_name)):
    t,pe = np.loadtxt(pjoin(path,file_name),unpack=True)
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

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
figsize = np.array([200,200/1.618]) #Figure size in mm (FOR SINGLE FIGURE)
dpi = 1200                          #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24   #Screen resolution

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Y = np.fft.fft(pe)
N = len(Y)
Y_mag = (np.abs(Y))/N
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Constants
eps0 = constants('electric constant')
kb = constants('Boltzmann constant')
me = constants('electron mass')
AMU = constants('atomic mass constant')
e = constants('elementary charge')
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
NUM_TS = 50000
write_interval = 5
DT_coeff = 0.01
#-----------------------------------------------------
n0 = 1E10
alp = 1.0
beta = 0.04
# ---------------------------------------------------
ni0 = n0
nec0 = n0/(1+alp+beta)
neh0 = alp*nec0
neb0 = beta*nec0
wp  = np.sqrt(n0*e**2/(eps0*me)) # Total Plasma Frequency
# -----------------------------------------------------
DT = DT_coeff*(1.0/wp)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++
actual_sim_time = (NUM_TS)*(DT)
omega = 2*np.pi*np.arange(N)/(actual_sim_time)
# Note: Putting N or NUM_TS does not change the magnitude. Also the graph remains the same irrespective of this choice  
omega /=wp
#fs = 10000
#freq = (fs/N)*np.arange(N) 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
halflen = int(N/2+1)
freq_plot = omega[0:halflen]
#freq_plot = freq[0:halflen]
Y_mag_plot = 2*Y_mag[0:halflen]
Y_mag_plot[0] = Y_mag_plot[0]/2 
print('Size of fft(Y):',len(Y_mag))
print('Size of omega:',len(omega))
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot the figure
fig,ax = plt.subplots(figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
ax.plot(freq_plot[0:], Y_mag_plot[0:])
#ax.stem(freq_plot[1:], Y_mag_plot[1:])
ax.set_xlim([-0.5,5.0])
ax.set_xlabel('$\omega$/$\omega_{p}$')
ax.set_ylabel('Magnitude')
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.savefig(pjoin(path,'pe_fft_%d.png'%(vd)),dpi=dpi)

plt.show()
