"""
Plots the FFT of the hot density graph of from the hot electron density data.
Use %reset -f to clear the python workspace.
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

file_name = 'processed_results_all.npz'
path = './'
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
NUM_TS = 50000
write_interval = 5
DT_coeff = 0.01
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
vd = 20;
n0 = 1E10;
Tec = 1*e;
Teh = 100*e;
Teb = 1*e;
Ti = 0.026*e;
mi = 40*AMU;
#-------------------------------------------------------
alp = 1.0;
beta = 0.04;
#------------------------------------------------------
# SIM Vars
NC = 1024;
Time = 0;
#-----------------------------------------------------
ni0 = (1+alp+beta)*n0;
nec0 = n0;
neh0 = alp*n0;
neb0 = beta*n0;
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
DATA_TS = int(NUM_TS/write_interval) + 1

LDH = np.sqrt(eps0*Teh/(neh0*e**2)) # Hot electron Debye Length
LDC = np.sqrt(eps0*Tec/(nec0*e**2)) # Cold electron Debye Length
LDB = np.sqrt(eps0*Tec/(nec0*e**2)) # Cold electron Debye Length
LD = (LDC*LDC*LDH)/(LDH*LDB + LDC*LDB + LDC*LDH)

wpi = np.sqrt(e**2*ni0/(eps0*mi))
wp = np.sqrt(e**2*ni0/(eps0*me)) # Total Plasma Frequency

wpec = np.sqrt((nec0*e**2)/(eps0*me))
wpeh = np.sqrt(e**2*neh0/(eps0*me))
wpeb = np.sqrt(e**2*neb0/(eps0*me))
#ud = vd*np.sqrt(Tec/me);
ud = vd*(LDC*wpec);

DT = DT_coeff*(1.0/wp)
#++++++++++++++++++++ UNPACK ++++++++++++++++++++++++++++++++++++++++++++++++++
if os.path.exists(pjoin(path,file_name)):
    data = np.load(pjoin(path,file_name))
    x = data['x']
    #ndec = data['ndec']
    ndeh = data['ndeh']

else:
    print('No data')
    exit()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
x = x.reshape(DATA_TS,(NC+1))
x = x[0,:] # SPATIAL GRID: Select only a column, all the columns have the same value
dx = x[1]-x[0] # Normalized StepSize
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
In each row of 'ndeh', there are NC+1 = 1025 columns. Data is put in the first row first,
and then it goes into the second row. Therefore, all the values of the 'ndeh' for the
first time step is written in first row in 1025 columns. The data of next time step
is in the second row and so on. Similar is the case for writing and reshaping x.
w_pe*t = NUM_TS*DT
"""
ndeh = ndeh.reshape(DATA_TS,(NC+1))
mat = np.ones(ndeh.shape)
ndeh = ndeh - mat

print("The shape of ndeh is: ", ndeh.shape)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
wpet_1 = 0 #1000
wpet_2 = 500
y1 = wpet_1/(DT_coeff*write_interval)
y2 = wpet_2/(DT_coeff*write_interval)

ndh = ndeh[int(y1):int(y2),:]
print("The shape of ndh is: ",ndh.shape)
#+++++++++++++++++++++++++ FFT of E-field data ++++++++++++++++++++++++++++++++
Fh = np.fft.fftn(ndh, norm = 'ortho')
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define Omega and K
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
NUM_TS1 = wpet_1/DT_coeff
NUM_TS2 = wpet_2/DT_coeff
actual_sim_time = (NUM_TS2 - NUM_TS1)*(DT) # Previously it was (NUM_TS*DT)
# -----------------------------------------------------------------------------
omega = 2*np.pi*np.arange(NUM_TS)/(actual_sim_time) #(DT*NUM_TS) #(actual_sim_time) # Unnormalized Omega
k     = 2*np.pi*np.arange(NC+1)/(NC*dx*LDC)       # Normalized k
print('Length of k: ',len(k))
print('Max of k: ',np.max(k))

Omega, K = np.meshgrid(omega, k, indexing='ij')
print('Shape of Omega: ',Omega.shape)
#------------------------------------------------------------------------------
halflen = np.array(Fh.shape, dtype=int)//2
Omega = Omega[:halflen[0],:halflen[1]]
K = K[:halflen[0],:halflen[1]]
Fh = Fh[:halflen[0],:halflen[1]]
Omega /=wp # Normalized Omega
K = K*LDC  # Normalized K
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Zh = np.log(np.abs(Fh))

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
fig,ax = plt.subplots(figsize=figsize/25.4,constrained_layout=True,dpi=ppi)

plt.pcolor(K, Omega, Zh, cmap='rainbow',shading='auto')
ax.set_xlabel('$k \lambda_{Dc}$')
ax.set_ylabel('$\omega/\omega_{pe}$')
ax.set_xlim([0, 1.0])
ax.set_ylim([0, 3.0])
plt.savefig(pjoin(path,'dispersion_hot_density_FFT_%d_%d.png'%(wpet_1,wpet_2)),dpi=dpi)
"""
wpet = NUM_TS*DT_coeff # It is 500 in this case
xx = np.linspace(0, NC, (NC+1))
yy = np.linspace(0, wpet, DATA_TS)
plt.pcolor(xx,yy,ndeh, cmap='viridis', shading='auto')
cbar = plt.colorbar()
cbar.set_label('$\Delta n_{h}$')
ax.set_xlabel('$x$')
ax.set_ylabel('$\omega_{pe}t$')
#leg = ax.legend(loc='upper right',framealpha=0.5)
#plt.savefig(pjoin(path,'dispersion_Delta_density_%d_%d.png'%(wpet_1,wpet_2)),dpi=dpi)
"""
plt.show()
