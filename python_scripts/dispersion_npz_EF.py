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

file_name = 'processed_results_all.npz'
#path = './'
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
NUM_TS = 50000
write_interval = 5
DT_coeff = 0.01
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
vd = int(input('Enter vd:'))
n0 = 1E10
Tec = 1*e
Teh = 100*e
Teb = 1*e
Ti = 0.026*e
mi = 32*AMU
#-------------------------------------------------------
alp = 1.0
beta = 0.04
#------------------------------------------------------
# SIM Vars
NC = 1024
Time = 0
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
ud = vd*(LDC*wpec)

DT = DT_coeff*(1.0/wp)
#++++++++++++++++++++ UNPACK ++++++++++++++++++++++++++++++++++++++++++++++++++
if os.path.exists(pjoin(path,file_name)):
    data = np.load(pjoin(path,file_name))
    x = data['x']
    ndec = data['ndec']
    ndeh = data['ndeh']
    ndeb = data['ndeb']
    phi = data['phi']
    EF = data['EF']
else:
    print('No data')
    exit()


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
x = x.reshape(DATA_TS,(NC+1))
x = x[0,:] # SPATIAL GRID: Select only a column, all the columns have the same value
dx = x[1]-x[0] # Normalized StepSize

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
In each row of EF, there are NC+1 = 1025 columns. Data is put in the first row first
and then it goes into the second row. Therefore, all the values of the EF for the
first time step is written in first row in 1025 columns. The data of next time step
is in the second row and so on. Similar is the case for writing and reshaping x.
w_pe*t = NUM_TS*DT
"""
EF = EF.reshape(DATA_TS,(NC+1))

print("The shape of EF is: ", EF.shape)
print("The shape of ndeh is: ", ndeh.shape)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
wpet_1 = 0 #1000
wpet_2 = NUM_TS*DT_coeff
y1 = wpet_1/(DT_coeff*write_interval)
y2 = wpet_2/(DT_coeff*write_interval)
E = EF[int(y1):int(y2),:]

print("The shape of E (reduced EF) is: ", E.shape)
#+++++++++++++++++++++++++ FFT of E-field data ++++++++++++++++++++++++++++++++
F = np.fft.fftn(E, norm='ortho')
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define Omega and K
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
NUM_TS1 = wpet_1/DT_coeff
NUM_TS2 = wpet_2/DT_coeff
actual_sim_time = (NUM_TS2 - NUM_TS1)*(DT) # Previously it was (NUM_TS*DT)
# -----------------------------------------------------------------------------
omega = 2*np.pi*np.arange(NUM_TS)/(actual_sim_time) #(DT*NUM_TS) #(actual_sim_time) # Unnormalized Omega
k     = 2*np.pi*np.arange(NC+1)/(NC*dx*LD)       # Normalized k
print('Length of k: ',len(k))
print('Max of k: ',np.max(k))

Omega, K = np.meshgrid(omega, k, indexing='ij')
print('Shape of Omega: ',Omega.shape)
#------------------------------------------------------------------------------
halflen = np.array(F.shape, dtype=int)//2
Omega = Omega[:halflen[0],:halflen[1]]
K = K[:halflen[0],:halflen[1]]
F = F[:halflen[0],:halflen[1]]
Omega /=wp # Normalized Omega
K = K*LD  # Normalized K
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++ Raw Analytic Part Calculation ++++++++++++++++++++++++++++++
raw_analytic = True
if raw_analytic:
    wea = np.sqrt((wpec**2)*( (1+3*(k*LDC)**2)/(1 + (1/(k*LDH)**2)) ));
    wla = np.sqrt((wpec**2*(1 + 3*k**2*LDC**2)) + (wpeh**2*(1 + 3*k**2*LDH**2)));
    ea = wea/wp    # Electron Acoustic Wave
    ep = wla/wp    # Electron Plasma Wave (Langmuir Wave)
    eb = k*ud/wp   # Electron Beam Wave
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Z = np.log(np.abs(F))
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
#plt.pcolor(K, Omega, Z,cmap='rainbow',shading='auto',vmin=np.min(Z),vmax=np.max(Z))
plt.pcolor(K, Omega, Z,cmap='rainbow',shading='auto',vmin=-3,vmax=2)

if raw_analytic:
    plt.plot(k*LD, ep, color='k', linestyle='-.', lw = 1.0, label='$EPW$')
    plt.plot(k*LD, ea, color='k', linestyle='--', lw = 1.0, label='$EAW$')
    plt.plot(k*LD, eb, color='k', linestyle='-', lw = 1.0, label='$EBW$')

# Note: Changing vmin & vmax will change the depth of the plots.
cbar = plt.colorbar()
cbar.set_label('$\zeta$')

ax.set_xlabel('$k \lambda_{D}$')
ax.set_ylabel('$\omega/\omega_{pe}$')
#ax.set_xlim([0, np.max(K)])
ax.set_xlim([0, 1.0])
ax.set_ylim([0, 3.0])
leg = ax.legend(loc='upper right',framealpha=0.5)
plt.savefig(pjoin(path,'dispersion_%d.png'%(vd)),dpi=dpi)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.show()
