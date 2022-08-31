"""
This file plots the dispersion graph from the npz file processed_results_E.npz
Use %reset -f to clear the python workspace
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
vd = 80;
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
    ndec = data['ndec']
    ndeh = data['ndeh']
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
wpet_2 = 500
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
k     = 2*np.pi*np.arange(NC+1)/(NC*dx*LDC)       # Normalized k
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
K = K*LDC  # Normalized K
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ka = np.linspace(0, np.max(K), NC+1)
#kappa = k*LDC
#ldr = LDH/LDC
#+++++++++++++++++++++ Solved Analytic Part Calculation ++++++++++++++++++++++++++++++
"""
def EAW_dispersion():
     # Coefficients
     coeff1 = ( 1 + ( 1 / (kappa*kappa*ldr*ldr) ) )
     coeff2 = 0
     coeff3 = - (1 + (3*kappa*kappa) )*(wpec*wpec)
     roots = []
     for i in range(1,len(kappa)):
         coeffs = [coeff1[i], coeff2, coeff3[i]]
         root = np.roots(coeffs)
         roots.append(root)
     roots = np.array(roots)
     print("Shape of root is: ", np.shape(roots))
     return roots/wp
def EPW_dispersion():
     # Coefficients
     coeff1 = 1
     coeff2 = 0
     coeff3 = - ( ((1 + (3*kappa*kappa))*(wpec*wpec)) + ((1 + (3*kappa*kappa*ldr*ldr))*(wpeh*wpeh)))
     roots = []
     for i in range(1,len(kappa)):
         coeffs = [coeff1, coeff2, coeff3[i]]
         root = np.roots(coeffs)
         roots.append(root)
     roots = np.array(roots)
     return roots/wp

roots_EAW = EAW_dispersion()
roots_EPW = EPW_dispersion()

solved_analytic = True
if solved_analytic:
    eaw = np.real(roots_EAW[:,0])
    epw = np.real(roots_EPW[:,0])
    ebw = k*ud/wp
"""
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
plt.pcolor(K, Omega, Z,cmap='rainbow',shading='auto',vmin=-3.0,vmax=2.0)
# Note: Changing vmin & vmax will change the depth of the plots. 
cbar = plt.colorbar()
cbar.set_label('$\zeta$')

"""
if solved_analytic:
    plt.plot(kappa[1:], eaw, color='b', linestyle='-.', lw = 1.0, label='$EAW$')
    plt.plot(kappa[1:], epw, color='b', linestyle='-', lw = 1.0, label='$EPW$')
    plt.plot(kappa[1:], ebw[1:], color='b', linestyle='--', lw = 1.0, label='$EBW$')
"""

if raw_analytic:
    plt.plot(k*LDC, ep, color='k', linestyle='-.', lw = 1.0, label='$EPW$')
    plt.plot(k*LDC, ea, color='k', linestyle='--', lw = 1.0, label='$EAW$')
    plt.plot(k*LDC, eb, color='k', linestyle='-', lw = 1.0, label='$EBW$')

ax.set_xlabel('$k \lambda_{Dc}$')
ax.set_ylabel('$\omega/\omega_{pe}$')
#ax.set_xlim([0, np.max(K)])
ax.set_xlim([0, 1.0])
ax.set_ylim([0, 3.0])
leg = ax.legend(loc='upper right',framealpha=0.5)
plt.savefig(pjoin(path,'dispersion_npz_EF_FFT_%d_%d.png'%(wpet_1,wpet_2)),dpi=dpi)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
plt.figure(2) # Plots the electric field at a definite w_pe*t
# Define the time at which electric field data is required
wpet = 49
# Get the index value corresponding to that time
s = wpet/(DT_coeff*write_interval)

plt.plot(x, E[int(s),:])
plt.xlabel('x')
plt.ylabel('Electric Field')
plt.savefig(pjoin(path,'EF_%d.png'%(wpet)),dpi=dpi)
"""
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
plt.figure(3)
plt.pcolor(ndh, cmap='rainbow',shading='auto')
plt.xlim([0, 1.0])
plt.ylim([0, 3.0])
"""
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.show()
