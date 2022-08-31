"""
The aim of this script is to select a range of data values of the electric 
field corresponding to a range of normalized time (w_pe*t). Thus, we select 
only a certain range of time over which the field values are to be selected. 
""" 
# TO RUN: dispersion_updated.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from scipy.constants import value as constants
import os.path
from os.path import join as pjoin

file_name = 'results_1024.txt'
path = './'
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Constants
eps0 = constants('electric constant')
kb = constants('Boltzmann constant')
me = constants('electron mass')
AMU = constants('atomic mass constant')
e = constants('elementary charge')
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
NUM_TS = 120000 # Total time step
write_interval = 5 # Time step interval for data writing  
DT_coeff = 0.01 # Coefficient of DT (signifies normalized time and 1% wp inverse)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
vd = 30;      # Coefficient of beam speed 
n0 = 1E10;    # Cold electron density
Tec = 1*e;    # Cold electron Temperature
Teh = 100*e;  # Hot electron Temperature
Teb = 1*e;    # Beam electron Temperature
Ti = 0.026*e; # Ion Temperature
mi = 40*AMU;  # Mass of the ion
#-------------------------------------------------------
alp = 1.0;    # Ratio of the hot electrons to cold electrons
beta = 0.1;   # Ratio of the beam electrons to cold electrons
#------------------------------------------------------
# SIM Vars
NC = 1024;    # Number of cells in the PIC simulation
Time = 0;     # Starting point of time
#-----------------------------------------------------
ni0 = (1+alp+beta)*n0; # Equilibrium Ion density
nec0 = n0;             # Equilibrium Cold electron density
neh0 = alp*n0;         # Equilibrium Hot electron density
neb0 = beta*n0;        # Equilibrium Beam density
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
DATA_TS = int(NUM_TS/write_interval) + 1 # Temporal length of data 

LDH = np.sqrt(eps0*Teh/(neh0*e**2)) # Hot electron Debye Length
LDC = np.sqrt(eps0*Tec/(nec0*e**2)) # Cold electron Debye Length
LDB = np.sqrt(eps0*Teb/(neb0*e**2)) # Beam electron Debye Length
LD = (LDC*LDH*LDB)/(LDH*LDB + LDC*LDB + LDH*LDC) # Total electron Debye Length

wpi = np.sqrt(e**2*ni0/(eps0*mi))# Ion plasma frequency
wp = np.sqrt(e**2*ni0/(eps0*me)) # Total Plasma Frequency
wpec = np.sqrt((nec0*e**2)/(eps0*me)) # Cold electron plasma frequency
wpeh = np.sqrt(e**2*neh0/(eps0*me)) # Hot electron plasma frequency
wpeb = np.sqrt(e**2*neb0/(eps0*me)) # Beam electron plasma frequency

#ud = vd*np.sqrt(Tec/me); # Beam drift speed expression 
ud = vd*(LDC*wpec); # Beam drift speed expression 

DT = DT_coeff*(1.0/wp) # Un-Normalized time step
#++++++++++++++++++++ UNPACK & Read Data ++++++++++++++++++++++++++++++++++++++
if os.path.exists(pjoin(path,file_name)):        
    x, ndec, ndeh, phi, EF = np.loadtxt(pjoin(path,file_name),unpack=True)
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
and then it goes into the second row, while reshaping. Therefore, all the values 
of the EF for the first time-step is written in first row (consecutively in 1025 columns). 
The data of next time step goes to the second row and so on. 
Similar is the case for writing while reshaping x. 
"""
EF = EF.reshape(DATA_TS,(NC+1))
print("The shape of EF is: ", EF.shape)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Select a range of w_pe*t and sort electric field accordingly
"""
Note:
    Total Normalized Time (w_pe*t) = DT_coeff * NUM_TS
    Total Normalized Time (w_pe*t) = DT_coeff * write_interval * (NUM_TS/write_interval)
    Total Normalized Time (w_pe*t) = DT_coeff * write_interval * y(= NUM_TS/write_interval say)
    y = w_pe*t/(DT_coeff*write_interval)
    Here, y gives the exact row number of the EF data corresponding to a value of w_pe*t
""" 
wpet_1 = 0  # first wpe*t 
wpet_2 = 1200 # second wpe*t

y1 = wpet_1/(DT_coeff*write_interval) # Select row no. of EF data corresponding to wpet_1
y2 = wpet_2/(DT_coeff*write_interval) # Select row no. of EF data corresponding to wpet_2

# Get a new Electric field data as E with data of EF between wpet_1 and wpet_2
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
omega = 2*np.pi*np.arange(NUM_TS)/(actual_sim_time) # Un-Normalized Omega (Frequency)
k     = 2*np.pi*np.arange(NC+1)/(NC*dx*LDC)         # Un-Normalized k 
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
# ka = np.linspace(0, np.max(K), NC+1) # defining alternative of k: better not to use
kappa = k*LDC
ldr = LDH/LDC
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
    eaw = np.real(roots_EAW[:,0]) # Electron Acoustic Wave
    epw = np.real(roots_EPW[:,0]) # Electron Plasma Wave(Langmuir Wave)  
    ebw = k*ud/wp                 # Electron Beam Wave
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
cbar = plt.colorbar()
cbar.set_label('$\zeta$')
"""
if solved_analytic:
    plt.plot(k[1:]*LDC, epw, color='b', linestyle='-', lw = 1.0, label='$EPW$')
    plt.plot(k[1:]*LDC, eaw, color='b', linestyle='-.', lw = 1.0, label='$EAW$')    
    plt.plot(k[1:]*LDC, ebw[1:], color='b', linestyle='--', lw = 1.0, label='$EBW$')
"""    
if raw_analytic:
    plt.plot(k*LDC, ep, color='k', linestyle='--', lw = 1.0, label='$EPW$')
    plt.plot(k*LDC, ea, color='k', linestyle='--', lw = 1.0, label='$EAW$')
    plt.plot(k*LDC, eb, color='k', linestyle='--', lw = 1.0, label='$EBW$')
    
ax.set_xlabel('$k \lambda_{Dc}$')    
ax.set_ylabel('$\omega/\omega_{pe}$')

#ax.set_xlim([0, np.max(K)])
ax.set_xlim([0, 1.0])
ax.set_ylim([0, 3.0])

leg = ax.legend(loc='upper right',framealpha=0.5)

plt.savefig(pjoin(path,'dispersion.png'),dpi=dpi)
plt.show()



























