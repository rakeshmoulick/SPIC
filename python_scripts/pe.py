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
import scipy.integrate as intg
import sys

vd = int(input('Enter vd:'))
file_name = 'processed_results_all.npz'
file_name_ke = 'ke_%d.txt'%(vd)
#path = '../data_vd_%d/files/'%(vd)

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
n0 = 1E10
Tec = 1*e   # cold electron temperature in joule
Teh = 100*e # hot electron temperature in joule
Teb = 1*e   # beam electron temperature in joule
Ti = 0.026*e
mi = 32*AMU
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
ud = vd*(LDC*wpec)
nParticlesE = 5000000
nParticlesI = nParticlesE 
xl = NC*dx

electron_cold_spwt = (nec0*xl*LD)/(nParticlesE)
electron_hot_spwt = (neh0*xl*LD)/(nParticlesE)
electron_beam_spwt = (neb0*xl*LD)/(nParticlesE)
ion_spwt = (ni0*xl*LD)/(nParticlesI)

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
w_pe*t = NUM_TS*DT_coeff
"""
EF = EF.reshape(DATA_TS,(NC+1))
phi = phi.reshape(DATA_TS,(NC+1))
# create array of the w_pe*t
ts = np.arange(0, NUM_TS+write_interval, write_interval)*DT_coeff

#print("The shape of EF is: ", EF.shape)

#+++++++++++++++++++++++++++++++ Method-1 : Electric Field +++++++++++++++++++++++++++++++
# each row of EF correspond to a separate time step. Hence, no of rows implies no of timesteps
pe = np.zeros(EF.shape[0])  
for i in range(len(pe)):        
    E = np.average(np.abs(EF[i,:]))
    # un-normalize the electric field
    E = (me*wp**2*LD/e)*E
    pe[i] = 0.5*eps0*(E**2)

# Normalize by the thermal energy 
# pe *= xl*LD # multiply by the volume of the system to get energy from energy density
# pe /= ((electron_cold_spwt)*nParticlesE)*Tec
pe /= nec0*Tec
# ++++++++++++++++++++++++++++++  Method-2 : Electric Field ++++++++++++++++++++++++++++++++
# each row of EF correspond to a separate time step. Hence, no of rows implies no of timesteps
PE1 = np.zeros(EF.shape[0])
for i in range(len(PE1)):
    xdata = x*LD # un-normalized x-data   
    ydata = (EF[i,:]**2) * ((me*wp**2*LD/e)**2) # un-normalized EF-square data
    # integrate the square electric field over the volume to get the un-normalized electric potential energy
    E_int = intg.trapz(ydata,xdata)        
    # multiply 0.5*eps0 to the un-normalized electric potential energy
    PE1[i] = 0.5*eps0*(E_int)

# Normalize by the thermal energy 
Tot_cold_energy = ((electron_cold_spwt)*nParticlesE)*Tec
# Normalize the electric potential energy by the total cold electron energy 
PE1 /= Tot_cold_energy

# ++++++++++++++++++++++++++++++  Method-3 : Electric Potential +++++++++++++++++++++++++++++++++++
# Calculate the potential energy from the electric potential

Total_electrons = nParticlesE*(electron_cold_spwt + electron_hot_spwt + electron_beam_spwt)
Total_cold_electrons = nParticlesE*(electron_cold_spwt)
Total_Charge = (Total_electrons)*e

PE = np.zeros(phi.shape[0])
for i in range(len(PE)):
    PHI = np.average(np.abs(phi[i,:]))
    # un-normalize the electric potential
    PHI = (Tec/e)*PHI
    PE[i] = PHI    

# calculate the un-normalized total potential energy
PE *= Total_Charge    
# calculate the normalized potential energy
PE /= (Total_cold_electrons)*Tec    
# ++++++++++++++++++++++++++++ Method-4 : Electric Potential ++++++++++++++++++++++++++++++++++++++
PEE = np.zeros(phi.shape[0])
for i in range(len(PE)):
    PHI = np.average(np.abs(phi[i,:]))
    # un-normalize the electric potential
    PHI = (Tec/e)*PHI
    PEE[i] = 0.5*eps0*(PHI**2)/(xl*LD)    

# calculate the un-normalized total potential energy    
# calculate the normalized potential energy
PEE /= (Total_cold_electrons)*Tec  

# +++++++++++++++  Assign Appropriate Potential Energy +++++++++++++++++++++++
Pot_En = PE1 # use Method-2 to calculate Pot_En
#+++++++++++++++++++++ Kinetic Energy ++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++ UNPACK ++++++++++++++++++++++++++++++++++++++++++++++++++
if os.path.exists(pjoin(path,file_name_ke)):
    t,kei,kec,keh,keb = np.loadtxt(pjoin(path,file_name_ke),unpack=True)
else:
    print('No data')
    exit()

ke = kec+keh+keb
#ke /= np.max(ke)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
figsize = np.array([80,80/1.618]) #Figure size in mm (FOR SINGLE FIGURE)
dpi = 1200                        #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24.5 #Screen resolution

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
ax[0].plot(ts,Pot_En)
ax[0].set_xlabel('$\omega_{pe}t$')
ax[0].set_ylabel('$PE$')
ax[0].grid(True)

ax[1].plot(t,ke)
ax[1].set_xlabel('$\omega_{pe}t$')
ax[1].set_ylabel('$KE$')
ax[1].grid(True)

ax[2].plot(ts,ke+Pot_En)
ax[2].set_ylim([min(ke+Pot_En)-1.0, max(ke+Pot_En)+1.0])
ax[2].set_xlabel('$\omega_{pe}t$')
ax[2].set_ylabel('$TE$')
ax[2].grid(True)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.savefig(pjoin(path,'pe_vd_%d.png'%(vd)),dpi=dpi)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.show()
