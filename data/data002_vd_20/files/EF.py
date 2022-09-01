import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import value as constants
import os.path
from os.path import join as pjoin

file_name = 'processed_results_E.npz'
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
#In each row of EF, there are NC+1 = 1025 columns. Data is put in the first row first
#nd then it goes into the second row. Therefore, all the values of the EF for the
#irst time step is written in first row in 1025 columns. The data of next time step
#is in the second row and so on. Similar is the case for writing and reshaping x.
#w_pe*t = NUM_TS*DT
"""


EF = EF.reshape(DATA_TS,(NC+1))
print("The shape of EF is: ", EF.shape)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
wpet_1 = 0 
wpet_2 = 500
y1 = wpet_1/(DT_coeff*write_interval)
y2 = wpet_2/(DT_coeff*write_interval)
E = EF[int(y1):int(y2),:]
print("The shape of E (reduced EF) is: ", E.shape)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define the time at which electric field data is required
wpet = [0, 100, 200, 300, 400]
# Get the index value corresponding to that time 
coeff = 1.0/(DT_coeff*write_interval)
s = [element*coeff for element in wpet]

EP0 = E[int(s[0]),:]
EP1 = E[int(s[1]),:]
EP2 = E[int(s[2]),:]
EP3 = E[int(s[3]),:]
EP4 = E[int(s[4]),:]

z = [element*np.ones(x.shape) for element in wpet]

plt.figure(figsize=(10,10))
ax = plt.subplot(projection='3d')
ax.plot(z[0], x, EP0, label="$\omega_{pe}t$=%d"%(wpet[0]))
ax.plot(z[1], x, EP1, label="$\omega_{pe}t$=%d"%(wpet[1]))
ax.plot(z[2], x, EP2, label="$\omega_{pe}t$=%d"%(wpet[2]))
ax.plot(z[3], x, EP3, label="$\omega_{pe}t$=%d"%(wpet[3]))
ax.plot(z[4], x, EP4, label="$\omega_{pe}t$=%d"%(wpet[4]))

ax.set_xlabel('$\omega_{pe}t$')
ax.set_ylabel('x')
ax.set_zlabel('Electric Field')
leg = ax.legend(loc=0,framealpha=0.5)
plt.savefig(pjoin(path,'EF.png'),dpi=600)

plt.show()
