"""
The code plots the kinetic energy of the three electron system 
containing cold, hot and beam electron species. Appropriate files 
must be chosen using file_name and path.
"""
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os.path import join as pjoin

file_name = 'ke_1024.txt'
path = './'

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
kei = kei*(e/me) #kei*((wpec*LDC)**2)*(e/Tec)
kec = kec*(e/me) #kec*((wpec*LDC)**2)*(e/Tec)
keh = keh*(e/me) #keh*((wpec*LDC)**2)*(e/Tec)
keb = keb*(e/me) #keb*((wpec*LDC)**2)*(e/Tec)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ke = kec+keh+keb # Total kinetic energy of the electrons

# Plot the figure 
plt.figure(1, figsize=(10,8))
plt.subplot(221)
plt.plot(t,ke,'r',linestyle='-',linewidth=1.0,label='KE')
plt.legend(loc='upper right',framealpha=0.5)

plt.xlabel('$\omega_{pe}t$')
plt.ylabel('KE')

plt.subplot(222)
plt.plot(t,kec,'g',linestyle='-',linewidth=1.0, label='$KE_{C}$')
plt.legend(loc='upper right',framealpha=0.5)

plt.xlabel('$\omega_{pe}t$')
plt.ylabel('$KE_{C}$')

plt.subplot(223)
plt.plot(t,keh,'b',linestyle='-',linewidth=1.0, label='$KE_{H}$')
plt.legend(loc='upper right',framealpha=0.5)

plt.xlabel('$\omega_{pe}t$')
plt.ylabel('$KE_{H}$')

plt.subplot(224)
plt.plot(t,keb,'k',linestyle='-',linewidth=1.0, label='$KE_{B}$')
plt.legend(loc='upper right',framealpha=0.5)
plt.xlabel('$\omega_{pe}t$')
plt.ylabel('$KE_{B}$')


plt.savefig(pjoin(path,'ke.png'),dpi=600)

plt.figure(2)
plt.semilogx(t,ke,'b',linestyle='-',linewidth=1.0)

plt.show()