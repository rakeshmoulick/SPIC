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

ke = kei+kec+keh+keb # Total kinetic energy of the system

# Plot the figure 
plt.figure(1)
plt.plot(t,ke,'r',linestyle='-',linewidth=1.0)

plt.xlabel('$\omega_{pe}t$')
plt.ylabel('System Kinetic Energy')

plt.savefig(pjoin(path,'ke.png'),dpi=600)

plt.figure(2)
ke_log = np.log(abs(ke))
plt.plot(t, ke_log)
plt.xlabel('$\omega_{pe}t$')
plt.ylabel('Logarithmic System Kinetic Energy')

plt.savefig(pjoin(path,'ke_log.png'),dpi=600)

plt.show()