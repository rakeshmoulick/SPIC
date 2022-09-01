# This file plots the particle data at fixed times
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os.path import join as pjoin

path = './data/'
path_fig = './'
# Define the time at which data is required
DT_coeff = 0.01
write_interval = 200

# File index value is k(say), change this index to get the plot at required time.
# Calculate the wpet for a k-value by using DT_coeff beforehand.
k = 150000

wpet = k*DT_coeff # Value of wpet

file_name_ec = 'ec%d.txt'%(int(k))
file_name_eh = 'eh%d.txt'%(int(k))
file_name_eb = 'eb%d.txt'%(int(k))

# Load Cold electron data
if os.path.exists(pjoin(path,file_name_ec)):
    xec,vec = np.loadtxt(pjoin(path,file_name_ec),unpack=True)
else:
    print('No data')
    exit()

# Load Hot electron data
if os.path.exists(pjoin(path,file_name_eh)):
    xeh,veh = np.loadtxt(pjoin(path,file_name_eh),unpack=True)
else:
    print('No data')
    exit()

# Load Beam electron data
if os.path.exists(pjoin(path,file_name_eb)):
    xeb,veb = np.loadtxt(pjoin(path,file_name_eb),unpack=True)
else:
    print('No data')
    exit()

# Plot the figure

# In scatter plot s specifies the area
plt.figure(1)
plt.scatter(xec,vec,color='b',marker='.',s=0.1,alpha=0.5)
plt.xlabel('$x_{ec}$')
plt.ylabel('$\u03C5_{ec}$')
plt.title("Cold Electron Phase Space at $\omega_{pe}t = %d$" %(wpet))
plt.savefig(pjoin(path_fig,'ec_%d'%(wpet)),dpi=600)

plt.figure(2)
plt.scatter(xeh,veh,color='r',marker='.',s=0.1,alpha=0.5)
plt.xlabel('$x_{eh}$')
plt.ylabel('$\u03C5_{eh}$')
plt.title("Hot Electron Phase Space at $\omega_{pe}t = %d$" %(wpet))
plt.savefig(pjoin(path_fig,'eh_%d'%(wpet)),dpi=600)

plt.figure(3)
plt.scatter(xeb,veb,color='g',marker='.',s=0.1,alpha=0.5)
plt.xlabel('$x_{eb}$')
plt.ylabel('$\u03C5_{eb}$')
plt.title("Beam Electron Phase Space at $\omega_{pe}t = %d$" %(wpet))
plt.savefig(pjoin(path_fig,'eb_%d'%(wpet)),dpi=600)
plt.show()
