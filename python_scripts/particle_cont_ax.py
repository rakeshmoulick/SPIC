"""
This file plots the continuous evolution of particle data.
Plots have been initiated using 'axes'.
Only citing the path to the data file is enough to plot,
No specific input corresponding to parameteric variation is required
Run as: 
"""
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os.path import join as pjoin

path = './output_1/'
# ----------------- Comments ---------------------------------------------------
# Path to data file for vd=20 is
# ../data/particle_data/part_vd_20/data
# Path to data file for vd=80 is
# ../data/particle_data/part_vd_80/data
# ------------------------------------------------------------------------------
NUM_TS = 100
write_interval = 5
rang = NUM_TS/write_interval

k = 0

fig, ax = plt.subplots(3,1,figsize=(10,8))
plt.cla()

for i in range(10):

    file_name_ec = 'ec%d.txt'%(k)
    file_name_eh = 'eh%d.txt'%(k)
    file_name_eb = 'eb%d.txt'%(k)

    if os.path.exists(pjoin(path,file_name_ec)):
        xec,vec = np.loadtxt(pjoin(path,file_name_ec),unpack=True)
    else:
        print('No data')
        exit()

    if os.path.exists(pjoin(path,file_name_eh)):
        xeh,veh = np.loadtxt(pjoin(path,file_name_eh),unpack=True)
    else:
        print('No data')
        exit()
    if os.path.exists(pjoin(path,file_name_eb)):
        xeb,veb = np.loadtxt(pjoin(path,file_name_eb),unpack=True)
    else:
        print('No data')
        exit()

    # Plot the figure
    # In scatter plot 's' specifies the area

    ax[0].scatter(xec,vec,color='r',marker='.',s=0.1,alpha=0.5)
    ax[0].set_xlabel('$x_{ec}$')
    ax[0].set_ylabel('$\u03C5_{ec}$')

    ax[1].scatter(xeh,veh,color='g',marker='.',s=0.1,alpha=0.5)
    ax[1].set_xlabel('$x_{eh}$')
    ax[1].set_ylabel('$\u03C5_{eh}$')

    ax[2].scatter(xeb,veb,color='b',marker='.',s=0.1,alpha=0.5)
    ax[2].set_xlabel('$x_{eb}$')
    ax[2].set_ylabel('$\u03C5_{eb}$')

    ax[0].set_title(i)
    print('k = ',k)
    k = k + write_interval
    plt.pause(0.01)
plt.show()
