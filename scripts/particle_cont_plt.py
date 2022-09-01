# This file is for the continuous time evolution of particle data.
# Only citing the path to the data file is enough to plot,
# No specific input corresponding to parameteric variation is required
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os.path import join as pjoin

path = './output_1/'
# ------------------- Comments -------------------------------------------------
# Path to data file for vd=20 is
# ../data/particle_data/part_vd_20/data
# Path to data file for vd=80 is
# ../data/particle_data/part_vd_80/data
# ------------------------------------------------------------------------------
NUM_TS = 100
write_interval = 5
rang = NUM_TS/write_interval

k = 0
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
    plt.cla()
    fig, ax = plt.figure(figsize=(10,8))
    # In scatter plot 's' specifies the area
    plt.subplot(311)
    plt.scatter(xec,vec,color='r',marker='.',s=0.1,alpha=0.5)
    plt.xlabel('$x_{ec}$')
    plt.ylabel('$\u03C5_{ec}$')

    plt.subplot(312)
    plt.scatter(xeh,veh,color='g',marker='.',s=0.1,alpha=0.5)
    plt.xlabel('$x_{eh}$')
    plt.ylabel('$\u03C5_{eh}$')

    plt.subplot(313)
    plt.scatter(xeb,veb,color='b',marker='.',s=0.1,alpha=0.5)
    plt.xlabel('$x_{eb}$')
    plt.ylabel('$\u03C5_{eb}$')

    plt.title(i)
    print('k = ',k)
    k = k + write_interval
    plt.pause(0.01)
plt.show()
