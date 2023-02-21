"""
Plots the kinetic energy of the total system along with cold, hot and beam electrons.
Appropriate files must be chosen using 'file_name' and 'path'.
Use %reset -f to clear the python workspace.
Data File Invoked: ke_1024.txt
Run as:
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import os.path
from os.path import join as pjoin
from scipy.constants import value as constants
from scipy.signal import find_peaks
import sys
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
eps0 = constants('electric constant')
e = constants('elementary charge')
me = constants('electron mass')
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
vd = [15, 20, 25, 35, 40, 45, 60, 80]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
figsize = np.array([200,200/1.618]) #Figure size in mm (FOR SINGLE FIGURE)
dpi = 1200                        #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
fig,ax = plt.subplots(figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
vd_holder = []
diff_holder = []
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for i in range(len(vd)):
    file_name = 'ke_%d.txt'%(vd[i])
    path = '../data_vd_%d/files/'%(vd[i])
    #++++++++++++++++++++ UNPACK ++++++++++++++++++++++++++++++++++++++++++++++++++
    if os.path.exists(pjoin(path,file_name)):
        t,kei,kec,keh,keb = np.loadtxt(pjoin(path,file_name),unpack=True)
    else:
        print('No data')
        exit()
       
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #keb /= np.max(keb)
    peaks, _ = find_peaks(keb, prominence=0.1)    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    dkeb = keb[0] - keb[peaks[0]]
    dkeb /= np.max(keb)
    diff = (1 - dkeb)    
    diff_percent = diff*100
    """
    range = np.max(keb) - np.min(keb)
    fp_range = keb[peaks[0]] - np.min(keb)
    ratio = fp_range/range
    diff_percent = ratio*100 
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    vd_holder.append(vd[i])
    diff_holder.append(diff_percent)

ax.plot(vd_holder,diff_holder,'--',marker='o',markersize=10)
for x, y in zip(vd_holder, diff_holder):
    ax.text(x,y, '$v_{d}=$'+str(x), color="black", fontsize=12)
#ax.bar(vd_holder,diff_holder, color ='maroon', width= 10, alpha=0.5)
ax.set_xlabel('$v_{d}$')
ax.set_ylabel('$\Delta$ $KE_{B}$(%)')
ax.grid(True)
plt.savefig('../ke_analysis_plot.png',dpi=dpi)
plt.show()
