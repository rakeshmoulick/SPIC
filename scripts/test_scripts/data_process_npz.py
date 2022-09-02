#!/usr/bin/env python
"""
This file is to be called for shortening the memory size of the data file results_1024.txt.
It can compress the file size from around 1 GB to around 150 MB.
Run as: python3 data_process_npz.py '../../data/data_test/' for processed_results_E
        python3 data_process_npz.py '../../data/data_test/' 'True' for processed_results_all
"""
import numpy as np
from os.path import join as pjoin
import os.path
import sys
from glob import glob
# -----------------------------------------------------------------------------
data_dir = sys.argv[1]
# -----------------------------------------------------------------------------
if len(sys.argv)>2:
    all_data = sys.argv[2]
else:
    all_data = False
# -----------------------------------------------------------------------------
file1 = glob(pjoin(data_dir, 'results_1024.txt'))[0]
x, ndec, ndeh, phi, EF = np.loadtxt(file1,unpack = True)
# -----------------------------------------------------------------------------
file2 = glob(pjoin(data_dir, 'input_vd_20.txt'))[0]
dtype1 = np.dtype([('vname', 'S1'), ('value', 'f8')])
if os.path.exists(pjoin(data_dir, file2)):
    data = np.loadtxt(pjoin(data_dir, file2),dtype=dtype1, skiprows=0,usecols=(0,2))
else:
    print("No Data")
    exit()
# -----------------------------------------------------------------------------
value_arr = data['value']
# The following should have the same order of variable names as given in the corresponding input variable file
NUM_TS, write_interval, DT_coeff, vd, n0, Tec, Teh, Teb, Ti, mi, alp, beta, NC, Time, wpet_1, wpet_2 = value_arr
# -----------------------------------------------------------------------------
print('Saving data...')
if all_data:
    pro_data = 'processed_results_all.npz'
    np.savez_compressed(pjoin(data_dir,pro_data), x=x, ndec=ndec, ndeh=ndeh, phi=phi, EF=EF, NUM_TS=NUM_TS,
    write_interval=write_interval, DT_coeff=DT_coeff, vd=vd, n0=n0, Tec=Tec, Teh=Teh, Teb=Teb, Ti=Ti, mi=mi, alp=alp, beta=beta,
    NC=NC,Time=Time,wpet_1=wpet_1, wpet_2=wpet_2)
else:
    pro_data = 'processed_results_E.npz'
    np.savez_compressed(pjoin(data_dir,pro_data), x = x, EF = EF)
print('Processed data saved to %s'%(pjoin(data_dir,pro_data)))
