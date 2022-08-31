#!/usr/bin/env python
"""
This file is to be called for shortening the memory size of the data file results_1024.txt.
It can compress the file size from around 1 GB to around 150 MB.
To Run: python3 data_process_npz.py data/foldername/ (folder location)
"""
import numpy as np
from os.path import join as pjoin
import os.path
import sys
from glob import glob


data_dir = sys.argv[1]

if len(sys.argv)>2:
    all_data = sys.argv[2]
else:
    all_data = False


file = glob(pjoin(data_dir, 'results_1024.txt'))[0]

x, ndec, ndeh, phi, EF = np.loadtxt(file,unpack = True)

print('Saving data...')
if all_data:
    pro_data = 'processed_results_all.npz'
    np.savez_compressed(pjoin(data_dir,pro_data), x = x, ndec = ndec, ndeh = ndeh, phi = phi, EF = EF)
else:
    pro_data = 'processed_results_E.npz'
    np.savez_compressed(pjoin(data_dir,pro_data), x = x, EF = EF)
print('Processed data saved to %s'%(pjoin(data_dir,pro_data)))
