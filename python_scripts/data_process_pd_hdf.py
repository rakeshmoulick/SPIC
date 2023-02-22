import pandas as pd
import numpy as np
import argparse
import glob
from os.path import join as pjoin
from os.path import splitext, basename

# Set up command-line argument parser
parser = argparse.ArgumentParser(
    description="Load data from a text files into a Pandas dataframe and save to compressed HDF5 file."
)
parser.add_argument("data_dir", help="Path to the text file containing the data.")

# Parse command-line arguments
args = parser.parse_args()

domain_size = 1025

hdf5_filename = splitext(args.data_dir)[0] + "data.h5"
hdf_file = pd.HDFStore(hdf5_filename, mode="w", complevel=9, complib="zlib")

glob_pattern = "**/*.txt"

data_files = glob.glob(pjoin(args.data_dir, glob_pattern), recursive=True)

# Load all text files into a dictionary of dataframes
dfs = {}

for file in data_files:
    # Get dataframe key from filename
    key = splitext(basename(file))[0]
    print(f"Processing {key}.txt")
    if "results" in file:
        col_names = [
            "density_electrons_cold",
            "density_electrons_hot",
            "den_electrons_beam",
            "phi",
            "efield",
        ]
    elif "ke" in file:
        col_names = [
            "Time",
            "ke_ions",
            "ke_electrons_cold",
            "ke_electrons_hot",
            "ke_electrons_beam",
        ]
    else:
        raise Exception("Data not recognized. Text data should be either results_<>.txt or ke_<>.txt.")

    if "results" in key:
        # Load data from file using NumPy and reshape
        data = np.loadtxt(file)
        tshape = data.shape[0] // domain_size
        time_data = np.reshape(data, (tshape, domain_size, len(col_names)+1), order='C')
        time_data[:, :, 0] = data[0:data.shape[0]:domain_size, 0].reshape((tshape, 1))
        time_data = time_data[:, :, 1:]

        for i in range(time_data.shape[0]):
            time_df = pd.DataFrame(time_data[i, :], columns=col_names)
            dataset_name = '{}/time_{}'.format(key, i)
            hdf_file.put(dataset_name, time_df)
    elif "ke" in key:
        # Load dataframe from text file
        df = pd.read_csv(file, sep="\s+", engine="python", names=col_names)
        # Add dataframe to dictionary
        hdf_file.put(key, df)

# close the HDF5 file
hdf_file.close()
