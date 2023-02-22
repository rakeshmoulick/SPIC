# python data_process.py path_to_data_dir
# Note: path_to_data_dir should contain either of ke_<>.txt and results_<>.txt or both.
# The script will store the processed data in the same directory as the data.
import numpy as np
import argparse
import glob
from os.path import join as pjoin
from os.path import splitext, basename
import h5py


# Set up command-line argument parser
parser = argparse.ArgumentParser(
    description="Load data from a text files into a Pandas dataframe and save to compressed HDF5 file."
)
parser.add_argument("data_dir", help="Path to the text file containing the data.")

# Parse command-line arguments
args = parser.parse_args()

domain_size = 1025
# NT = 10001

hdf5_filename = splitext(args.data_dir)[0] + "data.h5"
f = h5py.File(hdf5_filename, "w")
f.attrs["NC"] = domain_size

glob_pattern = "**/*.txt"

data_files = glob.glob(pjoin(args.data_dir, glob_pattern), recursive=True)


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
        f.attrs["results_params"] = col_names
        # Load data from file using NumPy and reshape
        data = np.loadtxt(file)
        tshape = data.shape[0] // domain_size
        time_data = np.reshape(
            data, (tshape, domain_size, len(col_names) + 1), order="C"
        )
        time_data[:, :, 0] = data[0 : data.shape[0] : domain_size, 0].reshape(
            (tshape, 1)
        )
        time_data = time_data[:, :, 1:]
        dsetResult = f.create_dataset(
            key,
            (time_data.shape[0], domain_size, len(col_names)),
            compression="gzip",
            compression_opts=9,
            dtype=time_data.dtype,
        )
        dsetResult[...] = time_data
    elif "ke" in file:
        col_names = [
            "Time",
            "ke_ions",
            "ke_electrons_cold",
            "ke_electrons_hot",
            "ke_electrons_beam",
        ]
        f.attrs["ke_params"] = col_names
        data = np.loadtxt(file)
        dsetKE = f.create_dataset(
            key,
            (data.shape),
            compression="gzip",
            compression_opts=9,
            dtype=data.dtype,
        )
        dsetKE[...] = data
    else:
        raise Exception(
            "Data not recognized. Text data should be either results_<>.txt or ke_<>.txt."
        )
