import matplotlib.pyplot as plt
import matplotlib as mp
from os.path import splitext
import argparse
import h5py
import numpy as np

# Set up command-line argument parser
parser = argparse.ArgumentParser(
    description="Load data from a text files into a Pandas dataframe and save to compressed HDF5 file."
)
parser.add_argument("data_file", help="Path to the text file containing the data.")
parser.add_argument(
    "-k", "--ke", action="store_true", help="Add this if want to plot KE"
)
parser.add_argument(
    "-r", "--results", action="store_true", help="Add this if want to plot all results"
)
parser.add_argument(
    "-a",
    "--average",
    default=1,
    type=int,
    help="Average results data over time, e.g. 100",
)
# Parse command-line arguments
args = parser.parse_args()

# #### FIG SIZE CALC ############
figsize = np.array([100, 100 / 1.618])  # Figure size in mm
dpi = 300  # Print resolution
# ppi = np.sqrt(1920**2+1200**2)/24  # Screen resolution
ppi = np.sqrt(3840**2 + 2160**2) / 40  # Screen resolution

mp.rc("text", usetex=False)
mp.rc("font", family="sans-serif", size=10, serif="Computer Modern Roman")
mp.rc("axes", titlesize=10)
mp.rc("axes", labelsize=10)
mp.rc("xtick", labelsize=10)
mp.rc("ytick", labelsize=10)
mp.rc("legend", fontsize=10)

h5 = h5py.File(args.data_file, "r")
results_params = h5.attrs["results_params"]
ke_params = h5.attrs["ke_params"]

for key in h5.keys():
    if "ke" in key:
        ke_data = h5[key]
    elif "results" in key:
        results = h5[key]

if args.ke:
    for i in range(ke_data.shape[-1] - 1):
        fig, ax = plt.subplots(figsize=figsize / 25.4, constrained_layout=True, dpi=ppi)
        # Plot the regression lines
        ax.plot(ke_data[:, 0], ke_data[:, i + 1])

        # Add labels, titles, and legend to the plot
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{ke_params[i+1]}")
        ax.tick_params(axis="both", which="major")

        # Add a grid
        ax.grid(True, linestyle="--", alpha=0.5)

        # plt.tight_layout()
        plt.savefig(splitext(args.data_file)[0] + f"_{ke_params[i+1]}.png", dpi=dpi)
        plt.show()
if args.results:
    # Average the data
    results_mean = np.mean(results[-args.average :, :], axis=0)
    for i in range(results.shape[-1]):
        fig, ax = plt.subplots(figsize=figsize / 25.4, constrained_layout=True, dpi=ppi)
        # Plot the regression lines
        ax.plot(results_mean[:, i])

        # Add labels, titles, and legend to the plot
        ax.set_xlabel("domain")
        ax.set_ylabel(f"{results_params[i]}")
        ax.tick_params(axis="both", which="major")

        # Add a grid
        ax.grid(True, linestyle="--", alpha=0.5)

        # plt.tight_layout()
        plt.savefig(splitext(args.data_file)[0] + f"_{results_params[i]}.png", dpi=dpi)
        plt.show()
