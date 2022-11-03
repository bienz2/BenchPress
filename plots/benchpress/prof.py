import os, inspect, benchpress
from pathlib import Path

benchpress_dir = os.path.dirname(inspect.getfile(benchpress)) + "../../"
benchpress_dir = Path(benchpress_dir).parents[2]

computer = "summit"
n_gpus = 6
folder = "%s/benchmarks/%s"%(benchpress_dir, computer)
folder_out = "%s/figures/summit"%(benchpress_dir)

if 0:
    computer = "lassen"
    n_gpus = 4
    folder = "%s/benchmarks/%s/spectrum"%(benchpress_dir, computer)
    folder_out = "%s/figures/lassen/spectrum"%(benchpress_dir)
    if 0:
        folder = "%s/benchmarks/%s/mvapich"%(benchpress_dir, computer)
        folder_out = "%s/figures/lassen/mvapich"%(benchpress_dir)
n_socket = n_gpus / 2

if 1:
    computer = "tioga"
    n_gpus = 8
    folder = "%s/benchmarks/tioga"%(benchpress_dir)
    folder_out = "%s/figures/tioga"%(benchpress_dir)
n_socket = n_gpus

max_ppn = 40
cuda_aware = 1

