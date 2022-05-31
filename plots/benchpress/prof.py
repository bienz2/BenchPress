import os, inspect, benchpress
benchpress_dir = os.path.dirname(inspect.getfile(benchpress))

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

max_ppn = 40
cuda_aware = 0
