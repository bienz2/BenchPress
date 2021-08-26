computer = "summit"
n_gpus = 6
folder = "../benchmarks/%s"%computer
folder_out = "../figures/summit"

if 0:
    computer = "lassen"
    n_gpus = 4
    folder = "../benchmarks/%s/spectrum"%computer
    folder_out = "../figures/lassen/spectrum"
    if 0:
        folder = "../benchmarks/%s/mvapich"%computer
        folder_out = "../figures/lassen/mvapich"

max_ppn = 40

