import numpy as np
import math
import glob
from benchpress import prof

n_nodes = 2
ppn = prof.n_gpus
n_procs = ppn*n_nodes
n_socket = prof.n_socket 

if prof.cuda_aware:

    files = glob.glob("%s/ping_pong_3step.*.out"%prof.folder)
    print(files)

    class Times():
        on_socket = ""
        on_node = ""
        network = ""

        def __init__(self):
            self.on_socket = list()
            self.on_node = list()
            self.network = list()

        def add_time(self, pos, time, rank0, rank1):
            time_list = ""
            if ((int)(rank0 / n_socket) == (int)(rank1 / n_socket)):
                time_list = self.on_socket
            elif ((int)(rank0 / ppn) == (int)(rank1 / ppn)):
                time_list = self.on_node
            else:
                time_list = self.network

            if pos == len(time_list):
                time_list.append(time)
            else:
                if time < time_list[pos]:
                    time_list[pos] = time


    gpu_times = Times()

    time_list = gpu_times

    rank0 = 0
    rank1 = 0
    size = 0

    for filename in files:
        f = open(filename, 'r')
        for line in f:
            if "Profiling" in line:
                continue
            if "app" in line:
                break
            elif prof.cuda_aware and "rank" in line:
                gpu_header = (line.rsplit(":")[0]).rsplit(' ')
                rank0 = (int)(gpu_header[3])
                rank1 = (int)(gpu_header[-1])
                times = line.rsplit('\t')
                for i in range(1, len(times)-1):
                    time_list.add_time(i-1, (float)(times[i]), rank0, rank1)
        f.close()

