import numpy as np
import math
import glob
from benchpress import prof

ppn = prof.max_ppn
files = glob.glob("%s/node_pong.*.out"%prof.folder)

class TimeList():
    ppn_times = ""

    def __init__(self, np):
        self.ppn_times = list()
        for i in range(np):
            self.ppn_times.append(list())

    def add_time(self, pos, time, ppn):
        time_list = self.ppn_times[ppn]

        if pos >= len(time_list):
            while pos > len(time_list):
                time_list.append(-1)
            time_list.append(time)
        else:
            if time < time_list[pos]:
                time_list[pos] = time

cpu_times = TimeList(ppn)
gpu_times = ""
if prof.cuda_aware:
    gpu_times = TimeList(prof.n_gpus)

time_list = ""

for filename in files:
    f = open(filename, 'r')
    for line in f:
        if "app" in line:
            break
        elif "Max-Rate" in line:
            if "CPU" in line:
                time_list = cpu_times
            elif prof.cuda_aware and "GPU" in line:
                time_list = gpu_times
        elif "Size" in line:
            times = line.rsplit('\t')
            size = (int)(times[0].rsplit(' ')[-1])
            for i in range(1, len(times)-1):
                time_list.add_time((int)(math.log2(size)), (float)(times[i]), i-1)
    f.close()



