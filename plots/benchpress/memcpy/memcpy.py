import numpy as np
import math
import glob
from benchpress import prof

n_procs = prof.n_gpus
n_socket = n_procs / 2
files = glob.glob("%s/memcpy.*.out"%prof.folder)

class Times():
    times = ""

    def __init__(self):
        self.times = list()

    def add_time(self, pos, time):
        if pos == len(self.times):
            self.times.append(time)
        else:
            if time < self.times[pos]:
                self.times[pos] = time

class MemcpyTimes():
    on_socket = ""
    off_socket = ""
    across_socket = ""
    d2d = False

    def __init__(self, d2d = False):
        self.on_socket = Times()
        self.off_socket = Times()
        if (d2d):
            self.across_socket = Times()
            self.d2d = True

test_list = ""
time_list = ""

cpu = 0
gpu0 = 0
gpu1 = 0

h2d = MemcpyTimes()
d2h = MemcpyTimes()
d2d = MemcpyTimes(True)

for filename in files:
    f = open(filename, 'r')
    for line in f:
        if "Memcpy Host To Device" in line:
            test_list = h2d
        elif "Memcpy Device To Host" in line:
            test_list = d2h
        elif "Memcpy Device To Device" in line:
            test_list = d2d
        elif "app" in line:
            break
        elif "CPU" in line:
            header = line.rsplit(":")[0]
            cpu = (int)((header.rsplit(',')[0]).rsplit( )[-1])
            gpu_header = header.rsplit(',')[-1]
            if "<->" in header:
                gpu0 = (int)(gpu_header.rsplit(' ')[2])
                gpu1 = (int)(gpu_header.rsplit(' ')[-1])
                if ((cpu < n_socket and gpu0 < n_socket and gpu1 < n_socket) or 
                        (cpu >= n_socket and gpu0 >= n_socket and gpu1 >= n_socket)):
                    time_list = test_list.on_socket
                elif ((cpu < n_socket and gpu0 >= n_socket and gpu1 >= n_socket) or
                        (cpu >= n_socket and gpu0 < n_socket and gpu1 < n_socket)):
                    time_list = test_list.across_socket
                else:
                    time_list = test_list.off_socket
            else:
                gpu0 = (int)(gpu_header.rsplit(' ')[-1])
                if (gpu0 < n_socket and cpu < n_socket) or (gpu0 >= n_socket and cpu >= n_socket):
                    time_list = test_list.on_socket
                else:
                    time_list = test_list.off_socket
            times = line.rsplit('\t')
            for i in range(1, len(times)-1):
                time_list.add_time(i-1, (float)(times[i]))

    f.close()
