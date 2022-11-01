import numpy as np
import math
import glob
from benchpress import prof

n_procs = prof.max_ppn
n_socket = (int)(n_procs / 2)
files = glob.glob("%s/memcpy_mult.*.out"%prof.folder)

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
    np_times = ""

    def __init__(self):
        self.np_times = list()
        for i in range(n_socket):
            self.np_times.append(Times())

test_list = ""
time_list = ""

np = 0
h2d = MemcpyTimes()
d2h = MemcpyTimes()

for filename in files:
    f = open(filename, 'r')
    for line in f:
        if "Memcpy Host To Device" in line:
            test_list = h2d
        elif "Memcpy Device To Host" in line:
            test_list = d2h
        elif "NP" in line:
            np = (int)((line.rsplit("\n")[0]).rsplit(" ")[-1])
            time_list = test_list.np_times[np];
        elif "app" in line:
            break
        elif "warning" in line:
            continue
        elif len(line) > 2:
            times = line.rsplit('\t')
            for i in range(0, len(times)-1):
                time_list.add_time(i, (float)(times[i]))

    f.close()



