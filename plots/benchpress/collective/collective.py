import numpy as np
import math
import glob
from benchpress import prof

def parse(method_name, n_nodes):
    files = glob.glob("%s/%s_%d.*.out"%(prof.folder, method_name, n_nodes))

    time_list = ""

    cuda_aware = list()
    three_step = list()
    three_step_msg = list()
    three_step_devptr = list()

    for filename in files:
        f = open(filename, 'r')
        for line in f:
            if "Cuda-Aware" in line:
                time_list = cuda_aware
            elif "Extra Message" in line:
                time_list = three_step_msg
            elif "DevPtr" in line:
                time_list = three_step_devptr
            elif "3-Step" in line:
                time_list = three_step
            elif "Warning" in line:
                continue
            elif "app" in line:
                break
            else:
                times = (line.rsplit('\n')[0]).rsplit('\t')
                for i in range(len(times)-1):
                    time = (float)(times[i])
                    if (i == len(time_list)):
                        time_list.append(time)
                    else:
                        if time < time_list[i]:
                            time_list[i] = time
        f.close()
    return (cuda_aware, three_step, three_step_msg, three_step_devptr)


