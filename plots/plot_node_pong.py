import numpy as np
import pyfancyplot.plot as plt
import math
import glob

computer = "summit"
ppn = 40
n_gpus = 6

folder = "../benchmarks/%s"%computer
files = glob.glob("%s/node_pong.*.out"%folder)

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
gpu_times = TimeList(n_gpus)

time_list = ""

for filename in files:
    f = open(filename, 'r')
    for line in f:
        if "app" in line:
            break
        elif "Max-Rate" in line:
            if "CPU" in line:
                time_list = cpu_times
            elif "GPU" in line:
                time_list = gpu_times
        elif "Size" in line:
            times = line.rsplit('\t')
            size = (int)(times[0].rsplit(' ')[-1])
            for i in range(1, len(times)-1):
                time_list.add_time((int)(math.log2(size)), (float)(times[i]), i-1)
    f.close()


if 1:
    # CPU Max-Rate 
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 6)
    x_data = [2**i for i in range(len(cpu_times.ppn_times[0]))]
    ppn_list = [0, 4, 9, 19, 29, 39]
    for p in ppn_list:
        yd = list()
        xd = list()
        for i in range(len(cpu_times.ppn_times[p])):
            if cpu_times.ppn_times[p][i] < 0:
                continue
            yd.append(cpu_times.ppn_times[p][i])
            xd.append(x_data[i])
        plt.line_plot(yd, xd, label = "PPN %d"%(p+1))
    plt.add_anchored_legend(ncol=3)
    plt.set_yticks([1e-6,1e-5,1e-4],['1e-6','1e-5','1e-4'])
    plt.set_scale('log', 'log')
    plt.set_yticks([1e-6,1e-5,1e-4],['1e-6','1e-5','1e-4'])
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    plt.save_plot("%s_cpu_node_pong.pdf"%computer)

if 1:
    # GPU Max-Rate 
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 4)
    x_data = [2**i for i in range(len(cpu_times.ppn_times[0]))]
    ppn_list = [0,1,2,3]
    for p in ppn_list:
        yd = list()
        xd = list()
        for i in range(len(gpu_times.ppn_times[p])):
            if gpu_times.ppn_times[p][i] < 0:
                continue
            yd.append(gpu_times.ppn_times[p][i])
            xd.append(x_data[i])
        plt.line_plot(yd, xd, label = "PPN %d"%(p+1))
    plt.add_anchored_legend(ncol=3)
    plt.set_yticks([1e-5,1e-4],['1e-5','1e-4'])
    plt.set_scale('log', 'log')
    plt.set_yticks([1e-5,1e-4],['1e-5','1e-4'])
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    plt.save_plot("%s_gpu_node_pong.pdf"%computer)



