import numpy as np
import pyfancyplot.plot as plt
import math
import glob

computer = "summit"
ppn = 40
n_gpus = 6

folder = "../benchmarks/%s"%computer
files = glob.glob("%s/maxrate.*.out"%folder)

class TimeList():
    ppn_times = ""

    def __init__(self, np):
        self.ppn_times = list()
        for i in range(np):
            self.ppn_times.append(list())

    def add_time(self, pos, time, ppn):
        time_list = self.ppn_times[ppn]

        if pos == len(time_list):
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
    plt.line_plot(cpu_times.ppn_times[0], x_data, label = "PPN 1")
    plt.line_plot(cpu_times.ppn_times[4], x_data, label = "PPN 5")
    plt.line_plot(cpu_times.ppn_times[9], x_data, label = "PPN 10")
    plt.line_plot(cpu_times.ppn_times[19], x_data, label = "PPN 20")
    plt.line_plot(cpu_times.ppn_times[29], x_data, label = "PPN 30")
    plt.line_plot(cpu_times.ppn_times[39], x_data, label = "PPN 40")
    plt.add_anchored_legend(ncol=3)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'log')
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    plt.save_plot("%s_cpu_maxrate.pdf"%computer)

if 1:
    # GPU Max-Rate 
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 4)
    x_data = [2**i for i in range(len(gpu_times.ppn_times[0]))]
    plt.line_plot(gpu_times.ppn_times[0], x_data, label = "PPN 1")
    plt.line_plot(gpu_times.ppn_times[1], x_data, label = "PPN 2")
    plt.line_plot(gpu_times.ppn_times[2], x_data, label = "PPN 3")
    plt.line_plot(gpu_times.ppn_times[3], x_data, label = "PPN 4")
    plt.add_anchored_legend(ncol=3)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'log')
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    plt.save_plot("%s_gpu_maxrate.pdf"%computer)



