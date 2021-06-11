import numpy as np
import math
import glob

n_nodes = 2
ppn = 4 
n_procs = ppn*n_nodes
n_socket = ppn / 2

files = ['../../benchmarks/lassen/04_13_21/ping_pong_3step_mvapich.2422384.out']

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




cpu_times = Times()
gpu_times = Times()

time_list = ""

rank0 = 0
rank1 = 0
size = 0

for filename in files:
    f = open(filename, 'r')
    for line in f:
        if "app" in line:
            break
        elif "Ping-Pongs" in line:
            if "CPU" in line:
                time_list = cpu_times
            elif "GPU" in line:
                time_list = gpu_times
        elif "CPU" in line:
            cpu_header = (line.rsplit(":")[0]).rsplit(' ')
            rank0 = (int)(cpu_header[1])
            rank1 = (int)(cpu_header[-1])
            times = line.rsplit('\t')
            for i in range(1, len(times)-1):
                time_list.add_time(i-1, (float)(times[i]), rank0, rank1)
        elif "GPU" in line:
            gpu_header = (line.rsplit(":")[0]).rsplit(' ')
            rank0 = (int)(gpu_header[3])
            rank1 = (int)(gpu_header[-1])
            times = line.rsplit('\t')
            for i in range(1, len(times)-1):
                time_list.add_time(i-1, (float)(times[i]), rank0, rank1)
    f.close()

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    from common import set_figure, lighten_color
    if 0:
        # CPU Ping Pong 
        set_figure(fontsize=7.97, columnwidth=397.0*2/3, heightratio=0.5)
        fig, ax = plt.subplots()

        #plt.set_palette(palette="deep", n_colors = 3)
        colors = [lighten_color('tab:blue',0.8), lighten_color('tab:red',0.8), lighten_color('tab:green',0.8)]
        x_data = [2**i for i in range(len(cpu_times.on_socket))]
        ax.plot(x_data, cpu_times.on_socket, color=colors[0], label = "On-Socket")
        ax.plot(x_data, cpu_times.on_node, color=colors[1], label = "On-Node")
        ax.plot(x_data, cpu_times.network, color=colors[2], label = "Network")
        #plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])

        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower right', borderaxespad=0, ncol=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Message Size (Bytes)")
        ax.set_ylabel("Measured Time (Seconds)")
        plt.savefig("../figures/lassen_cpu_ping_pong_new.pdf")

    if 1:
        # GPU Ping Pong 
        set_figure(fontsize=7.97, columnwidth=397.0*2/3, heightratio=0.5)
        fig, ax = plt.subplots()

        colors = [lighten_color('tab:blue',0.8), lighten_color('tab:red',0.8), lighten_color('tab:green',0.8)]
        x_data = [2**i for i in range(len(gpu_times.on_socket))]
        ax.plot(x_data[:18], gpu_times.on_socket[:18], color=colors[0], label = "On-Socket")
        ax.plot(x_data[:18], gpu_times.on_node[:18], color=colors[1], label = "On-Node")
        ax.plot(x_data[:18], gpu_times.network[:18], color=colors[2], label = "Network")
        #plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])

        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower right', borderaxespad=0, ncol=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Message Size (Bytes)")
        ax.set_ylabel("Measured Time (Seconds)")
        plt.savefig("../figures/mvapich_gpu_ping_pong_3step_small.pdf")

        # GPU Ping Pong 
        set_figure(fontsize=7.97, columnwidth=397.0*2/3, heightratio=0.5)
        fig, ax = plt.subplots()

        colors = [lighten_color('tab:blue',0.8), lighten_color('tab:red',0.8), lighten_color('tab:green',0.8)]
        x_data = [2**i for i in range(len(gpu_times.on_socket))]
        ax.plot(x_data, gpu_times.on_socket, color=colors[0], label = "On-Socket")
        ax.plot(x_data, gpu_times.on_node, color=colors[1], label = "On-Node")
        ax.plot(x_data, gpu_times.network, color=colors[2], label = "Network")
        #plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])

        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower right', borderaxespad=0, ncol=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Message Size (Bytes)")
        ax.set_ylabel("Measured Time (Seconds)")
        plt.savefig("../figures/mvapich_gpu_ping_pong_3step_all.pdf")
