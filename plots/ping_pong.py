import numpy as np
import math
import glob
import prof

n_nodes = 2
ppn = prof.n_gpus
n_procs = ppn*n_nodes
n_socket = ppn / 2

files = glob.glob("%s/ping_pong.*.out"%prof.folder)

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
    import pyfancyplot.plot as plt
    if 1:
        # CPU Ping Pong 
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = 3)
        x_data = [2**i for i in range(len(cpu_times.on_socket))]
        plt.line_plot(cpu_times.on_socket, x_data, label = "On-Socket")
        plt.line_plot(cpu_times.on_node, x_data, label = "On-Node")
        plt.line_plot(cpu_times.network, x_data, label = "Network")
        plt.add_anchored_legend(ncol=3)
        plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
        plt.set_scale('log', 'log')
        plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
        plt.save_plot("%s_cpu_ping_pong.pdf"%prof.computer)

    if 1:
        # GPU Ping Pong 
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = 3)
        x_data = [2**i for i in range(len(gpu_times.on_socket))]
        plt.line_plot(gpu_times.on_socket, x_data, label = "On-Socket")
        plt.line_plot(gpu_times.on_node, x_data, label = "On-Node")
        plt.line_plot(gpu_times.network, x_data, label = "Network")
        plt.add_anchored_legend(ncol=3)
        plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
        plt.set_scale('log', 'log')
        plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
        plt.save_plot("%s_gpu_ping_pong.pdf"%prof.computer)



