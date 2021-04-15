import numpy as np
import math
import glob

n_nodes = 2
ppn = 4 
n_procs = ppn*n_nodes
n_socket = ppn / 2

folder = '../../benchmarks/lassen/04_13_21_cuda_bs/ping_pong_gpu_mvapich.*.out'
files = sorted(glob.glob(folder))
print(files)

#block_sizes = [131072, 262144, 524288, 786432, 1048576, 1310720, 1835008]
block_sizes = [131072, 262144, 524288, 786432, 1048576, 1310720]

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


gpu_times_master = []

time_list = ""

rank0 = 0
rank1 = 0
size = 0

for i in range(len(block_sizes)):
    block_size = block_sizes[i]
    filename = files[i]
    f = open(filename, 'r')

    gpu_times = Times()

    for line in f:
        if "app" in line:
            break
        elif "Ping-Pongs" in line:
            if "GPU" in line:
                time_list = gpu_times
        elif "GPU" in line:
            gpu_header = (line.rsplit(":")[0]).rsplit(' ')
            rank0 = (int)(gpu_header[3])
            rank1 = (int)(gpu_header[-1])
            times = line.rsplit('\t')
            for i in range(1, len(times)-1):
                time_list.add_time(i-1, (float)(times[i]), rank0, rank1)
    f.close()
    gpu_times_master.append(gpu_times)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    from common import set_figure, lighten_color
    if 1:
        colors = [lighten_color('tab:blue',0.8), lighten_color('tab:red',0.8), lighten_color('tab:green',0.8), lighten_color('tab:purple',0.8), lighten_color('tab:orange',0.8), lighten_color('tab:pink',0.8)]
        # GPU Ping Pong 
        set_figure(fontsize=7.97, columnwidth=397.0*2/3, heightratio=0.5)
        fig, ax = plt.subplots()

        for i in range(len(block_sizes)):
            gpu_times = gpu_times_master[i]
            x_data = [2**i for i in range(len(gpu_times.on_socket))]
            ax.plot(x_data, gpu_times.network, linewidth=0.75, color=colors[i], label = "Block Size "+str(block_sizes[i]))

        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower right', borderaxespad=0, ncol=3, fontsize='small')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Message Size (Bytes)")
        ax.set_ylabel("Measured Time (Seconds)")
        plt.savefig("../figures/mvapich_gpu_ping_pong_vary_cuda_block.pdf")
        
        # GPU Ping Pong 
        set_figure(fontsize=7.97, columnwidth=397.0*2/3, heightratio=0.5)
        fig, ax = plt.subplots()

        for i in range(len(block_sizes)):
            gpu_times = gpu_times_master[i]
            x_data = [2**i for i in range(len(gpu_times.on_socket))]
            ax.plot(x_data[15:], gpu_times.network[15:], linewidth=0.75, color=colors[i], label = "Block Size "+str(block_sizes[i]))

        print(gpu_times_master[0].network[-1]/gpu_times_master[-1].network[-1])

        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower right', borderaxespad=0, ncol=3, fontsize='small')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Message Size (Bytes)")
        ax.set_ylabel("Measured Time (Seconds)")
        plt.savefig("../figures/mvapich_gpu_ping_pong_vary_cuda_block_zoom.pdf")

