import numpy as np
import math
import glob

n_nodes = 2
ppn = 40 
n_procs = ppn*n_nodes
n_socket = ppn / 2

nic_files = sorted(glob.glob('../../benchmarks/lassen/04_08_21/ping_pong_cpu_nic.*.out'))
vader_files = sorted(glob.glob('../../benchmarks/lassen/04_08_21/ping_pong_cpu_vader.*.out'))

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




vader_times = Times()
nic_times = Times()

time_list = ""

rank0 = 0
rank1 = 0
size = 0

for filename in nic_files:
    f = open(filename, 'r')
    for line in f:
        if "app" in line:
            break
        elif "Ping-Pongs" in line:
            if "CPU" in line:
                time_list = nic_times
        elif "CPU" in line:
            cpu_header = (line.rsplit(":")[0]).rsplit(' ')
            rank0 = (int)(cpu_header[1])
            rank1 = (int)(cpu_header[-1])
            times = line.rsplit('\t')
            for i in range(1, len(times)-1):
                time_list.add_time(i-1, (float)(times[i]), rank0, rank1)
    f.close()

for filename in vader_files:
    f = open(filename, 'r')
    for line in f:
        if "app" in line:
            break
        elif "Ping-Pongs" in line:
            if "CPU" in line:
                time_list = vader_times
        elif "CPU" in line:
            cpu_header = (line.rsplit(":")[0]).rsplit(' ')
            rank0 = (int)(cpu_header[1])
            rank1 = (int)(cpu_header[-1])
            times = line.rsplit('\t')
            for i in range(1, len(times)-1):
                time_list.add_time(i-1, (float)(times[i]), rank0, rank1)
    f.close()

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    from common import set_figure, lighten_color
    if 1:
        # CPU Ping Pong 
        set_figure(fontsize=7.97, columnwidth=397.0*2/3, heightratio=0.5)
        fig, ax = plt.subplots()
        lw = 2.0

        colors = [lighten_color('tab:blue',0.8), lighten_color('tab:red',0.8), lighten_color('tab:green',0.8)]
        x_data = [2**i for i in range(len(vader_times.on_socket))]
        ax.plot(x_data, vader_times.on_socket, color=colors[0], linewidth=lw, label = "Vader")
        ax.plot(x_data, nic_times.on_socket, color=colors[1], linewidth=lw, label = "NIC")

        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower right', borderaxespad=0, ncol=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Message Size (Bytes)")
        ax.set_ylabel("Measured Time (Seconds)")
        plt.savefig("../../figures/shelbys_figures/lassen_cpu_ping_pong_ppn40_onsocket_intra.pdf")
    
    if 1:
        # CPU Ping Pong 
        set_figure(fontsize=7.97, columnwidth=397.0*2/3, heightratio=0.5)
        fig, ax = plt.subplots()
        lw = 2.0

        colors = [lighten_color('tab:blue',0.8), lighten_color('tab:red',0.8), lighten_color('tab:green',0.8)]
        x_data = [2**i for i in range(len(vader_times.on_socket))]
        ax.plot(x_data, vader_times.on_node, color=colors[0], linewidth=lw, label = "Vader")
        ax.plot(x_data, nic_times.on_node, color=colors[1], linewidth=lw, label = "NIC")
        #ax.plot(x_data, cpu_times.network, color=colors[2], linewidth=lw, label = "Network")

        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower right', borderaxespad=0, ncol=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Message Size (Bytes)")
        ax.set_ylabel("Measured Time (Seconds)")
        plt.savefig("../../figures/shelbys_figures/lassen_cpu_ping_pong_ppn40_onnode_intra.pdf")
