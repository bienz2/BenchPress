import numpy as np
import math
import glob
import prof

n_nodes = 2
ppn = prof.n_gpus
n_procs = n_nodes * ppn
n_socket = ppn / 2

filename = "%s/ping_pong_large.out"%prof.folder

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

comp_t = list()
decomp_t = list()
orig_s = list()
comp_s = list()
f = open("zfp.out", 'r')
for line in f:
    if "Compression" in line:
        list_words = (line.rsplit('\n')[0]).rsplit(' ')
        comp_t.append((float)((list_words[2]).rsplit(',')[0]))
        decomp_t.append((float)((list_words[5]).rsplit(',')[0]))
        orig_s.append((int)((list_words[7]).rsplit(',')[0]))
        comp_s.append((int)(list_words[-1]))
f.close()

if __name__=='__main__':
    import pyfancyplot.plot as plt

    orig_pos = [(int) (math.log2(s/8)) for s in orig_s]
    comp_pos = [(int) (math.log2(s/8)) for s in comp_s]

    if 1:
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = 3)
        x_data = [8*2**i for i in range(len(gpu_times.on_socket))]
        comp_times = [comp_t[i] + decomp_t[i] + gpu_times.network[comp_pos[i]] for i in range(len(orig_pos))]
        orig_times = [gpu_times.network[orig_pos[i]] for i in range(len(orig_pos))]
        plt.line_plot(orig_times, x_data, label = "Standard")
        plt.line_plot(comp_times, x_data, label = "Compressed")
        plt.add_anchored_legend(ncol=1)
        plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
        plt.set_scale('log', 'linear')
        plt.add_labels("Message Size (Bytes)", "Speed-Up Over CUDA Aware")
        plt.save_plot("%s_gpu_compressed.pdf"%prof.computer)

    if 1:
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = 3)
        x_data = [8*2**i for i in range(len(gpu_times.on_socket))]
        comp_times = [comp_t[i] + decomp_t[i] + gpu_times.network[comp_pos[i]] for i in range(len(orig_pos))]
        orig_times = [gpu_times.network[orig_pos[i]] for i in range(len(orig_pos))]
        plt.line_plot([orig_times[i] / comp_times[i] for i in range(len(orig_times))], 
                x_data, label = "Compressed")
        plt.add_anchored_legend(ncol=1)
        plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
        plt.set_scale('log', 'linear')
        plt.add_labels("Message Size (Bytes)", "Speed-Up Over CUDA Aware")
        plt.save_plot("%s_gpu_compressed_speedup.pdf"%prof.computer)




