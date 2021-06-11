import numpy as np
import math
import glob
import prof

ppn = prof.max_ppn
files = glob.glob("%s/node_pong.*.out"%prof.folder)

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
gpu_times = TimeList(prof.n_gpus)

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

if __name__=='__main__':
    import pyfancyplot.plot as plt
    
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
        plt.save_plot("%s/%s_cpu_node_pong.pdf"%(prof.folder_out, prof.computer))

    if 1:
        # GPU Max-Rate 
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = prof.n_gpus)
        x_data = [2**i for i in range(len(cpu_times.ppn_times[0]))]
        for ppn in range(prof.n_gpus):
            yd = list()
            xd = list()
            for i in range(len(gpu_times.ppn_times[ppn])):
                if gpu_times.ppn_times[ppn][i] < 0:
                    continue
                yd.append(gpu_times.ppn_times[ppn][i])
                xd.append(x_data[i])
            plt.line_plot(yd, xd, label = "PPN %d"%(ppn+1))
        plt.add_anchored_legend(ncol=prof.n_gpus / 2)
        plt.set_yticks([1e-5,1e-4],['1e-5','1e-4'])
        plt.set_scale('log', 'log')
        plt.set_yticks([1e-5,1e-4],['1e-5','1e-4'])
        plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
        plt.save_plot("%s/%s_gpu_node_pong.pdf"%(prof.folder_out, prof.computer))



