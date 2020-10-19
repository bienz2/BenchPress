import numpy as np
import math
import glob
import prof

ppn = prof.max_ppn
split=""
if 1:
    split="_split"

files = glob.glob("%s/mult_pong%s.*.out"%(prof.folder,split))

class TimeList():
    ppn_times = ""

    def __init__(self, np):
        self.ppn_times = list()
        for i in range(np):
            self.ppn_times.append(list())

    def add_time(self, pos, time, ppn):
        time_list = self.ppn_times[ppn]

        while (pos > len(time_list)):
            time_list.append(-1)
        if (pos == len(time_list)):
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
        elif "Multiple Messages" in line:
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

    i_list = list()
    for i in range(len(cpu_times.ppn_times[0])):
        if (cpu_times.ppn_times[0][i] == -1):
            continue
        i_list.append(i)

    if 1:
        # CPU Mult Pong 
        np = [0,4,9,19,29,39]
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = 6)
        x_data = [4*2**i for i in i_list]
        for n in np:
            plt.line_plot([cpu_times.ppn_times[n][i] for i in i_list], x_data, label = "NMsgs %d"%(n+1))
        plt.add_anchored_legend(ncol=3)
        plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
        plt.set_scale('log', 'log')
        if (split==""):
            plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
        else:
            plt.add_labels("Total Bytes", "Time (Seconds)")
        plt.save_plot("%s_cpu_mult%s_pong.pdf"%(prof.computer,split))

    if 1:
        # CPU Mult Slowdown 
        np = [0,4,9,19,29,39]
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = 6)
        x_data = [4*2**i for i in i_list]
        for n in np:
            y_data = [cpu_times.ppn_times[n][i] / cpu_times.ppn_times[0][i] for i in i_list]
            plt.line_plot(y_data, x_data, label = "NMsgs %d"%(n+1))
        plt.add_anchored_legend(ncol=3)
        plt.set_scale('log', 'linear')
        if (split==""):
            plt.add_labels("Message Size (Bytes)", "Times Slowdown")
        else:
            plt.add_labels("Total Bytes", "Times Slowdown")
        plt.save_plot("%s_cpu_mult%s_slowdown.pdf"%(prof.computer,split))



    i_list = list()
    for i in range(len(gpu_times.ppn_times[0])):
        if (gpu_times.ppn_times[0][i] == -1):
            continue
        i_list.append(i)

    if 1:
        # GPU Max-Rate 
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = prof.n_gpus)
        x_data = [4*2**i for i in i_list]
        for i in range(prof.n_gpus):
            plt.line_plot([gpu_times.ppn_times[i][il] for il in i_list], x_data, label = "NMsgs %d"%(i+1))
        plt.add_anchored_legend(ncol=prof.n_gpus/2)
        plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
        plt.set_scale('log', 'log')
        if (split==""):
            plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
        else:
            plt.add_labels("Total Bytes", "Time (Seconds)")
        plt.save_plot("%s_gpu_mult%s_pong.pdf"%(prof.computer,split))

    if 1:
        # GPU Diff X
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors=prof.n_gpus)
        x_data = [4*2**i for i in i_list]
        for i in range(prof.n_gpus):
            y_data = [gpu_times.ppn_times[i][il] / gpu_times.ppn_times[0][il] for il in i_list]
            plt.line_plot(y_data, x_data, label = "NMsgs %d"%(i+1))
        plt.add_anchored_legend(ncol=prof.n_gpus/2)
        plt.set_scale('log', 'linear')
        if (split==""):
            plt.add_labels("Message Size (Bytes)", "Times Slowdown")
        else:
            plt.add_labels("Total Bytes", "Times Slowdown")
        plt.save_plot("%s_gpu_mult%s_slowdown.pdf"%(prof.computer, split))



