import numpy as np
import math
import glob
import prof

ppn = prof.max_ppn
files = glob.glob("%s/maxrate.*.out"%prof.folder)

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
        plt.save_plot("%s_cpu_maxrate.pdf"%prof.computer)

    if 1:
        # GPU Max-Rate 
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = prof.n_gpus)
        x_data = [2**i for i in range(len(gpu_times.ppn_times[0]))]
        for i in range(prof.n_gpus):
         plt.line_plot(gpu_times.ppn_times[i], x_data, label = "PPN %d"%(i+1))
        plt.add_anchored_legend(ncol=(int)(prof.n_gpus/2))
        plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
        plt.set_scale('log', 'log')
        plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
        plt.save_plot("%s_gpu_maxrate.pdf"%prof.computer)



