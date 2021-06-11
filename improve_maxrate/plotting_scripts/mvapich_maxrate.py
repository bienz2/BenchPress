import numpy as np
import math
import glob

ppn = 40
n_gpus = 4
files = ['../../benchmarks/lassen/04_13_21/maxrate.2422406.out','../../benchmarks/lassen/04_14_21/maxrate.2427559.out']

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

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    from common import set_figure, lighten_color
    colors = [lighten_color('tab:blue',0.8), lighten_color('tab:red',0.8), lighten_color('tab:green',0.8), lighten_color('tab:pink',0.8), lighten_color('tab:orange',0.8), lighten_color('tab:purple',0.8)]
    lw = 1.0

    if 1:
        # CPU Max-Rate 
        set_figure(fontsize=7.97, columnwidth=397.0*2/3, heightratio=0.5)
        fig, ax = plt.subplots()

        x_data = [2**i for i in range(len(cpu_times.ppn_times[0]))]
        ax.plot(x_data, cpu_times.ppn_times[0], color=colors[0], label = "PPN 1", linewidth=lw)
        ax.plot(x_data, cpu_times.ppn_times[4], color=colors[1], label = "PPN 5", linewidth=lw)
        ax.plot(x_data, cpu_times.ppn_times[9], color=colors[2], label = "PPN 10", linewidth=lw)
        ax.plot(x_data, cpu_times.ppn_times[19], color=colors[3], label = "PPN 20", linewidth=lw)
        ax.plot(x_data, cpu_times.ppn_times[29], color=colors[4], label = "PPN 30", linewidth=lw)
        ax.plot(x_data, cpu_times.ppn_times[39], color=colors[5], label = "PPN 40", linewidth=lw)

        #plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower right', borderaxespad=0, ncol=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Message Size (Bytes)")
        ax.set_ylabel("Measured Time (Seconds)")
        plt.savefig("../figures/lassen_cpu_maxrate.pdf")
        plt.show()

    if 1:
        # GPU Max-Rate 
        set_figure(fontsize=7.97, columnwidth=397.0*2/3, heightratio=0.5)
        fig, ax = plt.subplots()

        x_data = [2**i for i in range(len(gpu_times.ppn_times[0]))]
        for i in range(n_gpus):
            ax.plot(x_data, gpu_times.ppn_times[i], color=colors[i], label = "PPN %d"%(i+1), linewidth=lw)

        #plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
        ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower right', borderaxespad=0, ncol=(int)(n_gpus/2))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Message Size (Bytes)")
        ax.set_ylabel("Measured Time (Seconds)")
        #plt.savefig("../figures/lassen_gpu_maxrate.pdf")
        plt.show()

    if 1:
        # Output CPU and GPU Max-Rate data to file for plotting against runtime results
        gpu_x_data = [2**i for i in range(len(gpu_times.ppn_times[0]))]
        gpu_model = np.zeros((n_gpus,len(x_data)))
        for i in range(n_gpus): 
            gpu_model[i,:] = gpu_times.ppn_times[i]

        cpu_x_data = [2**i for i in range(len(cpu_times.ppn_times[0]))]
        cpu_model = np.zeros((6,len(cpu_x_data)))
        ppn = [1,5,10,20,30,40]
        for procs in ppn: 
            cpu_model[ppn.index(procs),:] = cpu_times.ppn_times[procs-1]

        # Save model data
        np.savez('data_files/maxrate_model_data.npz',
                cpu_msg_sizes = np.array(cpu_x_data),
                gpu_msg_sizes = np.array(gpu_x_data),
                cpu_ppn = np.array(ppn),
                cpu_maxrate = cpu_model,
                gpu_maxrate = gpu_model
        )
