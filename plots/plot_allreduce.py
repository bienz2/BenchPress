import numpy as np
import pyfancyplot.plot as plt
import math
import glob

computer = "summit"

def parse(n_nodes):
    folder = "../benchmarks/%s"%computer
    files = glob.glob("%s/all_reduce_%d.*.out"%(folder, n_nodes))

    time_list = ""

    cuda_aware = list()
    three_step = list()
    three_step_msg = list()
    three_step_devptr = list()

    for filename in files:
        f = open(filename, 'r')
        for line in f:
            if "Cuda-Aware" in line:
                time_list = cuda_aware
            elif "Extra Message" in line:
                time_list = three_step_msg
            elif "DevPtr" in line:
                time_list = three_step_devptr
            elif "3-Step" in line:
                time_list = three_step
            elif "Warning" in line:
                continue
            elif "app" in line:
                break
            else:
                times = (line.rsplit('\n')[0]).rsplit('\t')
                for i in range(len(times)-1):
                    time = (float)(times[i])
                    if (i == len(time_list)):
                        time_list.append(time)
                    else:
                        if time < time_list[i]:
                            time_list[i] = time
        f.close()
    return (cuda_aware, three_step, three_step_msg, three_step_devptr)

def plot_data(data, n_nodes):
    x_data = [2**i for i in range(len(data[0]))]
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors=4)
    plt.line_plot(data[0], x_data, label = "Cuda-Aware")
    plt.line_plot(data[1], x_data, label = "Copy To CPU")
    plt.line_plot(data[2], x_data, label = "Extra Msg")
    yd = list()
    xd = list()
    for i in range(len(data[3])):
        if data[3][i] < 0:
            continue
        yd.append(data[3][i])
        xd.append(x_data[i])
    plt.line_plot(yd, xd, label = "Dup Devptr")
    plt.add_anchored_legend(ncol=2)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'log')
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    plt.save_plot("%s_allreduce_%d.pdf"%(computer, n_nodes))

data_2 = parse(2)
data_4 = parse(4)
data_8 = parse(8)
data_16 = parse(16)

plot_data(data_2, 2)
plot_data(data_4, 4)
plot_data(data_8, 8)
plot_data(data_16, 16)


