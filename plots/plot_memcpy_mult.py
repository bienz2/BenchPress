import numpy as np
import pyfancyplot.plot as plt
import math
import glob

computer = "lassen"
n_procs = 40
n_socket = (int)(n_procs / 2)

folder = "../benchmarks/%s"%computer
files = glob.glob("%s/memcpy_mult.*.out"%folder)

class Times():
    times = ""

    def __init__(self):
        self.times = list()

    def add_time(self, pos, time):
        if pos == len(self.times):
            self.times.append(time)
        else:
            if time < self.times[pos]:
                self.times[pos] = time

class MemcpyTimes():
    np_times = ""

    def __init__(self):
        self.np_times = list()
        for i in range(n_socket):
            self.np_times.append(Times())

test_list = ""
time_list = ""

np = 0
h2d = MemcpyTimes()
d2h = MemcpyTimes()

for filename in files:
    f = open(filename, 'r')
    for line in f:
        if "Memcpy Host To Device" in line:
            test_list = h2d
        elif "Memcpy Device To Host" in line:
            test_list = d2h
        elif "NP" in line:
            np = (int)((line.rsplit("\n")[0]).rsplit(" ")[-1])
            time_list = test_list.np_times[np];
        elif "app" in line:
            break
        elif "warning" in line:
            continue
        elif len(line) > 2:
            times = line.rsplit('\t')
            for i in range(0, len(times)-1):
                time_list.add_time(i, (float)(times[i]))

    f.close()

x_data = [2**i for i in range(len(h2d.np_times[0].times))]

if 1:
    # Memcpy Host To Device
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 6)
    plt.line_plot(h2d.np_times[0].times, x_data, label = "NP 1")
    plt.line_plot(h2d.np_times[1].times, x_data, label = "NP 2")
    plt.line_plot(h2d.np_times[3].times, x_data, label = "NP 4")
    plt.line_plot(h2d.np_times[5].times, x_data, label = "NP 6")
    plt.line_plot(h2d.np_times[7].times, x_data, label = "NP 8")
    plt.line_plot(h2d.np_times[9].times, x_data, label = "NP 10")
    plt.add_anchored_legend(ncol=3)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'log')
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    plt.save_plot("%s_h2d_memcpy_mult.pdf"%computer)


if 1:
    # Memcpy Device To Host
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 6)
    plt.line_plot(d2h.np_times[0].times, x_data, label = "NP 1")
    plt.line_plot(d2h.np_times[1].times, x_data, label = "NP 2")
    plt.line_plot(d2h.np_times[3].times, x_data, label = "NP 4")
    plt.line_plot(d2h.np_times[5].times, x_data, label = "NP 6")
    plt.line_plot(d2h.np_times[7].times, x_data, label = "NP 8")
    plt.line_plot(d2h.np_times[9].times, x_data, label = "NP 10")
    plt.add_anchored_legend(ncol=3)
    plt.set_yticks([1e-5, 1e-4],['1e-5','1e-4'])
    plt.set_scale('log', 'log')
    plt.set_yticks([1e-5, 1e-4],['1e-5','1e-4'])
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    plt.save_plot("%s_d2h_memcpy_mult.pdf"%computer)



