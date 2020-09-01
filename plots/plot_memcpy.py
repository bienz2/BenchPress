import numpy as np
import pyfancyplot.plot as plt
import math
import glob

computer = "summit"
n_procs = 4
n_socket = n_procs / 2

folder = "../benchmarks/%s"%computer
files = glob.glob("%s/memcpy.*.out"%folder)

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
    on_socket = ""
    off_socket = ""

    def __init__(self):
        self.on_socket = Times()
        self.off_socket = Times()

class D2DTimes():
    on_socket = ""
    off_socket = ""
    across_socket = ""

    def __init__(self):
        self.on_socket = Times()
        self.off_socket = Times()
        self.across_socket = Times()

test_list = ""
time_list = ""

cpu = 0
gpu0 = 0
gpu1 = 0

h2d = MemcpyTimes()
d2h = MemcpyTimes()
d2d = D2DTimes()

for filename in files:
    f = open(filename, 'r')
    for line in f:
        if "Memcpy Host To Device" in line:
            test_list = h2d
        elif "Memcpy Device To Host" in line:
            test_list = d2h
        elif "Memcpy Device To Device" in line:
            test_list = d2d
        elif "app" in line:
            break
        elif "CPU" in line:
            header = line.rsplit(":")[0]
            cpu = (int)((header.rsplit(',')[0]).rsplit( )[-1])
            gpu_header = header.rsplit(',')[-1]
            if "<->" in header:
                gpu0 = (int)(gpu_header.rsplit(' ')[2])
                gpu1 = (int)(gpu_header.rsplit(' ')[-1])
                if ((cpu < n_socket and gpu0 < n_socket and gpu1 < n_socket) or 
                        (cpu >= n_socket and gpu0 >= n_socket and gpu1 >= n_socket)):
                    time_list = test_list.on_socket
                elif ((cpu < n_socket and gpu0 >= n_socket and gpu1 >= n_socket) or
                        (cpu >= n_socket and gpu0 < n_socket and gpu1 < n_socket)):
                    time_list = test_list.across_socket
                else:
                    time_list = test_list.off_socket
            else:
                gpu0 = (int)(gpu_header.rsplit(' ')[-1])
                if (gpu0 < n_socket and cpu < n_socket) or (gpu0 >= n_socket and cpu >= n_socket):
                    time_list = test_list.on_socket
                else:
                    time_list = test_list.off_socket
            times = line.rsplit('\t')
            for i in range(1, len(times)-1):
                time_list.add_time(i-1, (float)(times[i]))

    f.close()

x_data = [2**i for i in range(len(h2d.on_socket.times))]

if 1:
    # Memcpy Host To Device
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 2)
    plt.line_plot(h2d.on_socket.times, x_data, label = "On-Socket")
    plt.line_plot(h2d.off_socket.times, x_data, label = "Off-Socket")
    plt.add_anchored_legend(ncol=2)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'log')
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    plt.save_plot("%s_h2d_memcpy.pdf"%computer)


if 1:
    # Memcpy Device To Host
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 2)
    plt.line_plot(d2h.on_socket.times, x_data, label = "On-Socket")
    plt.line_plot(d2h.off_socket.times, x_data, label = "Off-Socket")
    plt.add_anchored_legend(ncol=2)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'log')
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    plt.save_plot("%s_d2h_memcpy.pdf"%computer)

if 1:
    # Memcpy Device To Device
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 3)
    plt.line_plot(d2d.on_socket.times, x_data, label = "On-Socket")
    plt.line_plot(d2d.off_socket.times, x_data, label = "Off-Socket")
    plt.line_plot(d2d.across_socket.times, x_data, label = "CPU Off-Socket")
    plt.add_anchored_legend(ncol=3)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'log')
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    plt.save_plot("%s_d2d_memcpy.pdf"%computer)

