from benchpress import prof
from benchpress.memcpy import memcpy, memcpy_mult
import pyfancyplot.plot as plt

def plot_memcpy(display_plot=False):
    h2d = memcpy.h2d
    d2h = memcpy.d2h

    x_data = [2**i for i in range(len(h2d.on_socket.times))]

    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 2)
    plt.line_plot(h2d.on_socket.times, x_data, label = "On-Socket")
    plt.line_plot(h2d.off_socket.times, x_data, label = "Off-Socket")
    plt.line_plot(d2h.on_socket.times, x_data, tickmark='--')
    plt.line_plot(d2h.off_socket.times, x_data, tickmark='--')
    plt.add_anchored_legend(ncol=2)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'log')
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    if display_plot:
        plt.display_plot()
    else:
        plt.save_plot("%s/%s_memcpy.pdf"%(prof.folder_out, prof.computer))

def plot_memcpy_d2d(display_plot=False):
    d2d = memcpy.d2d

    x_data = [2**i for i in range(len(d2d.on_socket.times))]

    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 3)
    plt.line_plot(d2d.on_socket.times, x_data, label = "On-Socket")
    plt.line_plot(d2d.off_socket.times, x_data, label = "Off-Socket")
    plt.line_plot(d2d.across_socket.times, x_data, label = "CPU Off-Socket")
    plt.add_anchored_legend(ncol=3)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'log')
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    if display_plot:
        plt.display_plot()
    else:
        plt.save_plot("%s/%s_d2d_memcpy.pdf"%(prof.folder_out, prof.computer))

def plot_mult_memcpys(display_plot=False):
    h2d = memcpy_mult.h2d
    d2h = memcpy_mult.d2h

    x_data = [2**i for i in range(len(h2d.np_times[0].times))]

    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 6)
    plt.line_plot(h2d.np_times[0].times, x_data, label = "NP 1")
    plt.color_ctr -= 1
    plt.line_plot(d2h.np_times[0].times, x_data, '--')

    plt.line_plot(h2d.np_times[1].times, x_data, label = "NP 2")
    plt.color_ctr -= 1
    plt.line_plot(d2h.np_times[1].times, x_data, '--')

    plt.line_plot(h2d.np_times[3].times, x_data, label = "NP 4")
    plt.color_ctr -= 1
    plt.line_plot(d2h.np_times[3].times, x_data, '--')

    plt.line_plot(h2d.np_times[5].times, x_data, label = "NP 6")
    plt.color_ctr -= 1
    plt.line_plot(d2h.np_times[5].times, x_data, '--')

    plt.add_anchored_legend(ncol=2)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'log')
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    if display_plot:
        plt.display_plot()
    else:
        plt.save_plot("%s/%s_memcpy_mult.pdf"%(prof.folder_out, prof.computer))

def plot_mult_memcpy_slowdown(display_plot=False):
    h2d = memcpy_mult.h2d
    d2h = memcpy_mult.d2h

    x_data = [2**i for i in range(len(h2d.np_times[0].times))]

    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 6)
    plt.line_plot([1]*len(x_data), x_data, '--')

    h2d_slowdown = [h2d.np_times[1].times[x] / h2d.np_times[0].times[x] for x in range(len(x_data))]
    d2h_slowdown = [d2h.np_times[1].times[x] / d2h.np_times[0].times[x] for x in range(len(x_data))]
    plt.line_plot(h2d_slowdown, x_data, label = "NP 2")
    plt.color_ctr -= 1
    plt.line_plot(d2h_slowdown, x_data, '--')

    h2d_slowdown = [h2d.np_times[3].times[x] / h2d.np_times[0].times[x] for x in range(len(x_data))]
    d2h_slowdown = [d2h.np_times[3].times[x] / d2h.np_times[0].times[x] for x in range(len(x_data))]
    plt.line_plot(h2d_slowdown, x_data, label = "NP 4")
    plt.color_ctr -= 1
    plt.line_plot(d2h_slowdown, x_data, '--')

    h2d_slowdown = [h2d.np_times[5].times[x] / h2d.np_times[0].times[x] for x in range(len(x_data))]
    d2h_slowdown = [d2h.np_times[5].times[x] / d2h.np_times[0].times[x] for x in range(len(x_data))]
    plt.line_plot(h2d_slowdown, x_data, label = "NP 6")
    plt.color_ctr -= 1
    plt.line_plot(d2h_slowdown, x_data, '--')

    plt.add_anchored_legend(ncol=3)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'linear')
    plt.add_labels("Message Size (Bytes)", "Slowdown (1/SingleMemcpyTime)")
    if display_plot:
        plt.display_plot()
    else:
        plt.save_plot("%s/%s_memcpy_mult.pdf"%(prof.folder_out, prof.computer))

