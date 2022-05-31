from benchpress import prof
from benchpress.ping_pong import ping_pong, node_pong, mult_pong, mult_pong_split
import pyfancyplot.plot as plt

def plot_cpu_ping_pong(display_plot=False):
    cpu_times = ping_pong.cpu_times
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 3)
    x_data = [2**i for i in range(len(cpu_times.on_socket))]
    plt.line_plot(cpu_times.on_socket, x_data, label = "On-Socket")
    plt.line_plot(cpu_times.on_node, x_data, label = "On-Node")
    plt.line_plot(cpu_times.network, x_data, label = "Network")
    plt.add_anchored_legend(ncol=3)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'log')
    plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
    if display_plot:
        plt.display_plot()
    else:
        plt.save_plot("%s/%s_cpu_ping_pong.pdf"%(prof.folder_out, prof.computer))

def plot_gpu_ping_pong(display_plot=False):
    if prof.cuda_aware:
        gpu_times = ping_pong.gpu_times
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = 3)
        x_data = [2**i for i in range(len(gpu_times.on_socket))]
        plt.line_plot(gpu_times.on_socket, x_data, label = "On-Socket")
        plt.line_plot(gpu_times.on_node, x_data, label = "On-Node")
        plt.line_plot(gpu_times.network, x_data, label = "Network")
        plt.add_anchored_legend(ncol=3)
        plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
        plt.set_scale('log', 'log')
        plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
        if display_plot:
            plt.display_plot()
        else:
            plt.save_plot("%s/%s_gpu_ping_pong.pdf"%(prof.folder_out, prof.computer))


def plot_ping_pong_comparison(display_plot=False):
    if prof.cuda_aware:
        cpu_times = ping_pong.cpu_times
        gpu_times = ping_pong.gpu_times

        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = 3)
        x_data = [2**i for i in range(len(cpu_times.on_socket))]

        plt.line_plot(cpu_times.on_socket, x_data, label = "On-Socket")
        plt.line_plot(cpu_times.on_node, x_data, label = "On-Node")
        plt.line_plot(cpu_times.network, x_data, label = "Network")

        plt.line_plot(gpu_times.on_socket, x_data, tickmark='--')
        plt.line_plot(gpu_times.on_node, x_data, tickmark='--')
        plt.line_plot(gpu_times.network, x_data, tickmark='--')

        plt.add_anchored_legend(ncol=3)
        plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
        plt.set_scale('log', 'log')
        plt.add_labels("Message Size (Bytes)", "Measured Time (Seconds)")
        if display_plot:
            plt.display_plot()
        else:
            plt.save_plot("%s/%s_ping_pong_compare.pdf"%(prof.folder_out, prof.computer))



def plot_cpu_node_pong(display_plot=False):
    cpu_times = node_pong.cpu_times

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
    if display_plot:
        plt.display_plot()
    else:
        plt.save_plot("%s/%s_cpu_node_pong.pdf"%(prof.folder_out, prof.computer))


def plot_gpu_node_pong(display_plot=False):
    if prof.cuda_aware:
        gpu_times = node_pong.gpu_times

        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = prof.n_gpus)
        x_data = [2**i for i in range(len(gpu_times.ppn_times[0]))]
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
        if display_plot:
            plt.display_plot()
        else:
            plt.save_plot("%s/%s_gpu_node_pong.pdf"%(prof.folder_out, prof.computer))


def plot_cpu_mult_pong(display_plot=False, constant_node_size=False):
    cpu_times = ""
    if constant_node_size:
        cpu_times = mult_pong_split.cpu_times
    else:
        cpu_times = mult_pong.cpu_times

    i_list = list()
    for i in range(len(cpu_times.ppn_times[0])):
        if (cpu_times.ppn_times[0][i] == -1):
            continue
        i_list.append(i)

    np = [0,4,9,19,29,39]
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors = 6)
    x_data = [4*2**i for i in i_list]
    for n in np:
        plt.line_plot([cpu_times.ppn_times[n][i] for i in i_list], x_data, label = "NMsgs %d"%(n+1))
    plt.add_anchored_legend(ncol=3)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'log')
    if constant_node_size:
        plt.add_labels("Total Bytes", "Time (Seconds)")
    else:
        plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
    if display_plot:
        plt.display_plot()
    else:
        if constant_node_size:
            plt.save_plot("%s/%s_cpu_mult_pong_constant_node_size.pdf"%(prof.folder_out, prof.computer))
        else:
            plt.save_plot("%s/%s_cpu_mult_pong.pdf"%(prof.folder_out, prof.computer))


def plot_gpu_mult_pong(display_plot=False, constant_node_size=False):
    gpu_times = ""
    if prof.cuda_aware:
        if constant_node_size:
            gpu_times = mult_pong_split.gpu_times
        else:
            gpu_times = mult_pong.gpu_times

        i_list = list()
        for i in range(len(gpu_times.ppn_times[0])):
            if (gpu_times.ppn_times[0][i] == -1):
                continue
            i_list.append(i)

        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = prof.n_gpus)
        x_data = [4*2**i for i in i_list]
        for i in range(prof.n_gpus):
            plt.line_plot([gpu_times.ppn_times[i][il] for il in i_list], x_data, label = "NMsgs %d"%(i+1))
        plt.add_anchored_legend(ncol=prof.n_gpus/2)
        plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
        plt.set_scale('log', 'log')
        if constant_node_size:
            plt.add_labels("Total Bytes", "Time (Seconds)")
        else:
            plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
        if display_plot:
            plt.display_plot()
        else:
            if constant_node_size:
                plt.save_plot("%s/%s_gpu_mult_pong_constant_node_size.pdf"%(prof.folder_out, prof.computer))
            else:
                plt.save_plot("%s/%s_gpu_mult_pong.pdf"%(prof.folder_out, prof.computer))


