from benchpress.collective import collective 
from benchpress import prof
import pyfancyplot.plot as plt

def plot_collective(method_name, n_nodes, display_plot=False):
    data = collective.parse(method_name, n_nodes)
    x_data = [2**i for i in range(len(data[0]))]
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors=4)
    if prof.cuda_aware:
        plt.line_plot(data[0], x_data, label = "Cuda-Aware")
    plt.line_plot(data[1], x_data, label = "3-Step")
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
    if display_plot:
        plt.display_plot()
    else:
        plt.save_plot("%s/%s_%s_%d.pdf"%(prof.folder_out, prof.computer, method_name, n_nodes))
        

def plot_collective_speedup(method_name, n_nodes, display_plot=False):
    data = collective.parse(method_name, n_nodes)
    x_data = [8*2**i for i in range(len(data[1]))]
    plt.add_luke_options()
    plt.set_palette(palette="deep", n_colors=4)
    plt.line_plot([1]*len(x_data), x_data)
    rel_data = ""
    if prof.cuda_aware:
        rel_data = data[0]
        plt.line_plot([rel_data[i]/data[1][i] for i in range(len(x_data))], x_data, label = "3-Step")         
    else:
        rel_data = data[1]
    plt.line_plot([rel_data[i]/data[2][i] for i in range(len(x_data))], x_data, label = "Extra Msg")
    yd = list()
    xd = list()
    for i in range(len(data[3])):
        if data[3][i] < 0:
            continue
        yd.append(rel_data[i] / data[3][i])
        xd.append(x_data[i])
    plt.line_plot(yd, xd, label = "Dup Devptr")
    plt.add_anchored_legend(ncol=3)
    plt.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3],['1e-7','1e-6','1e-5','1e-4','1e-3'])
    plt.set_scale('log', 'linear')
    if prof.cuda_aware:
        plt.add_labels("Message Size (Bytes)", "Speed-Up Over CUDA Aware")
    else:
        plt.add_labels("Message Size (Bytes)", "Speed-Up Over 3-Step")
    if display_plot:
        plt.display_plot()
    else:
        plt.save_plot("%s/%s_%s_speedup_%d.pdf"%(prof.folder_out, prof.computer, method_name, n_nodes))



def plot_allreduce(n_nodes, display_plot=False):
    plot_collective("all_reduce", n_nodes, display_plot)

def plot_allreduce_speedup(n_nodes, display_plot=False):
    plot_collective_speedup("all_reduce", n_nodes, display_plot)

def plot_alltoall(n_nodes, display_plot=False):
    plot_collective("all_to_all", n_nodes, display_plot)

def plot_alltoall_speedup(n_nodes, display_plot=False):
    plot_collective_speedup("all_to_all", n_nodes, display_plot)

def plot_alltoallv(n_nodes, display_plot=False):
    plot_collective("all_to_all_v", n_nodes, display_plot)

def plot_alltoallv_speedup(n_nodes, display_plot=False):
    plot_collective_speedup("all_to_all_v", n_nodes, display_plot)

