import numpy as np

eager_i = 6
eager = 4*2**eager_i
rend_i = 12
rend = 4* 2 ** rend_i

class Model():
    alpha = 0
    beta = 0
    sizes = ""
    
    def __init__(self, times, start = 0):
        self.sizes = [4*2**(i+start) for i in range(len(times))]
        mat = list()
        t = list()
        for i in range(len(times)):
            mat.append([1,self.sizes[i]])
            t.append(times[i])
        A = np.matrix(mat)
        b = np.array(t)
        self.alpha, self.beta = np.linalg.lstsq(A, b)[0]

    def model_times(self):
        return [self.alpha + self.beta * s for s in self.sizes]

    def model_time_sizes(self, sizes):
        return [self.alpha + self.beta * s for s in sizes]

    def plot_model(self, times, label):
        plt.line_plot([self.alpha + self.beta * s for s in self.sizes], self.sizes, label = label)
        plt.color_ctr -= 1
        plt.line_plot(times, self.sizes, tickmark = "--")

    def get_model(self, size):
        return self.alpha, self.beta

## Ping Pong Models T_pong = alpha_pong + beta_pong * pong_bytes
import ping_pong
class ModeModel():
    short = ""
    eager = ""
    rend = ""
    sizes = ""

    def __init__(self, times):
        self.short = Model([times[i] for i in range(eager_i)])
        self.eager = Model([times[i] for i in range(eager_i, rend_i)], eager_i)
        self.rend = Model([times[i] for i in range(rend_i, len(times))], rend_i)
        self.sizes = [4*2**i for i in range(len(times))]

    def model_times(self):
        short_model = self.short.model_times()
        eager_model = self.eager.model_times()
        rend_model = self.rend.model_times()
        return short_model + eager_model + rend_model

    def plot_model(self, times, label):
        model = self.model_times()
        plt.line_plot(model, self.sizes, label = label)
        plt.color_ctr -= 1
        plt.line_plot(times, self.sizes, tickmark = "--")

    def get_model(self, size):
        if size < eager:
            return self.short.alpha, self.short.beta
        elif size < rend:
            return self.eager.alpha, self.eager.beta
        else:
            return self.rend.alpha, self.rend.beta

class PongModel():
    on_socket = ""
    on_node = ""
    network = ""
    sizes = ""

    def __init__(self, times, gpu = False):
        model = ModeModel
        if gpu:
            model = Model
        self.on_socket = model(times.on_socket)
        self.on_node = model(times.on_node)
        self.network = model(times.network)
        self.sizes = [4*2**i for i in range(len(times.on_socket))]

    #def plot_model(self, times, name):
    #    import pyfancyplot.plot as plt
    #    plt.add_luke_options()
    #    plt.set_palette(palette="deep", n_colors=3)
    #    self.on_socket.plot_model(times.on_socket, "On-Socket")
    #    self.on_node.plot_model(times.on_node, "On-Node")
    #    self.network.plot_model(times.network, "Network")
    #    print(self.sizes, times.network)
    #    plt.add_anchored_legend(ncol=3)
    #    plt.set_scale('log', 'log')
    #    plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
    #    plt.save_plot("%s/%s_%s_model.pdf"%(prof.folder_out, prof.computer, name))


cpu_model = PongModel(ping_pong.cpu_times)
gpu_model = PongModel(ping_pong.gpu_times, True)

if 1:
    print("Network:")
    print("CPU: Rend:", cpu_model.network.rend.alpha, cpu_model.network.rend.beta)
    print("CPU: Eager:", cpu_model.network.eager.alpha, cpu_model.network.eager.beta)
    print("CPU: Short:", cpu_model.network.short.alpha, cpu_model.network.short.beta)

if 1:
    print("Node:")
    print("CPU: Rend:", cpu_model.on_node.rend.alpha, cpu_model.on_node.rend.beta)
    print("CPU: Eager:", cpu_model.on_node.eager.alpha, cpu_model.on_node.eager.beta)
    print("CPU: Short:", cpu_model.on_node.short.alpha, cpu_model.on_node.short.beta)

if 1:
    print("Socket:")
    print("CPU: Rend:", cpu_model.on_socket.rend.alpha, cpu_model.on_socket.rend.beta)
    print("CPU: Eager:", cpu_model.on_socket.eager.alpha, cpu_model.on_socket.eager.beta)
    print("CPU: Short:", cpu_model.on_socket.short.alpha, cpu_model.on_socket.short.beta)

if 1:
    print("GPU: ", gpu_model.network.alpha, gpu_model.network.beta)
    print("GPU: ", gpu_model.on_node.alpha, gpu_model.on_node.beta)
    print("GPU: ", gpu_model.on_socket.alpha, gpu_model.on_socket.beta)

if __name__=='__main__':

    def model_times(model, sizes):
        return [model.alpha + model.beta*s for s in sizes]
       
    # Save model data
    np.savez('data_files/postal_model_data.npz',
            cpu_network_alpha = np.array([cpu_model.network.short.alpha,cpu_model.network.eager.alpha,cpu_model.network.rend.alpha]),
            cpu_network_beta = np.array([cpu_model.network.short.beta,cpu_model.network.eager.beta,cpu_model.network.rend.beta]),
            cpu_node_alpha = np.array([cpu_model.on_node.short.alpha,cpu_model.on_node.eager.alpha,cpu_model.on_node.rend.alpha]),
            cpu_node_beta = np.array([cpu_model.on_node.short.beta,cpu_model.on_node.eager.beta,cpu_model.on_node.rend.beta]),
            cpu_socket_alpha = np.array([cpu_model.on_socket.short.alpha,cpu_model.on_socket.eager.alpha,cpu_model.on_socket.rend.alpha]),
            cpu_socket_beta = np.array([cpu_model.on_socket.short.beta,cpu_model.on_socket.eager.beta,cpu_model.on_socket.rend.beta]),
            gpu_network = np.array([gpu_model.network.alpha, gpu_model.network.beta]),
            gpu_node = np.array([gpu_model.on_node.alpha, gpu_model.on_node.beta]),
            gpu_socket = np.array([gpu_model.on_socket.alpha, gpu_model.on_socket.beta])
    )

    ## Plot CPU and GPU Ping Pong Times
    #if 0:
    #    cpu_model.plot_model(ping_pong.cpu_times, "cpu")
    
    #if 1:
    #    gpu_model.plot_model(ping_pong.gpu_times, "gpu")
