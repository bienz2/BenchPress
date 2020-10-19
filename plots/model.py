import prof
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

    def model_times(self, sizes):
        return [self.alpha + self.beta * s for s in sizes]

    def plot_model(self, times, label):
        plt.line_plot([self.alpha + self.beta * s for s in self.sizes], self.sizes, label = label)
        plt.color_ctr -= 1
        plt.line_plot(times, self.sizes, tickmark = "--")

    def get_model(self, size):
        return self.alpha, self.beta


## First, T_memcpy = alpha_memcpy + beta_memcpy*memcpy_bytes
import memcpy
class MemcpyModel():
    on_socket = ""
    off_socket = ""
    across_socket = ""

    def __init__(self, times):
        self.on_socket = Model(times.on_socket.times)
        self.off_socket = Model(times.off_socket.times)
        if (times.d2d):
            self.across_socket = Model(times.across_socket.times)

    def plot_model(self, times, name):
        import pyfancyplot.plot as plt
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors=3)
        self.on_socket.plot_model(times.on_socket.times, "On-Socket")
        self.off_socket.plot_model(times.off_socket.times, "Off-Socket")
        n_cols = 2
        if (times.d2d):
            self.across_socket.plot_model(times.across_socket.times, "Across-Socket")
            n_cols = 3
        plt.add_anchored_legend(ncol=n_cols)
        plt.set_scale('log', 'log')
        plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
        plt.save_plot("%s_%s_model.pdf"%(prof.computer, name))


h2d_model = MemcpyModel(memcpy.h2d)
print("H2D:", h2d_model.alpha, h2d_model.beta)
d2h_model = MemcpyModel(memcpy.d2h)
print("D2H:", d2h_model.alpha, d2h_model.beta)
d2d_model = MemcpyModel(memcpy.d2d)



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

    def plot_model(self, times, name):
        import pyfancyplot.plot as plt
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors=3)
        self.on_socket.plot_model(times.on_socket, "On-Socket")
        self.on_node.plot_model(times.on_node, "On-Node")
        self.network.plot_model(times.network, "Network")
        plt.add_anchored_legend(ncol=3)
        plt.set_scale('log', 'log')
        plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
        plt.save_plot("%s_%s_model.pdf"%(prof.computer, name))


cpu_model = PongModel(ping_pong.cpu_times)
print("CPU: Rend:", cpu_model.network.rend_model.alpha, cpu_model.network.rend_model.beta)
print("CPU: Eager:", cpu_model.network.eager_model.alpha, cpu_model.network.eager_model.beta)
gpu_model = PongModel(ping_pong.gpu_times, True)



## Model CUDA-Aware (gpu_model) vs 3Step (memcpy + cpu_model + memcpy)
import node_pong
class NodeModel():
    omega = ""
    max_ppn = 0

    def __init__(self, ppn_times, cpu_model):
        self.max_ppn = len(ppn_times)
        mat = list()
        t = list()
        for i in range(4, self.max_ppn):
            ppn = i+1
            for j in range(len(ppn_times[i])):
                if (ppn_times[i][j] < 0):
                    continue
                size = 4*2**j 
                s = size / ppn
                if (s >= rend):
                    alpha, beta = cpu_model.network.get_model(size)
                    t.append(ppn_times[i][j] - alpha)
                #elif (size >= eager):
                #    t.append(ppn_times[i][j] - cpu_model.network.eager.alpha)
                else:
                    continue
                mat.append([size])
        A = np.matrix(mat)
        b = np.array(t)
        self.omega, = np.linalg.lstsq(A, b)[0]

    def plot_model(self):
        import pyfancyplot.plot as plt

        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = 6)
        sizes = [4*2**i for i in range(len(node_pong.cpu_times.ppn_times[0]))]
        ppn_list = [1, 5, 10, 20, 40]
        for ppn in ppn_list:
            model_t = list()
            for s in sizes:
                size = s / ppn
                alpha, beta = cpu_model.network.get_model(size)
                if size >= rend and ppn >= 4:
                    model_t.append(alpha + node_model.omega * s)
                else:
                    model_t.append(alpha + beta*size)
            plt.line_plot(model_t, sizes, label = "PPN %d"%ppn)
            plt.color_ctr -= 1
            xdata = list()
            ydata = list()
            for i in range(len(sizes)):
                if node_pong.cpu_times.ppn_times[ppn-1][i] > 0:
                    xdata.append(sizes[i])
                    ydata.append(node_pong.cpu_times.ppn_times[ppn-1][i])
            plt.line_plot(ydata, xdata, tickmark="--")
        plt.add_anchored_legend(ncol=3)
        plt.set_scale('log', 'log')
        plt.set_yticks([1e-6,1e-5,1e-4],['1e-6','1e-5','1e-4'])
        plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
        plt.save_plot("%s_node_model.pdf"%(prof.computer))
            
node_model = NodeModel(node_pong.cpu_times.ppn_times, cpu_model)
gpu_node_model = NodeModel(node_pong.gpu_times.ppn_times, gpu_model)
print("CPU NODE:", node_model.omega)
print("GPU NODE:", gpu_node_model.omega)


## Multiple Ping Pongs
import mult_pong
class MultModel():
    sizes = ""
    theta = 0
    max_n_msgs = 0

    def __init__(self, nmsg_times, orig_model):
        self.sizes = [4*2**i for i in range(len(nmsg_times[0]))]
        mat = list()
        t = list()
        self.max_n_msgs = len(nmsg_times)
        for n_msg in range(1, len(nmsg_times)):
            for i in range(len(nmsg_times[n_msg])):
                if (nmsg_times[n_msg][i] < 0):
                    continue
                size = self.sizes[i] / (n_msg+1)
                if (size >= rend):
                    continue
                alpha, beta = orig_model.network.get_model(size)
                mat.append([(n_msg+1)])
                t.append(nmsg_times[n_msg][i] - alpha * (n_msg+1) - beta * self.sizes[i])
        A = np.matrix(mat)
        b = np.array(t)
        self.theta, = np.linalg.lstsq(A, b)[0]
        print(self.theta)

    def plot_model(self):
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors = 6)
        for i in range(0, self.max_n_msgs):
            ydata = list()
            j_list = list()
            for j in range(len(self.sizes)):
                if (mult_pong.gpu_times.ppn_times[i][j] > 0):
                    size = (int) (self.sizes[j] / (i+1))
                    j_list.append(j)
                    alpha, beta = gpu_model.network.get_model(size)
                    model = alpha * (i+1) + beta * self.sizes[j] + self.theta * i;
                    ydata.append(model)
            xdata = [self.sizes[j] for j in j_list]
            print(i, j_list)
            print(mult_pong.gpu_times.ppn_times[i])
            plt.line_plot([mult_pong.gpu_times.ppn_times[i][j] for j in j_list], xdata, label = "%d Msgs"%(i+1))
            plt.color_ctr -= 1
            plt.line_plot(ydata, xdata, tickmark = '--')
        plt.add_anchored_legend(ncol=3)
        plt.set_scale('log', 'log')
        plt.set_yticks([1e-6,1e-5,1e-4],['1e-6','1e-5','1e-4'])
        plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
        plt.save_plot("%s_mult_model.pdf"%(prof.computer))

    def get_model(self, n_msgs):
        model_t = list()
        for size in sizes:
            alpha, beta = gpu_model.network.get_model(size)
            model_t.append(alpha * n_msgs + beta * size * n_msgs + self.theta * (n_msgs-1))
        return model_t
        


gpu_mult_model = MultModel(mult_pong.gpu_times.ppn_times, gpu_model)



if __name__=='__main__':
    import pyfancyplot.plot as plt

    def model_times(model, sizes):
        return [model.alpha + model.beta*s for s in sizes]

    ## Plot Memcpy Times
    if 0:
        h2d_model.plot_model(memcpy.h2d, "h2d")
        d2h_model.plot_model(memcpy.d2h, "d2h")
        d2d_model.plot_model(memcpy.d2d, "d2d")

    ## Plot CPU and GPU Ping Pong Times
    if 0:
        cpu_model.plot_model(ping_pong.cpu_times, "cpu")
        gpu_model.plot_model(ping_pong.gpu_times, "gpu")

    ## Plot CUDA-Aware and 3Step Network Times
    if 0:
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors=2)

        ## Cuda-Aware is gpu_model
        gpu_model_t = gpu_model.network.model_times()

        ## 3Step is h2d + cpu_model + d2h
        h2d_model_t = h2d_model.on_socket.model_times()
        cpu_model_t = cpu_model.network.model_times()
        d2h_model_t = d2h_model.on_socket.model_times()
        model_t = [h2d_model_t[i] + cpu_model_t[i] + d2h_model_t[i] for i in range(len(h2d_model_t))]

        plt.line_plot(gpu_model_t, gpu_model.network.sizes, label = "CUDA-Aware")
        plt.line_plot(model_t, h2d_model.on_socket.sizes, label = "3 Step")
        plt.add_anchored_legend(ncol=2)
        plt.set_scale('log', 'log')
        plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
        plt.save_plot("%s_3step_model.pdf"%(prof.computer))

    ## Plot Max-Rate Model
    if 0:
        node_model.plot_model()

    ## Plot CUDA-aware, 3step, and 3step with max_ppn/n_gpus procs
    if 0:
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors=3)

        ## Cuda-Aware is gpu_model
        gpu_model_t = gpu_model.network.model_times()
        plt.line_plot(gpu_model_t, gpu_model.network.sizes, label = "CUDA-Aware")

        h2d_model_t = h2d_model.on_socket.model_times()
        d2h_model_t = d2h_model.on_socket.model_times()

        ## 3Step is h2d + cpu_model + d2h
        cpu_model_t = cpu_model.network.model_times()
        model_t = [h2d_model_t[i] + cpu_model_t[i] + d2h_model_t[i] for i in range(len(h2d_model_t))]
        plt.line_plot(model_t, h2d_model.on_socket.sizes, label = "3 Step")

        ## 3Step node 
        sizes = [4*2**i for i in range(len(node_pong.cpu_times.ppn_times[0]))]        
        ppn = (int) (prof.max_ppn / prof.n_gpus)
        model_t = list()
        for s in sizes:
            size = s / ppn
            alpha, beta = cpu_model.network.get_model(size)
            if size >= rend and ppn >= 4:
                model_t.append(alpha + node_model.omega * s)
            else:
                model_t.append(alpha + beta*size)
        xdata = list()
        ydata = list()
        for i in range(len(sizes)):
            if node_pong.cpu_times.ppn_times[ppn-1][i] > 0:
                xdata.append(sizes[i])
                ydata.append(h2d_model_t[i] + d2h_model_t[i] + node_pong.cpu_times.ppn_times[ppn-1][i])
        plt.line_plot(ydata, xdata, label = "3 Step Split")

        plt.add_anchored_legend(ncol=3)
        plt.set_scale('log', 'log')
        plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
        plt.save_plot("%s_3step_node_model.pdf"%(prof.computer))

    ## Plot Cost of Sending Multiple Messages
    if 0:
        gpu_mult_model.plot_model()


    ## Plot Model for Sending N Msgs of Different Sizes, Cuda-Aware vs 3Step
    if 1:
        plt.add_luke_options()
        plt.set_palette(palette="deep", n_colors=4)

        nmsg_list = [1, 5, 10, 50]
        xdata = gpu_mult_model.sizes

        for n_msgs in nmsg_list:
            sizes = [s * n_msgs for s in xdata]

            h2d_model_t = h2d_model.on_socket.model_times(sizes)
            d2h_model_t = d2h_model.on_socket.model_times(sizes)

            cpu_model_t = list()
            gpu_model_t = list()
            for s in sizes:
                alpha, beta = cpu_model.network.get_model(s/n_msgs)
                cpu_model_t.append(alpha*n_msgs + beta * s)
                alpha, beta = gpu_model.network.get_model(s/n_msgs)
                gpu_model_t.append(alpha*n_msgs + beta*s + gpu_mult_model.theta * (n_msgs-1))

            plt.line_plot(gpu_model_t, xdata, label = "%d Msgs"%n_msgs)
            plt.color_ctr -= 1
            plt.line_plot([h2d_model_t[i] + d2h_model_t[i] + cpu_model_t[i] for i in range(len(h2d_model_t))], xdata, tickmark='--')

        plt.add_anchored_legend(ncol=2)
        plt.set_scale('log', 'log')
        plt.add_labels("Message Size (Bytes)", "Time (Seconds)")
        plt.save_plot("%s_3step_mult_model.pdf"%(prof.computer))



