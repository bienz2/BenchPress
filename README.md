# Benchmarking Suite for Heterogenenous Architectures
This program provides benchmarking tools for data movement on heterogenenous architecutres, including:
    - Inter-CPU data movement
    - Inter-GPU data movement
    - Cuda Memcpys
    - Injection bandwidth limitations

# Modeling and Plotting
The plotting folder includes python files that plot the benchmarks.  Furthermore, there is a modeling script that creates performance models for the various paths of inter-GPU data movement, and compares these paths for single messages, utilizing multiple CPU cores, and communicating numbers of messages.

# Summit Example Plots
The modeled performance of communicating data with GPUDirect, 3-step copying to CPU, and split 3-step, which distributes data across all CPUs on Summit.
![](figures/summit_3step_node_model)

The modeled performance of communicating multiple messages with GPUDirect (solid) versus copying to CPU with a single cudaMemcpyAsync before sending all messages bewteen CPUs (dotted) on Summit.
![](figures/summit_3step_node_model)

# License

This code is distributed under BSD: http://opensource.org/licenses/BSD-2-Clause

Please see `LICENSE.txt` for more information.
