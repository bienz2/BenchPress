all : time_memcpy time_memcpy_peer time_ping_pong profile_ping_pong time_inj_bw time_node_ping_pong time_allreduce time_memcpy_all time_memcpy_all_node time_ping_pong_multiple time_comparison time_memcpy_async time_node_async time_memcpy_multiple time_node_async_multiple stream


time_memcpy : time_memcpy.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_memcpy.cu -o time_memcpy

time_memcpy_async : time_memcpy_async.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_memcpy_async.cu -o time_memcpy_async

time_memcpy_peer : time_memcpy_peer.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_memcpy_peer.cu -o time_memcpy_peer

time_ping_pong : time_ping_pong.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_ping_pong.cu -o time_ping_pong

time_ping_pong_multiple : time_ping_pong_multiple.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_ping_pong_multiple.cu -o time_ping_pong_multiple

time_comparison : time_comparison.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_comparison.cu -o time_comparison

profile_ping_pong : profile_ping_pong.cu
	nvcc -arch=sm_70 -ccbin=mpicxx profile_ping_pong.cu -o profile_ping_pong

time_inj_bw : time_inj_bw.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_inj_bw.cu -o time_inj_bw

time_node_ping_pong : time_node_ping_pong.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_node_ping_pong.cu -o time_node_ping_pong

time_node_async : time_node_async.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_node_async.cu -o time_node_async

time_allreduce : time_allreduce.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_allreduce.cu -o time_allreduce

time_memcpy_all : time_memcpy_all.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_memcpy_all.cu -o time_memcpy_all

time_memcpy_multiple : time_memcpy_multiple.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_memcpy_multiple.cu -o time_memcpy_multiple

time_memcpy_all_node : time_memcpy_all_node.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_memcpy_all_node.cu -o time_memcpy_all_node

time_node_async_multiple : time_node_async_multiple.cu
	nvcc -arch=sm_70 -ccbin=mpicxx time_node_async_multiple.cu -o time_node_async_multiple

stream : stream.c
	nvcc -o stream stream.c

clean :
	rm time_memcpy time_memcpy_peer time_ping_pong profile_ping_pong time_inj_bw time_node_ping_pong time_ping_pong_multiple time_comparison time_memcpy_async time_node_async time_memcpy_multiple time_node_async_multiple time_allreduce time_memcpy_all time_memcpy_all_node stream


