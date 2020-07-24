#MPICXX=mpicxx
MPICXX=CC
#MPIFLAGS="-arch=sm_70"

all : time_memcpy time_memcpy_ppn time_ping_pong profile_ping_pong time_inj_bw time_node_ping_pong time_allreduce time_ping_pong_multiple time_comparison stream


time_memcpy : time_memcpy.cu
	nvcc ${MPIFLAGS} -ccbin=${MPICXX} time_memcpy.cu -o time_memcpy

time_memcpy_ppn : time_memcpy_ppn.cu
	nvcc ${MPIFLAGS} -ccbin=${MPICXX} time_memcpy_ppn.cu -o time_memcpy_ppn

time_memcpy_ppn_partial : time_memcpy_ppn_partial.cu
	nvcc ${MPIFLAGS} -ccbin=${MPICXX} time_memcpy_ppn_partial.cu -o time_memcpy_ppn_partial

time_ping_pong : time_ping_pong.cu
	nvcc ${MPIFLAGS} -ccbin=${MPICXX} time_ping_pong.cu -o time_ping_pong

time_ping_pong_multiple : time_ping_pong_multiple.cu
	nvcc ${MPIFLAGS} -ccbin=${MPICXX} time_ping_pong_multiple.cu -o time_ping_pong_multiple

time_comparison : time_comparison.cu
	nvcc ${MPIFLAGS} -ccbin=${MPICXX} time_comparison.cu -o time_comparison

profile_ping_pong : profile_ping_pong.cu
	nvcc ${MPIFLAGS} -ccbin=${MPICXX} profile_ping_pong.cu -o profile_ping_pong

time_inj_bw : time_inj_bw.cu
	nvcc ${MPIFLAGS} -ccbin=${MPICXX} time_inj_bw.cu -o time_inj_bw

time_node_ping_pong : time_node_ping_pong.cu
	nvcc ${MPIFLAGS} -ccbin=${MPICXX} time_node_ping_pong.cu -o time_node_ping_pong

time_allreduce : time_allreduce.cu
	nvcc ${MPIFLAGS} -ccbin=${MPICXX} time_allreduce.cu -o time_allreduce

stream : stream.c
	nvcc -o stream stream.c

clean :
	rm time_memcpy time_memcpy_ppn time_memcpy_ppn_partial

