.PHONY : cleanall all cleanrt

all: mstream

mstream: mstream.cu
	nvcc --cudart shared -o mstream mstream.cu

cleanall:
	rm *.o mstream
	rm *.ptx *.ptxas _app_cuda_version*
	rm _cuobjdump_*

cleanrt:
	rm _cuobjdump_*
	rm *.ptx *.ptxas _app_cuda_version*