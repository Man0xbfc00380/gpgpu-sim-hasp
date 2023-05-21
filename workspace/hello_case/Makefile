.PHONY : cleanall all cleanrt

all: hello

hello: hello.cu
	nvcc --cudart shared -o hello hello.cu

cleanall:
	rm *.o hello
	rm *.ptx *.ptxas _app_cuda_version*
	rm _cuobjdump_*

cleanrt:
	rm _cuobjdump_*
	rm *.ptx *.ptxas _app_cuda_version*