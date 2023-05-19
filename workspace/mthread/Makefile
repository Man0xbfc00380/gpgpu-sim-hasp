.PHONY : cleanall all cleanrt

all: vector

vector: main.o vector.o vector_kernel.o
	nvcc -lpthread --cudart shared -o vector main.o vector.o vector_kernel.o

vector_kernel.o: vector_kernel.cu vector_kernel.h
	nvcc --cudart shared -c vector_kernel.cu

vector.o: vector.cu vector.h vector_kernel.h
	nvcc --cudart shared -c vector.cu

main.o: main.cc vector.h
	cc -lpthread  -c main.cc

cleanall:
	rm *.o vector
	rm *.ptx *.ptxas _app_cuda_version*
	rm _cuobjdump_*

cleanrt:
	rm _cuobjdump_* *.txt
	rm *.ptx *.ptxas _app_cuda_version*