CC=g++
CFLAGS= --std=c++11 -I. -lcudart -pthread

all: example

example: toojpeg_cuda.o examples/example.cpp
	$(CC) -o examples/example_cuda examples/example.cpp toojpeg_cuda.o gpu.o $(CFLAGS)


toojpeg_cuda.o: gpu.o toojpeg_cuda.cpp
	$(CC) -c -o toojpeg_cuda.o toojpeg_cuda.cpp $(CFLAGS) gpu.o

gpu.o: gpu.cu
	nvcc -c --default-stream per-thread -arch=sm_35 -Xcompiler "$(CFLAGS)" gpu.cu -o gpu.o  

clean:
	rm -f toojpeg_cuda.o gpu.o examples/example_cuda

clean-gpu:
	rm -f gpu.o

clean-main:
	rm -f toojpeg_cuda.o examples/example_cuda