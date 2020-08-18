CC=g++
CFLAGS= --std=c++11 -I.

all: example

example: toojpeg_cuda.o


toojpeg_cuda.o: gpu.o toojpeg_cuda.cpp
	$(CC) -o toojpeg_cuda.o toojpeg_cuda.cpp $(CFLAGS)

gpu.o: gpu.cu
	nvcc -c --default-stream per-thread -Xcompiler "$(CFLAGS)" gpu.cu -o gpu.o 

clean:
	rm -f toojpeg_cuda.o gpu.o