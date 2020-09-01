CC=g++
CFLAGS=--std=c++11 -I. -fpermissive
CUFLAGS=-lcudart -pthread

all: example

example: toojpeg_cuda.o examples/example.cpp
	$(CC) -o examples/example_cuda examples/example.cpp toojpeg_cuda.o gpu.o utility.o $(CFLAGS) $(CUFLAGS)

toojpeg_cuda.o: gpu.o toojpeg_cuda.cpp utility.o
	$(CC) -c -o toojpeg_cuda.o toojpeg_cuda.cpp $(CFLAGS) $(CUFLAGS) gpu.o

toojpeg_cpu.o: toojpeg_cpu.cpp utility.o
	$(CC) -c -o toojpeg_cpu.o toojpeg_cpu.cpp $(CFLAGS)

gpu.o: gpu.cu
	nvcc -c --default-stream per-thread -arch=sm_35 -Xcompiler "$(CFLAGS) $(CUFLAGS)" gpu.cu -o gpu.o  

utility.o: utility.cpp
	$(CC) -c -o utility.o utility.cpp $(CFLAGS)

comparison: toojpeg_cuda.o toojpeg_cpu.o examples/comparison.cpp utility.o
	$(CC) -o examples/comparison examples/comparison.cpp toojpeg_cuda.o toojpeg_cpu.o utility.o gpu.o $(CFLAGS) $(CUFLAGS)


clean:
	rm -f toojpeg_cuda.o gpu.o examples/example_cuda examples/comparison toojpeg_cpu.o

clean-gpu:
	rm -f gpu.o

clean-main:
	rm -f toojpeg_cuda.o examples/example_cuda