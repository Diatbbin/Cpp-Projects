# Makefile

NVCC        := /usr/local/cuda/bin/nvcc
COMMON_SRCS := src/common.cpp
CUDA_SRCS   := src/kernel.cu

OUTPUT        := scanner
DEBUG-OUTPUT  := scanner-debug

.DEFAULT_TARGET: all

all: $(OUTPUT) $(DEBUG-OUTPUT)

$(OUTPUT): $(CUDA_SRCS) $(COMMON_SRCS)
	$(NVCC) -std c++17 -O3 -lineinfo -arch=native -o $@ $^

$(DEBUG-OUTPUT): $(CUDA_SRCS) $(COMMON_SRCS)
	$(NVCC) -G -std c++17 -O3 -arch=native -o $@ $^

clean:
	rm -f $(OUTPUT)

cleanlogs:
	rm -f logs/*.slurmlog
