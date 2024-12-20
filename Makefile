ifeq (x$(CUDA_PATH),x)
	CUDA_PATH=/usr/local/cuda
endif

NVCC=$(CUDA_PATH)/bin/nvcc
NVCCOPT=\
                -O3 \
                --compiler-options -fno-strict-aliasing \
                --compiler-options -fopenmp \
                --compiler-options -Wall \
                -DUNIX \
                --generate-code arch=compute_80,code=sm_80 \
                --generate-code arch=compute_90,code=sm_90 \
		--extra-device-vectorization \
		--restrict \
		--ptxas-options='--allow-expensive-optimizations true' \
		--ptxas-options='--register-usage-level 0' \
		--maxrregcount=127 \
		--dopt=on \
		--no-compress \
		--prec-div=true --prec-sqrt=true --fmad=true
NVCCOPTmin=\
                -O3 \
                --compiler-options -fno-strict-aliasing \
                --compiler-options -fopenmp \
                --compiler-options -Wall \
                -DUNIX \
                --generate-code arch=compute_80,code=sm_80 \
                --generate-code arch=compute_90,code=sm_90 \
		--extra-device-vectorization \
		--restrict \
		--ptxas-options='--allow-expensive-optimizations true' \
		--ptxas-options='--register-usage-level 0' \
		--maxrregcount=127 \
		--dopt=on \
		--no-compress \
		--prec-div=true --prec-sqrt=true --fmad=true

OBJS = main.o eigen_GPU_check.o
OBJSS = $(OBJS) eigen_GPU_batch.o
LIBS = libeigenGbatch.a
LIBOPT = -leigenGbatch

all: a.out $(LIBS)
a.out : $(OBJSS) $(LIBS)
	$(NVCC) -o $@ $(OBJSS) -L./ -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcusolver -lcublas -lm -lgomp
	cp a.out a.out-cuda

main.o: main.cpp
	$(NVCC) -c -o $@ $< -I$(CUDA_PATH)/include -DPRINT_DIAGNOSTIC=0 --compiler-options -fopenmp
libeigenGbatch.a: eigen_GPU_batch.o
	ar cr libeigenGbatch.a $<
	ranlib libeigenGbatch.a
eigen_GPU_batch.o: eigen_GPU_batch.cu
	$(NVCC) -c -o $@ $(NVCCOPT) $<
	$(NVCC) $(NVCCOPTmin) $<
eigen_GPU_check.o: eigen_GPU_check.cu
	$(NVCC) -c -o $@ $(NVCCOPT) $<

clean:
	-\rm a.out a.out-* *.o *.cu_o *.ptx lib*.a

