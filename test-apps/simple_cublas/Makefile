TARGET = simple_cublas
#USE_TENSOR_CORES=0
#USE_TENSOR_CORES=1

# CUDA
CUDA_PATH=/usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

SMS = 70
$(foreach sm,$(SMS), $(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

FLAGS = -Xptxas -v $(ARCH) -L$(CUDA_PATH)/lib64 -lcublas

ifeq ($(USE_TENSOR_CORES), 1)
FLAGS+= -DUSE_TENSOR_CORES=1
endif

all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -o $(TARGET) $(FLAGS) $(TARGET).cu

test:
	./$(TARGET)

golden:
	./$(TARGET) >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ $(TARGET)

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
