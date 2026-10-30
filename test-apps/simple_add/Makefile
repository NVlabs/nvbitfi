TARGET = simple_add

# CUDA
NVCC = `which nvcc`
# sm_120 = Blackwell RTX / RTX PRO; override with e.g. make ARCH=89
ARCH ?= 120

all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -o simple_add -Xptxas -v -arch=sm_$(ARCH) simple_add.cu 

test:
	./simple_add 

golden:
	./simple_add >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ simple_add

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
