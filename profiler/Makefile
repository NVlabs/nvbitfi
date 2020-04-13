NVCC=nvcc -ccbin=`which gcc` -D_FORCE_INLINES


NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

NVBIT_PATH=../../../core
INCLUDES=-I$(NVBIT_PATH)
INCLUDES+=-I../common

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES=$(wildcard *.cu)
OBJECTS=$(SOURCES:.cu=.o)
ARCH=35

# Profiling is often slow. Using an approximate profile can speed it up by
# orders of magnitude.  Set FAST_APPROXIMATE_PROFILE to SKIP_PROFILED_KERNELS
# to use an approximate profile.  When SKIP_PROFILED_KERNELS flag is enabled,
# the static kernels are profiled only once.  It assumes that the profile of
# different invocations of a static kernel are the same (same as the first
# dynamic invocation of that kernel).
# 
# FAST_APPROXIMATE_PROFILE=-DSKIP_PROFILED_KERNELS 
FAST_APPROXIMATE_PROFILE=


NVBIT_TOOL=profiler.so

all: $(NVBIT_TOOL)

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=sm_$(ARCH) -O3 $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

%.o: %.cu
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=sm_$(ARCH) -O3 -Xcompiler -fPIC $(FAST_APPROXIMATE_PROFILE) $< -o $@

inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) -maxrregcount=16 -Xptxas -astoolspatch --keep-device-functions -arch=sm_$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f *.so *.o

