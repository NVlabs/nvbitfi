# Instruction profiler (dynamic instruction counts per opcode and group).
# Build settings are shared with the other tools in ../common/build.mk.
# Usage: make [NVBIT_PATH=...] [ARCH=sm_120] [FAST_APPROXIMATE_PROFILE=1]
#
# Profiling is often slow. FAST_APPROXIMATE_PROFILE=1 profiles each static
# kernel only once and reuses that profile for all its later invocations
# (assumes every invocation of a kernel behaves like the first one).
NVBIT_TOOL = profiler.so
ifeq ($(FAST_APPROXIMATE_PROFILE),1)
TOOL_FLAGS = -DSKIP_PROFILED_KERNELS
endif
include ../common/build.mk
