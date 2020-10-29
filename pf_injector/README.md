# Experimental feature: Simulate ISA-level permanent hardware errors

User should select a SM, hardware lane, bit-mask, and opcode for a permanent error injection run. Errors will be injected into all dynamic instances of the specified opcode in the specified lane and SM.

pf_injector.so expects nvbitfi-injection-info.txt file to be present in the current working directory.  This file should contain four things (one per line): 
* SM ID: 0-Max SMs in the GPU being used
* Lane ID: 0-31
* Mask: uint32 number used to inject error into the destination register (corrupted value = Mask XOR original value)
* opcode ID: 0-171 (see enum InstructionType in common/arch.h for the mapping). 171: all opcodes.

Example usage: `LD_PRELOAD=<path-to-so>/pf_injector.so <path-to-workload>/simple_add; cat nvbitfi-injection-log-temp.txt`.  A sample nvbitfi-injection-info.txt is provided to test the tool on the provided simple_add test-app. 

## Notes

Set `TOOL_VERBOSE=1` to print more information about the injection run in the console output.

Ensure that the correct `nvcc` version is being used by adding appropriate directory to PATH. `nvcc` is typically installed in `/usr/local/cuda/bin/`.

Set `NVDISASM` to point to the correct disassembler. This is typically provided in `/usr/local/cuda/bin/`.

Set `INPUT_INJECTION_INFO` for custom input file name and path. Default is nvbitfi-injection-info.txt in the directory where the application is being launched. 

Set `OUTPUT_INJECTION_LOG` for custom output file name and path. Default is nvbitfi-injection-log-temp.txt in the directory where the application is being launched. 

Instrumenting all the kernels for permanent error injection can make the application too slow (and non-operational for some). One can limit the number of dynamic kernels that will be instrumented and observe errors by setting `INSTRUMENTATION_LIMIT` to 1 or 10, for example.
