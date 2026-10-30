/*
 * Copyright 2020, NVIDIA CORPORATION.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>
#include <signal.h>
#include <unistd.h>
#include <unordered_set>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

#include "globals.h"
#include "pf_injector.h"

int verbose;
__managed__ int verbose_device;
int limit = INT_MAX;

// injection parameters input filename: This file is created the the script
// that launched error injections
std::string injInputFilename = "nvbitfi-injection-info.txt";

pthread_mutex_t mutex;

__managed__ inj_info_t inj_info; 

void reset_inj_info() {
		inj_info.injInstType = 0; 
		inj_info.injSMID = 0; 
		inj_info.injLaneID = 0;
		inj_info.injMask = 0;
		inj_info.injNumActivations = 0;
		inj_info.errorInjected = false;
}

// for debugging 
void print_inj_info() {
		assert(fout.good());
		std::cout << "InstType=" << inj_info.injInstType << ", SMID=" << inj_info.injSMID<< ", LaneID=" << inj_info.injLaneID;
		std::cout << ", Mask=" << inj_info.injMask << std::endl;
}

// Parse error injection site info from a file. This should be done on host side.
void parse_params(std::string filename) {
		static bool parse_flag = false; // file will be parsed only once - performance enhancement
		if (!parse_flag) {
				parse_flag = true;
				reset_inj_info(); 

				std::ifstream ifs (filename.c_str(), std::ifstream::in);
				if (ifs.is_open()) {
						ifs >> inj_info.injSMID; 
						assert(inj_info.injSMID < 1000); // we don't have a 1000 SM system yet. 

						ifs >> inj_info.injLaneID; 
						assert(inj_info.injLaneID < 32); // Warp-size is 32 or less today. 

						ifs >> inj_info.injMask; 

						// instruction type: opcode name (e.g. FFMA, HMMA), "ALL", or numeric index
						// (0..NUM_ISA_INSTRUCTIONS-1 = one opcode of common/arch.h, NUM_ISA_INSTRUCTIONS = all)
						std::string instType;
						ifs >> instType;
						if (instType == "ALL") {
								inj_info.injInstType = NUM_ISA_INSTRUCTIONS;
						} else if (!instType.empty() && isdigit(instType[0])) {
								inj_info.injInstType = atoi(instType.c_str());
						} else {
								inj_info.injInstType = opcode_index(instType);
								if (inj_info.injInstType == UNKNOWN_OPCODE) {
										printf("pf_injector: opcode %s is not in common/arch.h\n", instType.c_str());
										assert(false);
								}
						}
						assert(inj_info.injInstType <= NUM_ISA_INSTRUCTIONS);

				} else {
						printf(" File %s does not exist!", filename.c_str());
						printf(" This file should contain enough information about the fault site to perform a permanent error injection run: ");
						printf("(1) SM ID, (2) Lane ID (within a warp), (3) 32-bit mask (as int32), (4) Instruction type (opcode name such as FFMA, ALL, or opcode index in common/arch.h). \n"); 
						assert(false);
				}
				ifs.close();

				if (verbose) {
						print_inj_info();
				}
		}
}

void update_verbose() {
		static bool update_flag = false; // update it only once - performance enhancement
		if (!update_flag) {
			update_flag = true;
			cudaDeviceSynchronize();
			verbose_device = verbose;
			cudaDeviceSynchronize();
		}
}

int get_maxregs(CUcontext ctx, CUfunction func) {
		func_config_t config;
		nvbit_get_func_config(ctx, func, &config);
		return config.num_registers;
}

void INThandler(int sig) {
		signal(sig, SIG_IGN); // disable Ctrl-C

		fout << ":::NVBit-inject-error; ERROR FAIL Detected Singal SIGKILL;";
		fout << " injNumActivations: " << inj_info.injNumActivations << ":::";
		fout.flush();
		exit(-1);
}


/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
// DO NOT USE UVM (__managed__) variables in this function
void nvbit_at_init() {
		/* just make sure all managed variables are allocated on GPU */
		setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC","1",1);

		/* we get some environment variables that are going to be use to selectively
		 * instrument (within a interval of kernel indexes and instructions). By
		 * default we instrument everything. */
		if (getenv("TOOL_VERBOSE")) {
				verbose = atoi(getenv("TOOL_VERBOSE"));
		} else {
				verbose = 0;
		}

		if (getenv("INPUT_INJECTION_INFO")) {
				injInputFilename = getenv("INPUT_INJECTION_INFO");
		}
		if (getenv("OUTPUT_INJECTION_LOG")) {
				injOutputFilename = getenv("OUTPUT_INJECTION_LOG");
		}
		if (getenv("INSTRUMENTATION_LIMIT")) {
				limit = atoi(getenv("INSTRUMENTATION_LIMIT"));
		} 

		// GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool (1, 2, 3,..)");


		signal(SIGINT, INThandler); // install Ctrl-C handler

		open_output_file(injOutputFilename);
		if (verbose) printf("nvbit_at_init:end\n");
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;


void instrument_function_if_needed(CUcontext ctx, CUfunction func) {

		parse_params(injInputFilename);  // injParams are updated based on injection seed file
		update_verbose();

		/* Get related functions of the kernel (device function that can be
		 * called by the kernel) */
		std::vector<CUfunction> related_functions =
				nvbit_get_related_functions(ctx, func);

		/* add kernel itself to the related function vector */
		related_functions.push_back(func);

		/* iterate on function */
		for (auto f : related_functions) {
				/* "recording" function was instrumented, if set insertion failed
				 * we have already encountered this function */
				if (!already_instrumented.insert(f).second) {
						continue;
				}

				std::string kname = removeSpaces(nvbit_get_func_name(ctx,f));
				/* Get the vector of instruction composing the loaded CUFunction "func" */
				const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

				int maxregs = get_maxregs(ctx, f);
				assert(fout.good());
				fout << "Inspecting: " << kname << ";num_static_instrs: " << instrs.size() << ";maxregs: " << maxregs << "(" << maxregs << ")" << std::endl;
				for(auto i: instrs)  {
						/* destination registers from NVBit's decoded operands (common/dests.h) */
						StaticInstInfo si = analyze_instr(i);
						if (si.opcode == UNKNOWN_OPCODE) warn_unknown_opcode(i);
						bool opcodeMatch = ((uint32_t)si.opcode == inj_info.injInstType || inj_info.injInstType == NUM_ISA_INSTRUCTIONS);
						// The permanent fault model corrupts the first destination GPR (per-lane datapath).
						// Uniform-datapath instructions write uniform registers, not GPRs: not targeted.
						if (opcodeMatch && si.group != G_UNIFORM && si.dests.num_gpr > 0) {
								if (verbose) {
										printf("instruction selected for instrumentation: 0x%04x %s dests=%s\n", i->getOffset(), i->getSass(), si.dests.str().c_str());
								}
								nvbit_insert_call(i, "inject_error", IPOINT_AFTER);
								nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_info);
								nvbit_add_call_arg_const_val64(i, (uint64_t)&verbose_device);
								nvbit_add_call_arg_const_val32(i, si.dests.gpr_base); // destination GPR register number
								nvbit_add_call_arg_reg_val(i, si.dests.gpr_base); // destination GPR register val
								nvbit_add_call_arg_const_val32(i, si.dests.num_gpr); // number of destination GPR registers
								nvbit_add_call_arg_const_val32(i, maxregs); // max regs used by the inst info
								nvbit_add_call_arg_guard_pred_val(i); // predicated-out instances are not corrupted
						}
				}
		}
}

/* This call-back is triggered every time a CUDA event is encountered.
 * Here, we identify CUDA kernel launch events and reset the "counter" before
 * th kernel is launched, and print the counter after the kernel has completed
 * (we make sure it has completed by using cudaDeviceSynchronize()). To
 * selectively run either the original or instrumented kernel we used
 * nvbit_enable_instrumented() before launching the kernel. */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
				const char *name, void *params, CUresult *pStatus) {
		/* Identify all the possible CUDA launch events */
		if (is_kernel_launch(cbid)) {

				/* CUfunction of the launched kernel (also for cuLaunchKernelEx) */
				CUfunction func = get_launched_function(cbid, params);

				if(!is_exit) {
						pthread_mutex_lock(&mutex);
						if (kernel_id < limit) {
							instrument_function_if_needed(ctx, func);
							// cudaDeviceSynchronize();

							nvbit_enable_instrumented(ctx, func, true); // run the instrumented version
							// cudaDeviceSynchronize();
						} else {
							nvbit_enable_instrumented(ctx, func, false); // do not use the instrumented version
						}

				}  else {
						if (kernel_id < limit) {
								if (verbose) printf("is_exit\n"); 
								cudaDeviceSynchronize();

								cudaError_t le = cudaGetLastError();

								std::string kname = removeSpaces(nvbit_get_func_name(ctx,func));
								int num_ctas = get_num_ctas(cbid, params);
								assert(fout.good());
								fout << "Injection data; " ;
								fout << "index: " << kernel_id << ";"; 
								fout << "kernel_name: " << kname << ";"; 
								fout << "ctas: " << num_ctas << ";";
								fout << "selected SM: " << inj_info.injSMID <<  ";";
								fout << "selected Lane: " << inj_info.injLaneID <<  ";";
								fout << "selected Mask: " << inj_info.injMask <<  ";";
								fout << "selected InstType: " << inj_info.injInstType <<  ";";
								fout << "injNumActivations: " << inj_info.injNumActivations << std::endl;

								if ( cudaSuccess != le ) {
										assert(fout.good());
										fout << "ERROR FAIL in kernel execution (" << cudaGetErrorString(le) << "); " <<std::endl;
										exit(1); // let's exit early 
								}

								if (verbose) printf("\n index: %d; kernel_name: %s; \n", kernel_id, kname.c_str());
								kernel_id++; // always increment kernel_id on kernel exit

						}
						// Always release the lock taken at launch entry. Originally the unlock
						// was inside "if (kernel_id < limit)", so the first launch past
						// INSTRUMENTATION_LIMIT never released it and the next launch deadlocked.
						pthread_mutex_unlock(&mutex);
				}
		}
}

void nvbit_at_term() { } // nothing to do here. 
