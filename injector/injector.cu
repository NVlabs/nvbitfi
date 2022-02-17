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
#include "injector.h"

int verbose;
__managed__ int verbose_device;

// injection parameters input filename: This file is created the the script
// that launched error injections
std::string injInputFilename = "nvbitfi-injection-info.txt";

pthread_mutex_t mutex;

__managed__ inj_info_t inj_info; 

void reset_inj_info() {
	inj_info.areParamsReady = false;
	inj_info.kernelName[0] = '\0';
	inj_info.kernelCount = -1;
	inj_info.groupID = 0; // arch state id 
	inj_info.instID = 0; // instruction id 
	inj_info.opIDSeed = 0; // destination id seed (float, 0-1)
	inj_info.bitIDSeed = 0; // bit location seed (float, 0-1)
	inj_info.bitFlipModel = 0; // fault model: single bit flip, all bit flip, random value
	inj_info.mask = 0;
	inj_info.beforeVal = 0;
	inj_info.afterVal = 0;
	inj_info.regNo = -1;
	inj_info.opcode = NOP;
	inj_info.pcOffset = 0;
	inj_info.tid = -1;
	inj_info.errorInjected = false;
	for (int i=0; i<NUM_DEBUG_VALS; i++) {
		inj_info.debug[i] = -1;
	}
}

void write_inj_info() {
	assert(fout.good());
	for (int i=0; i<NUM_INST_GROUPS; i++) {
		fout << " grp " << i << ": " << counters[NUM_ISA_INSTRUCTIONS+i];
	}
	fout << std::endl;
	fout << "mask: 0x" << std::hex << inj_info.mask << std::endl;
	fout << "beforeVal: 0x" << inj_info.beforeVal  << ";";
	fout << "afterVal: 0x" << inj_info.afterVal << std::endl;
	fout << "regNo: " << std::dec << inj_info.regNo << std::endl;
	fout << "opcode: " << instTypeNames[inj_info.opcode] << std::endl;
	fout << "pcOffset: 0x" << std::hex << inj_info.pcOffset  << std::endl;
	fout << "tid: " << std::dec << inj_info.tid<< std::endl; 
}

// for debugging 
void print_inj_info() {
	assert(fout.good());
	fout << "kernelName=" << inj_info.kernelName << std::endl;
	fout << "kernelCount=" << inj_info.kernelCount << std::endl;
	fout << "groupID=" << inj_info.groupID << std::endl; 
	fout << "bitFlipModel=" << inj_info.bitFlipModel  << std::endl;
	fout << "instID=" << inj_info.instID << std::endl;
	fout << "opIDSeed=" << inj_info.opIDSeed << std::endl;
	fout << "bitIDSeed=" << inj_info.bitIDSeed << std::endl;
}

// Parse error injection site info from a file. This should be done on host side.
void parse_params(std::string filename) {
	static bool parse_flag = false; // file will be parsed only once - performance enhancement
	if (!parse_flag) {
		parse_flag = true;
		reset_inj_info(); 

		std::ifstream ifs (filename.c_str(), std::ifstream::in);
		if (ifs.is_open()) {
			ifs >> inj_info.groupID; // arch state id 
			assert(inj_info.groupID >=0 && inj_info.groupID < NUM_INST_GROUPS); // ensure that the value is in the expected range

			ifs >> inj_info.bitFlipModel; // fault model: single bit flip, all bit flip, random value
			assert(inj_info.bitFlipModel < NUM_BFM_TYPES); // ensure that the value is in the expected range

			ifs >> inj_info.kernelName;
			ifs >> inj_info.kernelCount;
			ifs >> inj_info.instID; // instruction id 

			ifs >> inj_info.opIDSeed; // destination id seed (float, 0-1 for inst injections and 0-256 for reg)
			assert(inj_info.opIDSeed >=0 && inj_info.opIDSeed < 1.01); // ensure that the value is in the expected range

			ifs >> inj_info.bitIDSeed; // bit location seed (float, 0-1)
			assert(inj_info.bitIDSeed >= 0 && inj_info.bitIDSeed < 1.01); // ensure that the value is in the expected range
		} else {
			printf(" File %s does not exist!", filename.c_str());
			printf(" This file should contain enough information about the fault site to perform an error injection run: ");
			printf("(1) arch state id, (2) bit flip model, (3) kernel name, (4) kernel count, (5) instruction id, (6) seed to select destination id, (7) sed to select bit location.\n");
			assert(false);
		}
		ifs.close();

		if (verbose) 
			print_inj_info();
	}
}

int get_maxregs(CUfunction func) {
	int maxregs = -1;
	cuFuncGetAttribute(&maxregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
	return maxregs;
}

// custom signal handler such that we don't miss the injection information.
void INThandler(int sig) {
	signal(sig, SIG_IGN); // disable Ctrl-C

	fout << "ERROR FAIL Detected Signal SIGKILL\n";
	write_inj_info();
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
	// GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool (1, 2, 3,..)");

	initInstTypeNameMap();

	signal(SIGINT, INThandler); // install Ctrl-C handler

	open_output_file(injOutputFilename);
	if (verbose) 
		printf("nvbit_at_init:end\n");
}


/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {

	parse_params(injInputFilename.c_str());  // injParams are updated based on injection seed file
	cudaDeviceSynchronize();
	verbose_device = verbose;
	cudaDeviceSynchronize();

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
		if(strcmp(inj_info.kernelName, kname.c_str()) == 0) { // this is the kernel selected for injection 
			assert(fout.good()); // ensure that the log file is good.

			/* Get the vector of instruction composing the loaded CUFunction "f" */
			const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

			/* If verbose we print function name and number of" static" instructions
			 */
			if (verbose) {
				printf("inspecting %s - num instrs %ld\n",
						nvbit_get_func_name(ctx, f), instrs.size());
			}

			int maxregs = get_maxregs(f);
			fout << "inspecting: " << kname << "\nnum_static_instrs: " << instrs.size() << "\nmaxregs: " << maxregs << "(" << maxregs << ")" << std::endl;

			/* We iterate on the vector of instruction */
			for (auto i : instrs) {
				std::string opcode = i->getOpcode(); 
				std::string instType = extractInstType(opcode); 
				// printf("extracted instType: %s\n", instType.c_str());
				// printf("index of instType: %d\n", instTypeNameMap[instType]);

				// Tokenize the instruction 
				std::vector<std::string> tokens;
				std::string buf; // a buffer string
				std::stringstream ss(i->getSass()); // Insert the string into a stream
				while (ss >> buf)
					tokens.push_back(buf);

				int destGPRNum = -1;
				int numDestGPRs = 0;
				int destPRNum1 = -1;
				int destPRNum2 = -1;

				int instGrpNum = getOpGroupNum(instTypeNameMap[instType]); ;
				if (tokens.size() > 0 && instGrpNum != G_NODEST) { // an actual instruction that writes to either a GPR or PR register
					if (verbose) 
						printf("num tokens = %ld ", tokens.size());
					int start = 1; // first token is opcode string
					if (tokens[0].find('@') != std::string::npos) { // predicated instruction, ignore first token
						start = 2; // first token is predicate and 2nd token is opcode
					}

					// Parse the first operand - this is the first destination
					int regnum1 = -1;
					int regnum2 = -1;
					int regtype = extractRegNo(tokens[start], regnum1);
					if (regtype == 0) { // GPR reg
						destGPRNum = regnum1;
						numDestGPRs = (instGrpNum == G_FP64) ? 2 : 1;

						int sz = extractSize(opcode); 
						if (sz != 0) { // for LD, IMMA, HMMA
							numDestGPRs = sz/32; 
						}

						int regtype2 = extractRegNo(tokens[start+1], regnum2);
						// the following is probably not possible in Pascal ISA
						if (regtype2 == 1) { // PR reg, it looks like this instruction has two destination registers
							destPRNum1  = regnum2;
						}
					} 
					if (regtype == 1) {
						destPRNum1  = regnum1;

						if(instGrpNum != G_PR) { // this is not a PR-only instruction.
							int regtype2 = extractRegNo(tokens[start+1], regnum2);
							if (regtype2 == 0) { // a GPR reg, it looks like this instruction has two destination registers
								destGPRNum = regnum2;
								numDestGPRs = (instGrpNum == G_FP64) ? 2 : 1;
							}
						} else { // check if the 2nd reg is a PR dest
							if (tokens.size() > 5) { // this seems like the instruction that has 2 PR destinations 
								int regtype2 = extractRegNo(tokens[start+1], regnum2);
								if (regtype2 == 1) { // a PR reg, it looks like this instruction has two destination registers
									destPRNum2  = regnum2;
								}
							}
						}
					}
					if (verbose) 
						printf("offset = 0x%x, opcode_info=%d, instType=%s, opcode=%s, numDestGPRs=%d, destGPRNum=%d, destPRNum1=%d, destPRNum2=%d, instruction: %s\n", i->getOffset(), instTypeNameMap[instType], instType.c_str(), opcode.c_str(), numDestGPRs, destGPRNum, destPRNum1, destPRNum2, i->getSass());
				}

				nvbit_insert_call(i, "inject_error", IPOINT_AFTER);
				nvbit_add_call_arg_const_val64(i, (uint64_t)&inj_info);
				nvbit_add_call_arg_const_val64(i, (uint64_t)counters);
				nvbit_add_call_arg_const_val64(i, (uint64_t)&verbose_device);

				nvbit_add_call_arg_const_val32(i, i->getOffset()); // offset (for pc) info
				nvbit_add_call_arg_const_val32(i, instTypeNameMap[instType]); // opcode info
				nvbit_add_call_arg_const_val32(i, instGrpNum); // instruction group info

				nvbit_add_call_arg_guard_pred_val(i); // predicate value

				nvbit_add_call_arg_const_val32(i, destGPRNum); // destination GPR register number
				if (destGPRNum != -1) {
					nvbit_add_call_arg_reg_val(i, destGPRNum); // destination GPR register val
				} else {
					nvbit_add_call_arg_const_val32(i, (unsigned int)-1); // destination GPR register val 
				}
				nvbit_add_call_arg_const_val32(i, numDestGPRs); // number of destination GPR registers

				if (isGPInst(instGrpNum) && inj_info.groupID == G_GP) { // PR register numbers should be -1, if the injection model is G_GP. This way we will never inject errors into them
					nvbit_add_call_arg_const_val32(i, (unsigned int)-1); // first destination PR register number 
					nvbit_add_call_arg_const_val32(i, (unsigned int)-1); // second destination PR register number 
				} else {
					nvbit_add_call_arg_const_val32(i, destPRNum1); // first destination PR register number 
					nvbit_add_call_arg_const_val32(i, destPRNum2); // second destination PR register number 
				}

				nvbit_add_call_arg_const_val32(i, maxregs); // max regs used by the inst info
			}
		} else {
			const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
			if (verbose)
				printf(":::NVBit-inject-error; NOT inspecting: %s; %d, %d, num_static_instrs: %ld; maxregs: %d:::", kname.c_str(), kernel_id, inj_info.kernelCount, instrs.size(), get_maxregs(f));
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
	if (cbid == API_CUDA_cuLaunch ||
			cbid == API_CUDA_cuLaunchKernel_ptsz ||
			cbid == API_CUDA_cuLaunchGrid ||
			cbid == API_CUDA_cuLaunchGridAsync || 
			cbid == API_CUDA_cuLaunchKernel) {

		/* cast params to cuLaunch_params since if we are here we know these are
		 * the right parameters type */
		cuLaunch_params * p = (cuLaunch_params *) params;

		if(!is_exit) {
			pthread_mutex_lock(&mutex);
            		instrument_function_if_needed(ctx, p->f);
			init_counters();
			cudaDeviceSynchronize();
			parse_params(injInputFilename);  // injParams are updated based on injection seed file

			// print_inj_info();
			inj_info.errorInjected = false;
			inj_info.areParamsReady = (inj_info.kernelCount== kernel_id); // areParamsReady = true for the selected kernel 
			if (verbose) inj_info.debug[NUM_DEBUG_VALS-1] = -1; // set debug flag to check whether the the instrumented kernel was executed 
			if (verbose) printf("setting areParamsReady=%d, inj_info.kernelCount=%d, kernel_id=%d\n", inj_info.areParamsReady, inj_info.kernelCount, kernel_id); 
			cudaDeviceSynchronize();

			nvbit_enable_instrumented(ctx, p->f, inj_info.areParamsReady); // should we run the un-instrumented code? 
			// nvbit_enable_instrumented(ctx, p->f, false); // for debugging
			cudaDeviceSynchronize();
		}  else {
			if (verbose) printf("is_exit\n"); 
			cudaDeviceSynchronize();

			cudaError_t le = cudaGetLastError();
			if ( cudaSuccess != le ) {
				assert(fout.good());
				std::cout << "ERROR FAIL in kernel execution (" << cudaGetErrorString(le) << "); ";
				fout << "ERROR FAIL in kernel execution (" << cudaGetErrorString(le) << "); ";
				fout.flush();
				exit(1); // let's exit early because no error was injected
			}

			std::string kname = removeSpaces(nvbit_get_func_name(ctx,p->f));
			if (inj_info.areParamsReady) {
				inj_info.areParamsReady = false;
				int num_ctas = 0;
				if ( cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {
					cuLaunchKernel_params * p2 = (cuLaunchKernel_params*) params;
					num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
				}
				assert(fout.good());
				fout << "Injection data" << std::endl;
				fout << "index: " << kernel_id << std::endl;
				fout << "kernel_name: " << kname  << std::endl;
				fout << "ctas: " << num_ctas << std::endl;
				fout << "instrs: " << get_inst_count() << std::endl; 

				write_inj_info(); 

				if (inj_info.opcode == NOP) {
					fout << "Error not injected\n";
				}

				if (verbose != 0 && inj_info.debug[2] != inj_info.debug[3]) { // sanity check
					fout << "ERROR FAIL in kernel execution; Expected reg value doesn't match; \n";
					fout << "maxRegs: " << inj_info.debug[0] << ", destGPRNum: " << inj_info.debug[1] << ", expected_val: " 
						<< std::hex << inj_info.debug[2] << ", myval: " <<  inj_info.debug[3] << std::dec << "\n"; 
					fout << std::endl;
					std::cout << "NVBit-inject-error; ERROR FAIL in kernel execution; Expected reg value doesn't match; \n";
					std::cout << "maxRegs: " << inj_info.debug[0] << ", destGPRNum: " << inj_info.debug[1] << ", expected_val: " 
						<< std::hex << inj_info.debug[2] << ", myval: " <<  inj_info.debug[3] << std::dec << "\n"; 
					for (int x=4; x<10; x++) {
						std::cout << "debug[" << x << "]: " << std::hex << inj_info.debug[x] << "\n";
					}
					std::cout << "debug[11]: " << std::hex << inj_info.debug[11] << "\n";
					std::cout << "debug[12]: " << inj_info.debug[12] << " " << instTypeNames[inj_info.debug[12]]<< "\n";
					std::cout << "debug[13]: " << inj_info.debug[13] << "\n"; 
					std::cout << "debug[14]: " << std::hex <<  inj_info.debug[14] << "\n"; 
					assert(inj_info.debug[2] == inj_info.debug[3]);
					// printf("\nmaxRegs: %d, destGPRNum: %d, expected_val: %x, myval: %x, myval@-1: %x, myval@+1: %x, myval with maxRegs+1: %x, myval with maxRegs-1: %x\n", 
					// inj_info.debug[0], inj_info.debug[1], inj_info.debug[2], inj_info.debug[3], inj_info.debug[4], inj_info.debug[5], inj_info.debug[6], inj_info.debug[7]);
				}
				fout.flush();
			}
			if (verbose) printf("\n index: %d; kernel_name: %s; used_instrumented=%d; \n", kernel_id, kname.c_str(), inj_info.debug[NUM_DEBUG_VALS-1]);
			kernel_id++; // always increment kernel_id on kernel exit

			cudaDeviceSynchronize();
			pthread_mutex_unlock(&mutex);
		}
	}
}
void nvbit_at_term() { } // nothing to do here. 
