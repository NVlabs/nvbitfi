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
#include <vector>
#include <map>
#include <unordered_set>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

#include "globals.h"

/* provide some __device__ functions */
#include "utils/utils.h"

pthread_mutex_t mutex;

int verbose = 0;

std::string line_buffer; 
bool enable_instrumentation = false;

std::string get_profiled_details(std::string kname) {
#ifdef SKIP_PROFILED_KERNELS
	std::ifstream  infile;
	std::string line;
	infile.open(injOutputFilename.c_str(), std::ifstream::in);
	if (infile.good()) {
		while (std::getline(infile, line)) {
			if (line.find(kname) != std::string::npos) { // found the string
				infile.close();
				return line;
			}
		}
		infile.close();
	}
#endif
	return "";
}


/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
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

	initInstTypeNameMap();
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
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

		/* Get the vector of instruction composing the loaded CUFunction "f" */
		const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

		/* If verbose we print function name and number of" static" instructions
		 */
		if (verbose) {
			printf("inspecting %s - num instrs %ld\n",
					nvbit_get_func_name(ctx, f), instrs.size());
		}

		/* We iterate on the vector of instruction */
		for (auto i : instrs) {
			if (verbose==2) { 
				printf("begin..\n");
			}
			// printf("instruction: %s\n", i.sass.c_str());
			std::string opcode = i->getOpcode(); 
			std::string instType = extractInstType(opcode); 

			if (verbose) { 
				i->print();
				printf("extracted instType: %s\n", instType.c_str());
				printf("index of instType: %d\n", instTypeNameMap[instType]);
			}

			nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
			nvbit_add_call_arg_const_val64(i, (uint64_t)counters);
			nvbit_add_call_arg_const_val32(i, instTypeNameMap[instType]);
			nvbit_add_call_arg_const_val32(i, getOpGroupNum(instTypeNameMap[instType]));
			nvbit_add_call_arg_guard_pred_val(i); //pass predicate value 
			nvbit_add_call_arg_const_val32(i, NUM_COUNTERS);
			if (verbose==2) { 
				printf("end..\n");
			}
		}
	}
}

/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids  which are defined in tools_cuda_api_meta.h.
 * This call back is triggered bith at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char * name, void * params, CUresult * pStatus) {
	if (cbid == API_CUDA_cuLaunch ||
			cbid == API_CUDA_cuLaunchKernel_ptsz ||
			cbid == API_CUDA_cuLaunchGrid ||
			cbid == API_CUDA_cuLaunchGridAsync || 
			cbid == API_CUDA_cuLaunchKernel) {

		cuLaunch_params * p = (cuLaunch_params *) params;

		if(!is_exit) {
			/* if we are entering in a kernel launch:
			 * 1. Lock the mutex to prevent multiple kernels to run concurrently
			 * (overriding the counter) in case the user application does that
			 * 2. Reset counters 
			 * 3. Instrument the function if needed
			 * 4. Select if we want to run the instrumented or original 
			 * version of the kernel */


			pthread_mutex_lock(&mutex);
			instrument_function_if_needed(ctx, p->f);

			init_counters();
			open_output_file(injOutputFilename);

			// This is an approximation that provides >10x speedup. Here we assume that all the dynamic kernels will behave similarly.
			std::string kname = removeSpaces(nvbit_get_func_name(ctx, p->f)).c_str();
			line_buffer = get_profiled_details(kname); 
			enable_instrumentation = (line_buffer.compare("") == 0); // if the kernel is already profiled, we will approximate the new profile to be same as the first one
			nvbit_enable_instrumented(ctx, p->f, enable_instrumentation); // should we run the un-instrumented code? true means skip instrumentation 
		}  else {
			cudaDeviceSynchronize();
			if(cudaGetLastError() != cudaSuccess) {
				printf("NVBit-igprofile; ERROR FAIL in kernel execution!!\n");
				exit(1);
			}
			int num_ctas = 0;
			if ( cbid == API_CUDA_cuLaunchKernel_ptsz ||
					cbid == API_CUDA_cuLaunchKernel) {
				cuLaunchKernel_params * p2 = (cuLaunchKernel_params*) params;
				num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
			}
			assert(fout.good());
			std::string kname = removeSpaces(nvbit_get_func_name(ctx, p->f)).c_str();
			// std::cout << "; kernel_name: " << kname << " " << nvbit_get_func_name(ctx, p->f) << "\n";
			fout << "NVBit-igprofile; index: " << kernel_id++ << "; kernel_name: " << kname 
				<< "; ctas: " << num_ctas;
			if (enable_instrumentation) {
				fout << "; instrs: " << get_inst_count(true) << ";" ;
				for (int i=0; i<NUM_COUNTERS; i++) {
					if (i < NUM_ISA_INSTRUCTIONS) {
						fout << " " << instTypeNames[i] << ": " << counters[i] << ",";
					} else {
						fout << " " << instGrouptNames[i-NUM_ISA_INSTRUCTIONS] << ": " << counters[i] << ",";
					}
				}
			} else {
				int pos = line_buffer.find("; instrs: "); 
				fout << line_buffer.substr(pos); // simply record the previously obtained counts
			}
			fout << "\n";
			fout.flush();
			pthread_mutex_unlock(&mutex);
		}
	}
}

void nvbit_at_term() {} // nothing to do here. 
