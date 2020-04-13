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

#include "utils/utils.h"
#include "arch.h"

// Global counters are incremented once per warp 
extern "C" __device__ __noinline__ void count_instrs(uint64_t pcounters, int index, int grp_index, int num_counters) {    
	// Optimization: Instead of all the threads in a warp performing atomicAdd,
	// let's count the number of active threads in a warp and let just one thread
	// (leader) in the warp perform the atomicAdd
	unsigned int active = __activemask();
	int leader = __ffs(active) - 1;

	uint64_t *counters = (uint64_t*)pcounters;
	if (threadIdx.x %32 == leader) { // Am I the leader thread
		int numActive = __popc(active);
		atomicAdd((unsigned long long *)&counters[index], numActive);
		atomicAdd((unsigned long long *)&counters[NUM_ISA_INSTRUCTIONS+grp_index], numActive);
		atomicAdd((unsigned long long *)&counters[num_counters-2], numActive*(grp_index != G_NODEST));
		atomicAdd((unsigned long long *)&counters[num_counters-1], numActive*(1 - ((grp_index == G_NODEST) || (grp_index == G_PR))));
	}
}
