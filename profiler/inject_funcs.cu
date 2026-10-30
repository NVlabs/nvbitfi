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

// Called BEFORE every SASS instruction. Counts dynamic instances per opcode
// and the INJECTABLE instances per group (eligibility decided on the host,
// see eligibility_flags() in common/globals.h). The injector uses exactly the
// same rules, so an injection index drawn from this profile points to the
// same dynamic instance in the injection run.
//
// One leader thread per warp performs the atomics (warp-aggregated).
// Regular instructions count one per thread whose guard predicate is true;
// uniform-datapath instructions execute once per warp and count 1.
extern "C" __device__ __noinline__ void count_instrs(uint64_t pcounters, int opcode, int grp, int flags, int predicate) {
	uint64_t *counters = (uint64_t*)pcounters;

	const unsigned active_mask = __activemask();
	const unsigned predicate_mask = __ballot_sync(active_mask, predicate);
	if (predicate_mask == 0) return;

	const int leader = __ffs(predicate_mask) - 1;
	if (get_laneid() != leader) return;

	const unsigned long long n = (flags & FLAG_UNIFORM) ? 1ULL : (unsigned long long)__popc(predicate_mask);

	atomicAdd((unsigned long long *)&counters[opcode], n);
	if (flags & FLAG_ELIG_OWN)
		atomicAdd((unsigned long long *)&counters[NUM_OPCODE_COUNTERS+grp], n);
	if (flags & FLAG_ELIG_GPPR)
		atomicAdd((unsigned long long *)&counters[NUM_COUNTERS-2], n);
	if (flags & FLAG_ELIG_GP)
		atomicAdd((unsigned long long *)&counters[NUM_COUNTERS-1], n);
}
