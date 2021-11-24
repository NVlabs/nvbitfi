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

#include "nvbit_reg_rw.h"
#include "utils/utils.h"
#include "injector.h"
#include "arch.h"

// flatten thread id
__inline__ __device__ int get_flat_tid() {
	int tid_b = threadIdx.x + (blockDim.x * (threadIdx.y + (threadIdx.z * blockDim.y))); // thread id within a block
	int bid = blockIdx.x + (gridDim.x * (blockIdx.y + (blockIdx.z * gridDim.y))); // block id 
	int tid = tid_b + (bid * blockDim.x * blockDim.y * blockDim.z);
	return tid;
}

// Get bit-mask for error injection. Old value will be XORed with this mask later to inject the error. 
__inline__ 
__device__ unsigned int get_mask(uint32_t bitFlipModel, float bitIDSeed, unsigned int oldVal) {
	return (bitFlipModel == FLIP_SINGLE_BIT) * ((unsigned int)1<<(int)(32*bitIDSeed)) + 
		(bitFlipModel == FLIP_TWO_BITS) * ((unsigned int)3<<(int)(31*bitIDSeed)) + 
		(bitFlipModel == RANDOM_VALUE)  * (((unsigned int)-1) * bitIDSeed) +
		(bitFlipModel == ZERO_VALUE) * oldVal;
}

extern "C" __device__ __noinline__ void inject_error(uint64_t piinfo, uint64_t pcounters, uint64_t pverbose_device, 
			int offset, int index, int grp_index,  int predicate, int destGPRNum, int regval, 
			int numDestGPRs, int destPRNum1, int destPRNum2, int maxRegs) {

	inj_info_t* inj_info = (inj_info_t*)piinfo; 
	uint32_t verbose_device = *((uint32_t *)pverbose_device);
 	uint64_t * counters = (uint64_t *)pcounters;

 	if (verbose_device)
 		inj_info->debug[NUM_DEBUG_VALS-1] = 1;
	// if error is injected, return
 	if (inj_info->errorInjected)
 		return;
	// Check if this is the kernel of interest. 
	// Sanity check - we shouldn't be here if this is not the kernel of interest. 
 	if (!inj_info->areParamsReady)
 		return; // This is not the selected kernel. No need to proceed.
 
	// Proceed only if the instruction is not predicated out, i.e., proceed only if the the predicate value is 1. 
	// We do not want to inject an error into the instruction that's predicated out.
	if (predicate == 0)
		return;

 	unsigned long long currCounter1 = atomicAdd((unsigned long long *)&counters[NUM_ISA_INSTRUCTIONS+grp_index], 1);
 	unsigned long long currCounter2 = atomicAdd((unsigned long long *)&counters[NUM_COUNTERS-2], (grp_index != G_NODEST));
 	unsigned long long currCounter3 = atomicAdd((unsigned long long *)&counters[NUM_COUNTERS-1], 1 - ((grp_index == G_NODEST) || (grp_index == G_PR)));
 
	uint32_t igid = inj_info->groupID;
 	if ((igid == G_GPPR && grp_index == G_NODEST) ||  // not a GPPR instruction
 			(igid == G_GP && ((grp_index == G_NODEST) || (grp_index == G_PR))) || // not the G_GP instructiop 
 			(igid < G_NODEST &&  grp_index != igid)) // this is not the instruction from the selected group
 		return; // This is not the selected intruction group 
  
	bool injectFlag = false;
 	switch (igid) {
 		case G_FP32: // inject into one of the dest reg 
 		case G_FP64: // inject into one of the regs written by the inst
 		case G_LD: // inject into one of the regs written by the inst
 		case G_PR: // inject into pr register
 			if(inj_info->instID == currCounter1) {
 				injectFlag = true;
 			}
 			break; 
 		case G_GPPR: // inject into one of the GPR or PR destination register
 			if(inj_info->instID == currCounter2) {
 				injectFlag = true;
 			}
 			break;
 		case G_GP: // inject into one of the GPR destination register
 			if(inj_info->instID == currCounter3) {
 				injectFlag = true;
 			}
 			break;
 		case G_NODEST: // do nothing
 		default:  break;
 	}
  
   	if (verbose_device && injectFlag) 
		printf("inj_info->instID=%ld, %ld, %ld, %ld\n", inj_info->instID, currCounter1, currCounter2, currCounter3);

	if (injectFlag) {
		// assert(0 == 10);
		if (verbose_device)
			printf("offset=0x%x, igid:%d, destGPRNum=%d, grp_index=%d\n", offset, igid, destGPRNum, grp_index); 
		// We need to randomly select one register from numDestGPRs + (destPRNum1 != -1) + (destPRNum2 != -1)
		int totalDest = numDestGPRs + (destPRNum1 != -1) + (destPRNum2 != -1);
		assert(totalDest > 0);
		int injDestID = totalDest*inj_info->opIDSeed;
		if (injDestID < numDestGPRs) {
			if (destGPRNum != -1) {
 				// for debugging
 				inj_info->debug[0] = maxRegs; 
 				inj_info->debug[1] = destGPRNum; 
 				inj_info->debug[2] = regval; 
 				inj_info->debug[3] = nvbit_read_reg((uint64_t)destGPRNum); // read the register value 
 				inj_info->debug[4] = nvbit_read_reg((uint64_t)0); // read the register value 
 				inj_info->debug[5] = nvbit_read_reg((uint64_t)1); // read the register value 
 				inj_info->debug[6] = nvbit_read_reg((uint64_t)2); // read the register value 
 				inj_info->debug[7] = nvbit_read_reg((uint64_t)3); // read the register value 
 				inj_info->debug[8] = nvbit_read_reg((uint64_t)4); // read the register value 
 				inj_info->debug[9] = nvbit_read_reg((uint64_t)5); // read the register value 
 				inj_info->debug[10] = nvbit_read_reg((uint64_t)6); // read the register value 
 				atomicAdd(&(inj_info->debug[11]),1); 
 				inj_info->debug[12] = index; 
 				inj_info->debug[13] = offset; 
 
 				inj_info->regNo = destGPRNum+injDestID; // record the register number
 				inj_info->beforeVal = nvbit_read_reg((uint64_t)inj_info->regNo); // read the register value
 				inj_info->mask = get_mask(inj_info->bitFlipModel, inj_info->bitIDSeed, inj_info->beforeVal); // bit-mask for error injection
 				if (DUMMY) { // no error is injected
 					inj_info->afterVal = inj_info->beforeVal;
 				} else {
 					inj_info->afterVal = inj_info->beforeVal ^ inj_info->mask; 
 					nvbit_write_reg((uint64_t)inj_info->regNo, inj_info->afterVal);
 				}
 				inj_info->opcode = index;  // record the opcode where the injection is performed
 				inj_info->pcOffset = offset;  // record the pc where the injection is performed (offset from the beginning of the function)
 				inj_info->tid = get_flat_tid(); // record the thread ID where the injection is performed
 				inj_info->errorInjected = true; // perf optimization
 				assert(inj_info->debug[12] == inj_info->opcode);
 				assert(inj_info->debug[13] == inj_info->pcOffset);
 				if (verbose_device) 
 					printf("done here\n"); 
			} else {
				assert(0 == 2); 
			}
		} else { 
 
 			// printf(":::ERROR Error injection into predicate registers is not supported by NVBit (as of April 10, 2020);"); 
 
 			// // assert(0 == 4); 
 			// if (destPRNum1 != -1 && destPRNum2 != -1) { // we want to inject into destPRNum1/destPRNum2 if it's not -1
 			//   inj_info->regNo = injDestID == numDestGPRs ? destPRNum1 : destPRNum2;
 			// } else if (destPRNum1 != -1) {
 			//   inj_info->regNo = destPRNum1;
 			// } else if (destPRNum2 != -1) {
 			//   inj_info->regNo = destPRNum2;
 			// }
 			// inj_info->beforeVal = read_predicate_reg(maxRegs, spillAddr, inj_info->regNo); // read the register value before writing
 			// if (DUMMY) {
 			//   inj_info->afterVal = inj_info->beforeVal;
 			// } else {
 			//   inj_info->afterVal = flip_predicate_reg(maxRegs, spillAddr, inj_info->regNo); // inject the error and record the value after injection
 			// }
 			// inj_info->opcode = index;  // record the opcode where the injection is performed
 			// inj_info->pcOffset = offset;  // record the pc where the injection is performed (offset from the beginning of the function)
 			// inj_info->tid = get_flat_tid(); // record the thread ID where the injection is performed
 			// inj_info->errorInjected = true; // perf optimization
		}
 	}
}

