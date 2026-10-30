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
 *
 * NVBitFI_for_ODC (2026): injection split in two calls.
 *
 *  inject_select (IPOINT_BEFORE) counts the injectable dynamic instances of
 *    the selected group exactly like the profiler (same point, same guard
 *    predicate, same eligibility) and marks the selected (thread, PC).
 *    Counting AFTER the instruction, as the original did, miscounts
 *    instructions that overwrite their own guard predicate (@P0 ISETP P0, ...)
 *    and never runs for EXIT/branches.
 *  inject_error (IPOINT_AFTER) corrupts one destination register of the
 *    selected instance: GPR, predicate (P0-P6), uniform register or uniform
 *    predicate, using the NVBit 1.8 register read/write device API.
 */

#include <stdint.h>
#include <stdio.h>

#include "nvbit_reg_rw.h"
#include "utils/utils.h"
#include "injector.h"
#include "arch.h"

__inline__ __device__ int get_flat_tid_in_cta() {
	return threadIdx.x + (blockDim.x * (threadIdx.y + (threadIdx.z * blockDim.y)));
}

__inline__ __device__ int get_flat_cta() {
	return blockIdx.x + (gridDim.x * (blockIdx.y + (blockIdx.z * gridDim.y)));
}

__inline__ __device__ int get_flat_tid() {
	return get_flat_tid_in_cta() + get_flat_cta() * (blockDim.x * blockDim.y * blockDim.z);
}

// get_laneid(), get_smid(), get_warpid() come from NVBit's utils/utils.h

// XOR mask for a 32-bit register. The old value will be XORed with it.
__inline__ __device__ unsigned int get_mask(uint32_t bitFlipModel, float bitIDSeed, unsigned int oldVal) {
	int b1 = min((int)(32 * bitIDSeed), 31);
	int b2 = min((int)(31 * bitIDSeed), 30);
	return (bitFlipModel == FLIP_SINGLE_BIT) * (1u << b1) +
		(bitFlipModel == FLIP_TWO_BITS) * (3u << b2) +
		(bitFlipModel == RANDOM_VALUE)  * (unsigned int)(((unsigned int)-1) * bitIDSeed) +
		(bitFlipModel == ZERO_VALUE) * oldVal;
}

// New value of a 1-bit predicate. Flip models invert it (a 1-bit register
// has no "two adjacent bits"), RANDOM_VALUE draws 0/1, ZERO_VALUE clears it.
__inline__ __device__ unsigned int corrupt_pred_bit(uint32_t bitFlipModel, float bitIDSeed, unsigned int oldBit) {
	if (bitFlipModel == RANDOM_VALUE) return bitIDSeed >= 0.5f ? 1u : 0u;
	if (bitFlipModel == ZERO_VALUE) return 0u;
	return oldBit ^ 1u;
}

// k-th set bit (0-based) of mask
__inline__ __device__ int nth_set_bit(uint32_t mask, int k) {
	for (int b = 0; b < 32; b++) {
		if (mask >> b & 1) { if (k == 0) return b; k--; }
	}
	return -1;
}

// ---------------------------------------------------------------------------
// BEFORE: count and select
// ---------------------------------------------------------------------------
extern "C" __device__ __noinline__ void inject_select(uint64_t piinfo, uint64_t pcounters,
		int offset, int opcode, int grp, int flags, int predicate) {
	inj_info_t* inj_info = (inj_info_t*)piinfo;
	uint64_t *counters = (uint64_t *)pcounters;

	if (!inj_info->areParamsReady || inj_info->selected) return; // not the kernel of interest / already selected

	// Same eligibility rules as the profiler (count_instrs)
	const int igid = inj_info->groupID;
	bool elig;
	int ctr;
	if (igid == G_GPPR) { elig = flags & FLAG_ELIG_GPPR; ctr = NUM_COUNTERS - 2; }
	else if (igid == G_GP) { elig = flags & FLAG_ELIG_GP; ctr = NUM_COUNTERS - 1; }
	else { elig = (grp == igid) && (flags & FLAG_ELIG_OWN); ctr = NUM_OPCODE_COUNTERS + igid; }
	if (!elig) return; // static property: the whole warp returns together

	const unsigned active_mask = __activemask();
	const unsigned pred_mask = __ballot_sync(active_mask, predicate);
	if (!predicate) return; // predicated-out threads do not execute the instruction

	const int lane = get_laneid();
	const int leader = __ffs(pred_mask) - 1;
	const bool uniform = flags & FLAG_UNIFORM;
	const unsigned long long n = uniform ? 1ULL : (unsigned long long)__popc(pred_mask);

	// warp-aggregated counting: the leader reserves n indices, every thread
	// takes base + its rank among the predicate-true lanes
	unsigned long long base = 0;
	if (lane == leader) base = atomicAdd((unsigned long long *)&counters[ctr], n);
	base = __shfl_sync(pred_mask, base, leader);

	if (uniform && lane != leader) return; // a uniform instruction is one instance per warp
	const unsigned long long myIdx = base + (uniform ? 0 : __popc(pred_mask & ((1u << lane) - 1)));

	if (myIdx == inj_info->instID) {
		inj_info->selPcOffset = offset;
		inj_info->selCtaFlat = get_flat_cta();
		inj_info->selTidInCta = get_flat_tid_in_cta();
		inj_info->opcode = opcode;
		__threadfence();
		inj_info->selected = true;
	}
}

// ---------------------------------------------------------------------------
// AFTER: corrupt one destination of the selected instance
// ---------------------------------------------------------------------------
extern "C" __device__ __noinline__ void inject_error(uint64_t piinfo, uint64_t pverbose_device,
		int offset, uint32_t gprRange, uint32_t prMask, uint32_t urRange, uint32_t upMask, int regval) {
	inj_info_t* inj_info = (inj_info_t*)piinfo;
	uint32_t verbose_device = *((uint32_t *)pverbose_device);

	if (!inj_info->selected || inj_info->errorInjected) return;
	if (offset != inj_info->selPcOffset) return;
	if (get_flat_cta() != inj_info->selCtaFlat || get_flat_tid_in_cta() != inj_info->selTidInCta) return;

	const int igid = inj_info->groupID;
	const int gBase = gprRange >> 8, nG = gprRange & 0xff;
	const int uBase = urRange >> 8, nU = urRange & 0xff;
	int nP = __popc(prMask), nUP = __popc(upMask);
	// candidate destinations for the selected group
	int candG = nG, candP = nP, candU = nU, candUP = nUP;
	if (igid == G_GP) { candP = 0; candU = 0; candUP = 0; }
	else if (igid == G_GPPR) { candU = 0; candUP = 0; }
	const int total = candG + candP + candU + candUP;

	inj_info->pcOffset = offset;
	inj_info->tid = get_flat_tid();
	inj_info->ctaFlat = get_flat_cta();
	inj_info->smID = get_smid();
	inj_info->warpID = get_warpid();
	inj_info->laneID = get_laneid();
	inj_info->debug[0] = 1;

	if (total == 0) { // should not happen: eligibility guarantees a destination
		inj_info->regType = -1;
		inj_info->errorInjected = true;
		return;
	}

	int d = min((int)(total * inj_info->opIDSeed), total - 1);

	if (d < candG) {                                   // ---- GPR ----
		int r = gBase + d;
		if (verbose_device) {
			inj_info->debug[1] = regval;                // value passed by NVBit for the first dest
			inj_info->debug[2] = nvbit_read_reg((uint64_t)gBase);
			inj_info->debug[3] = gBase;
		}
		unsigned int before = nvbit_read_reg((uint64_t)r);
		unsigned int mask = get_mask(inj_info->bitFlipModel, inj_info->bitIDSeed, before);
		inj_info->regType = REG_GPR;
		inj_info->regNo = r;
		inj_info->beforeVal = before;
		inj_info->mask = mask;
		inj_info->afterVal = DUMMY ? before : (before ^ mask);
		if (!DUMMY) nvbit_write_reg((uint64_t)r, inj_info->afterVal);
	} else if ((d -= candG) < candP) {                 // ---- predicate P0-P6 ----
		int p = nth_set_bit(prMask, d);
		unsigned int before = (unsigned int)nvbit_read_pred_reg();
		unsigned int bit = corrupt_pred_bit(inj_info->bitFlipModel, inj_info->bitIDSeed, (before >> p) & 1u);
		unsigned int after = (before & ~(1u << p)) | (bit << p);
		inj_info->regType = REG_PR;
		inj_info->regNo = p;
		inj_info->beforeVal = before;
		inj_info->mask = before ^ after;
		inj_info->afterVal = DUMMY ? before : after;
		if (!DUMMY) nvbit_write_pred_reg((int32_t)after);
	} else if ((d -= candP) < candU) {                 // ---- uniform register ----
		int r = uBase + d;
		unsigned int before = nvbit_read_ureg((uint64_t)r);
		unsigned int mask = get_mask(inj_info->bitFlipModel, inj_info->bitIDSeed, before);
		inj_info->regType = REG_UR;
		inj_info->regNo = r;
		inj_info->beforeVal = before;
		inj_info->mask = mask;
		inj_info->afterVal = DUMMY ? before : (before ^ mask);
		if (!DUMMY) nvbit_write_ureg((uint64_t)r, inj_info->afterVal);
	} else {                                           // ---- uniform predicate UP0-UP6 ----
		d -= candU;
		int p = nth_set_bit(upMask, d);
		unsigned int before = (unsigned int)nvbit_read_upred_reg();
		unsigned int bit = corrupt_pred_bit(inj_info->bitFlipModel, inj_info->bitIDSeed, (before >> p) & 1u);
		unsigned int after = (before & ~(1u << p)) | (bit << p);
		inj_info->regType = REG_UP;
		inj_info->regNo = p;
		inj_info->beforeVal = before;
		inj_info->mask = before ^ after;
		inj_info->afterVal = DUMMY ? before : after;
		if (!DUMMY) nvbit_write_upred_reg((int32_t)after);
	}
	__threadfence();
	inj_info->errorInjected = true;
	if (verbose_device)
		printf("NVBitFI: injected regType=%d reg=%d before=0x%x after=0x%x pc=0x%x tid=%d sm=%d\n",
				inj_info->regType, inj_info->regNo, inj_info->beforeVal, inj_info->afterVal, offset, inj_info->tid, inj_info->smID);
}
