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

#ifndef INJECTOR_H
#define INJECTOR_H

#define NUM_DEBUG_VALS 4
#define MAX_KNAME_SIZE 1000

typedef struct {
	// Parameters to inject an error (read from nvbitfi-injection-info.txt)
	bool areParamsReady;
	char kernelName[MAX_KNAME_SIZE];
	int32_t kernelCount;
	int32_t groupID; // instruction group (GroupType in arch.h)
	unsigned long long instID; // index of the dynamic instance among the injectable instances of the group
	float opIDSeed; // destination selection seed (0-1)
	uint32_t bitFlipModel; // bit-flip model
	float bitIDSeed; // bit selection seed (0-1)

	// Selection (written by the BEFORE call of the selected dynamic instance)
	bool selected;       // the selected instance has been reached
	int selPcOffset;     // its PC offset
	int selCtaFlat;      // its CTA (flattened blockIdx)
	int selTidInCta;     // its thread (flattened threadIdx)

	// Injection record (written by the AFTER call, after the instruction ran)
	int regType;          // RegType in arch.h: GPR, PR, UR, UP
	int regNo;            // register number (Rn, Pn, URn, UPn)
	unsigned int mask;    // XOR mask actually applied
	unsigned int beforeVal; // register value before injection (whole predicate register for PR/UP)
	unsigned int afterVal;  // register value after injection
	int opcode;           // opcode index (instTypeNames)
	int pcOffset;         // PC offset of the instruction in the function
	int tid;              // flattened global thread id
	int ctaFlat;          // flattened CTA id
	int smID;             // %smid of the SM that executed the injection
	int warpID;           // %warpid (hardware warp slot in the SM)
	int laneID;           // %laneid (0-31)
	bool errorInjected;

	int debug[NUM_DEBUG_VALS]; // [0]=1 if the instrumented kernel ran, [1]=expected GPR value, [2]=nvbit_read_reg value, [3]=reg
} inj_info_t;

#endif
