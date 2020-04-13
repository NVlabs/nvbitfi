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

#define NUM_DEBUG_VALS 17
#define MAX_KNAME_SIZE 1000

typedef struct {
	// Parameters to inject an error 
	bool areParamsReady;
	char kernelName[MAX_KNAME_SIZE]; 
	int32_t kernelCount;
	int32_t groupID; // arch state id
	unsigned long long instID; // injection inst id
	float opIDSeed; // injection operand id seed (random number between 0-1)
	uint32_t bitFlipModel; // bit-flip model 
	float bitIDSeed; // bit id seed (random number between 0-1)

	// The following are updated during/after error injection 
	unsigned int mask;
	unsigned int beforeVal;
	unsigned int afterVal;
	int regNo;
	int opcode;
	int pcOffset;
	int tid;
	bool errorInjected;

	int debug[NUM_DEBUG_VALS];
} inj_info_t; 
#endif
