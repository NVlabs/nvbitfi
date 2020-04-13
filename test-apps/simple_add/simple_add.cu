/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>

#define THREADS_PER_WARP 32
#define WARPS_PER_CTA 32
#define DEFAULT_CTAS 10
#define DEFAULT_NREPS 10

// typedef unsigned long long int uint64_t;
// typedef int uint32_t;

__managed__ uint32_t global_sum;

///////////////////////////////////////////////////////////////////////////////
// The is the core function of this program. 
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_add(int nreps)
{
	int local_sum = 0; 

	for (int i=0; i<nreps; i++) {
		local_sum += 1;
	}

	atomicAdd(&global_sum, local_sum);
}

///////////////////////////////////////////////////////////////////////////////
// This is a wrapper to call the simple_add.
///////////////////////////////////////////////////////////////////////////////
void simple_add_wrapper(int ctas, int nreps)
{
	dim3 block(WARPS_PER_CTA * THREADS_PER_WARP, 1);
	dim3 grid(ctas, 1);

	cudaDeviceSynchronize(); 
	simple_add<<<grid,block,0>>>(nreps);
	cudaDeviceSynchronize(); 
	cudaError_t error = cudaGetLastError(); 
	if (error != cudaSuccess) {
		printf("Error: kernel failed %s\n", cudaGetErrorString(error));
	}
}

int main(int argc, char *argv[])
{
	setbuf(stdout, NULL); // Disable stdout buffering

	//Set the device
	int device = 0;
	cudaSetDevice(device);
	cudaDeviceProp cudaDevicePropForChoosing;
	cudaGetDeviceProperties(&cudaDevicePropForChoosing, device);

	printf("Device %d (%s) is being used\n", device, cudaDevicePropForChoosing.name);
	printf("memory: %.4f GB %s %d SMs x%d\n", cudaDevicePropForChoosing.totalGlobalMem/(1024.f*1024.f*1024.f), (cudaDevicePropForChoosing.ECCEnabled)?"ECC on":"ECC off", cudaDevicePropForChoosing.multiProcessorCount, cudaDevicePropForChoosing.clockRate );

	int nreps = DEFAULT_NREPS;
	int ctas = DEFAULT_CTAS;
	printf("#CTAs=%d, nreps=%d, threads/CTA=%d\n", ctas, nreps, THREADS_PER_WARP*WARPS_PER_CTA);

	global_sum = 0; // initialize the sum to 0

	// Call the main function now
	simple_add_wrapper(ctas, nreps);

	printf("global sum = %d \n", global_sum); 

	return 0;
}
