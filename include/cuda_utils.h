/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

// A simple timer class
// use CUDA's high-resolution timers when possible
#include <cuda_runtime_api.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>


// #define TIMING

/********************/
/* CUDA ERROR CHECK */
/********************/
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


static void cuda_safe_call(cudaError_t error, const std::string& message = "")
{
    if (error)
        throw thrust::system_error(error, thrust::cuda_category(), message);
}

struct timer
{
    cudaEvent_t start;
    cudaEvent_t end;

    timer(void)
    {
#ifdef TIMING
        cuda_safe_call(cudaEventCreate(&start));
        cuda_safe_call(cudaEventCreate(&end));
        restart();
#endif

    }

    ~timer(void)
    {
#ifdef TIMING
        cuda_safe_call(cudaEventDestroy(start));
        cuda_safe_call(cudaEventDestroy(end));
#endif
    }

    void restart(void)
    {
#ifdef TIMING
        cuda_safe_call(cudaEventRecord(start, 0));
#endif
    }
    // In ms
    double elapsed(void)
    {
#ifdef TIMING
        cuda_safe_call(cudaEventRecord(end, 0));
        cuda_safe_call(cudaEventSynchronize(end));

        float ms_elapsed;
        cuda_safe_call(cudaEventElapsedTime(&ms_elapsed, start, end));
        return ms_elapsed;
#else
        return 0.0;
#endif
    }

    double epsilon(void)
    {
        return 0.5e-6;
    }
};
