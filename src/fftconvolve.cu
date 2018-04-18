/**
    fftconvolve.cu
    Compute complex-complex and real-real FFT convolutions on the GPU

    @author: Konstantinos Iliakis (konstantinos.iliakis@cern.ch)
    @version: 1.0 17/10/2017
*/


#include "cuda_utils.h"
#include <cufft.h>
#include <cuda.h>
#include <stdio.h>
#include <thrust/device_vector.h>

using namespace std;

extern "C" {
    std::unordered_map<std::string, std::vector<double>> timerMap;

    void reset_timer()
    {
        timerMap.clear();
    }

    void report_timer()
    {
        double total_time = 0;
        if (timerMap.find("total_time") != timerMap.end())
            total_time = std::accumulate(timerMap["total_time"].begin() + 1,
                                         timerMap["total_time"].end(), 0.0);
        for (auto &kv : timerMap) {
            auto v = kv.second;
            double sum = std::accumulate(v.begin() + 1, v.end(), 0.0);
            printf("[%s] Calls: %d, Average time: %.3lf ms, Global Percentage: %.2lf %\n",
                   kv.first.c_str(), v.size(), sum / v.size(), 100.0 * sum / total_time);
        }
    }



    struct complexMultiplier
    {
        double scale;
        complexMultiplier(double scale): scale(scale) {};

        __host__ __device__
        cufftDoubleComplex operator() (const cufftDoubleComplex &v1,
                                       const cufftDoubleComplex &v2) const
        {
            cufftDoubleComplex res;
            res.x = (v1.x * v2.x - v1.y * v2.y) * scale;
            res.y = (v1.x * v2.y + v1.y * v2.x) * scale;
            return res;
        }
    };

    /**
        Creates an FFT Plan if it has not been yet initialized

        @plan: Pointer to the plan that will be created/initialized
        @size: Size of the FFT for which this plan will be used
        @type: Type of the FFT
        @batch: Number of FFTs of the specified size that will be computed together.

    */
    void create_plan(cufftHandle *plan, size_t size, cufftType type, int batch = 1)
    {
        size_t workSize;
        int ret = cufftGetSize(*plan, &workSize);
        if (ret == CUFFT_INVALID_PLAN) {
            if (cufftPlan1d(plan, size, type, batch) != CUFFT_SUCCESS) {
                fprintf(stderr, "CUFFT error: Plan creation failed");
            }
        }
    }


    /**
        Computes the FFT convolution of two complex signals

        @signal: The first signal. This is a pointer to host(CPU) memory
        @signalSize: The signal size
        @kernel: The second signal, also called kernel. This is a pointer to
                 host(CPU) memory
        @kernelSize: The kernel size
        @result: Pointer to host(CPU) memory that contains the convolution result.
                 Sufficient memory ((singalSize + kernelSize -1) * sizeof(cufftDoubleComplex))
                 has to be allocated before calling the function.
        @d_in: Pointer to GPU memory used by the function. The size of the memory region
                has to be at least 2 * (signalSize + kernelSize - 1)
        @fwplan: An integer handle used to store the forward FFT plan.
        @bwplan: An integer handle used to store the backward FFT plan.
    */
    void convolve_complex(cufftDoubleComplex * signal, int signalSize,
                          cufftDoubleComplex * kernel, int kernelSize,
                          cufftDoubleComplex * result,
                          cufftDoubleComplex * d_in,
                          cufftHandle *fwplan,
                          cufftHandle *bwplan)
    {


        // timer timer, globalTimer;
        // globalTimer.restart();
        size_t real_size = signalSize + kernelSize - 1;

        // timer.restart();
        cudaMemset(d_in, 0, 2 * real_size * sizeof(cufftDoubleComplex));
        // timerMap["memset"].push_back(timer.elapsed());

        // timer.restart();
        cudaMemcpy(d_in, signal, signalSize * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_in + real_size, kernel, kernelSize * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
        // timerMap["memcpy"].push_back(timer.elapsed());

        // timer.restart();
        create_plan(fwplan, real_size, CUFFT_Z2Z, 2);
        create_plan(bwplan, real_size, CUFFT_Z2Z, 1);
        // timerMap["create_plans"].push_back(timer.elapsed());

        // timer.restart();
        cufftExecZ2Z(*fwplan, d_in, d_in, CUFFT_FORWARD);
        // timerMap["forward"].push_back(timer.elapsed());

        // timer.restart();
        thrust::device_ptr<cufftDoubleComplex> a(d_in);
        thrust::transform(a, a + real_size, a + real_size, a,
                          complexMultiplier(1.0 / real_size));
        // timerMap["multiply"].push_back(timer.elapsed());

        // timer.restart();
        cufftExecZ2Z(*bwplan, d_in, d_in, CUFFT_INVERSE);
        // timerMap["backward"].push_back(timer.elapsed());

        // timer.restart();
        cudaMemcpy(result, d_in, real_size * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
        // timerMap["copy_back"].push_back(timer.elapsed());
        // timerMap["total_time"].push_back(globalTimer.elapsed());
    }

    /**
        Computes the FFT convolution of two real signals

        @signal: The first signal. This is a pointer to host(CPU) memory
        @signalSize: The signal size
        @kernel: The second signal, also called kernel. This is a pointer to
                 host(CPU) memory
        @kernelSize: The kernel size
        @result: Pointer to host(CPU) memory where the convolution result will be copied.
                 Sufficient memory ((signalSize + kernelSize - 1)*sizeof(double))
                 has to be allocated before calling the function.
        @fwplan: An integer handle used to store the forward FFT plan.
        @bwplan: An integer handle used to store the backward FFT plan.
    */
    void convolve_real(double * signal, int signalSize,
                       double * kernel, int kernelSize,
                       double * result,
                       cufftHandle *fwplan,
                       cufftHandle *bwplan)
    {
        cufftDoubleComplex *d_out;
        double *d_in;

        size_t real_size = signalSize + kernelSize - 1;
        size_t complex_size = real_size / 2 + 1;

        cudaMalloc((void**)&d_in, 2 * real_size * sizeof(double));
        cudaMalloc((void**)&d_out, 2 * complex_size * sizeof(cufftDoubleComplex));

        cudaMemset(d_in, 0, 2 * real_size * sizeof(double));
        cudaMemcpy(d_in, signal, signalSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_in + real_size, kernel, kernelSize * sizeof(double), cudaMemcpyHostToDevice);

        create_plan(fwplan, real_size, CUFFT_D2Z, 2);
        create_plan(bwplan, real_size, CUFFT_Z2D);

        cufftExecD2Z(*fwplan, d_in, d_out);

        thrust::device_ptr<cufftDoubleComplex> a(d_out);
        thrust::transform(a, a + complex_size, a + complex_size, a,
                          complexMultiplier(1.0 / real_size));

        cufftExecZ2D(*bwplan, d_out, d_in);

        cudaMemcpy(result, d_in, real_size * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);
    }


    /**
        Computes the FFT convolution of two real signals

        @signal: The first signal. This is a pointer to GPU memory
        @signalSize: The signal size
        @kernel: The second signal, also called kernel. This is a pointer to
                 GPU memory
        @kernelSize: The kernel size
        @result: Pointer to GPU memory where the convolution result will be copied.
                 Sufficient memory ((signalSize + kernelSize - 1)*sizeof(double))
                 has to be allocated before calling the function.
        @fwplan: An integer handle used to store the forward FFT plan.
        @bwplan: An integer handle used to store the backward FFT plan.
    */
    void convolve_real_no_memcpy(double * signal, int signalSize,
                                 double * kernel, int kernelSize,
                                 double * result,
                                 cufftHandle *fwplan,
                                 cufftHandle *bwplan)
    {
        cufftDoubleComplex *d_out;
        size_t real_size = signalSize + kernelSize - 1;
        size_t complex_size = real_size % 2 == 1 ?
                              ((real_size + 1) / 2) : (real_size / 2 + 1);

        cudaMalloc((void**)&d_out, 2 * complex_size * sizeof(cufftDoubleComplex));

        create_plan(fwplan, real_size, CUFFT_D2Z);
        create_plan(bwplan, real_size, CUFFT_Z2D);

        cufftExecD2Z(*fwplan, signal, d_out);
        cufftExecD2Z(*fwplan, kernel, d_out + complex_size);

        thrust::device_ptr<cufftDoubleComplex> a(d_out);
        thrust::transform(a, a + complex_size, a + complex_size, a,
                          complexMultiplier(1.0 / real_size));

        cufftExecZ2D(*bwplan, d_out, result);

        cudaFree(d_out);
    }

}
