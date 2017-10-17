# simlpe test to use the cuda shared library
import ctypes
import numpy as np
import os
import sys
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

# from pyprof.timing import timeit, report

_libfftconvolve = ctypes.cdll.LoadLibrary(
    os.path.dirname(os.path.abspath(__file__)) + '/../../src/libfftconvolve.so')

_libfftconvolve.convolve_real.restype = None
_libfftconvolve.convolve_real.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p
]

_libfftconvolve.convolve_real_no_memcpy.restype = None
_libfftconvolve.convolve_real_no_memcpy.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p
]


# @timeit()
def convolve_real_gpu_v1(signal, kernel, output):
    _libfftconvolve.convolve_real(signal.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(len(signal)),
                                  kernel.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(len(kernel)),
                                  output.ctypes.data_as(ctypes.c_void_p),
                                  convolve_real_gpu_v1.fwplan.ctypes.data_as(
                                  ctypes.c_void_p),
                                  convolve_real_gpu_v1.bwplan.ctypes.data_as(
                                  ctypes.c_void_p))


def convolve_real_gpu_v2(signal, kernel, output):
    # Need to pass the correct signal and kernel sizes
    _libfftconvolve.convolve_real_no_memcpy(int(signal.gpudata),
                                            ctypes.c_int((len(signal)+1) // 2),
                                            int(kernel.gpudata),
                                            ctypes.c_int((len(kernel)+1)//2),
                                            int(output.gpudata),
                                            convolve_real_gpu_v2.fwplan.ctypes.data_as(
                                            ctypes.c_void_p),
                                            convolve_real_gpu_v2.bwplan.ctypes.data_as(
                                            ctypes.c_void_p))


# @timeit()
def convolve_real_cpu_v1(signal, kernel):
    from scipy.signal import fftconvolve
    return fftconvolve(signal, kernel)


# @timeit()
def convolve_real_cpu_v2(signal, kernel):
    from numpy import convolve
    return convolve(signal, kernel)


if __name__ == "__main__":
    n_turns = 100
    n_points = int(10000)

    if(len(sys.argv) > 1):
        n_turns = int(sys.argv[1])
    if(len(sys.argv) > 2):
        n_points = int(sys.argv[2])

    # Initializations
    np.random.seed(0)
    signal = np.random.randn(n_points)
    signal = np.array(signal, dtype=np.float64)
    kernel = np.random.randn(n_points)
    kernel = np.array(kernel, dtype=np.float64)

    real_size = len(signal) + len(kernel) - 1
    complex_size = real_size // 2 + 1

    result_gpu_v1 = np.zeros(len(signal) + len(kernel) - 1, dtype=np.float64)
    result_cpu_v1 = np.zeros(len(signal) + len(kernel) - 1, dtype=np.float64)
    result_cpu_v2 = np.zeros(len(signal) + len(kernel) - 1, dtype=np.float64)

    convolve_real_gpu_v1.fwplan = np.zeros(1, np.uint32)
    convolve_real_gpu_v1.bwplan = np.zeros(1, np.uint32)

    convolve_real_gpu_v2.fwplan = np.zeros(1, np.uint32)
    convolve_real_gpu_v2.bwplan = np.zeros(1, np.uint32)

    d_signal = gpuarray.to_gpu(np.concatenate(
        (signal, np.zeros(len(kernel) - 1))))

    d_kernel = gpuarray.to_gpu(np.concatenate((
        kernel, np.zeros(len(signal) - 1))))

    d_result = gpuarray.to_gpu(result_cpu_v1)

    # Computations
    for i in range(n_turns):
        convolve_real_gpu_v1(signal, kernel, result_gpu_v1)
    print('End of convolve_real_gpu_v1')

    for i in range(n_turns):
        convolve_real_gpu_v2(d_signal, d_kernel, d_result)
    print('End of convolve_real_gpu_v2')

    for i in range(n_turns):
        result_cpu_v1 = convolve_real_cpu_v1(signal, kernel)
    print('End of convolve_real_cpu_v1')

    for i in range(n_turns):
        result_cpu_v2 = convolve_real_cpu_v2(signal, kernel)
    print('End of convolve_real_cpu_v2')

    # Correctness tests
    if np.allclose(result_cpu_v1, result_gpu_v1, atol=1e-05):
        print('CPU-v1 and GPU-v1 agree')
    else:
        print('CPU-v1 and GPU-v1 disagree')

    if np.allclose(result_cpu_v1, d_result.get(), atol=1e-05):
        print('CPU-v1 and GPU-v2 agree')
    else:
        print('CPU-v1 and GPU-v2 disagree')

    if np.allclose(result_cpu_v1, result_cpu_v2, atol=1e-05):
        print('CPU-v2 and CPU-v1 agree')
    else:
        print('CPU-v2 and CPU-v1 disagree')

    print('Done!')
