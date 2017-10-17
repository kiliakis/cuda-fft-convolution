# simlpe test to use the cuda shared library
import ctypes
import numpy as np
import os
import sys
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

from pyprof.timing import timeit, report
import scipy.signal

_libfftconvolve = ctypes.cdll.LoadLibrary(
    os.path.dirname(os.path.abspath(__file__)) + '/../../src/libfftconvolve.so')

_libfftconvolve.fftconvolve.restype = None
_libfftconvolve.fftconvolve.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p
]

_libfftconvolve.fftconvolve_batch.restype = None
_libfftconvolve.fftconvolve_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p
]


_libfftconvolve.fftconvolve_fast.restype = None
_libfftconvolve.fftconvolve_fast.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p
]


@timeit()
def fftconvolve_gpu(signal, kernel, output):

    _libfftconvolve.fftconvolve(signal.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(len(signal)),
                                kernel.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(len(kernel)),
                                output.ctypes.data_as(ctypes.c_void_p))


@timeit()
def fftconvolve_gpu_batch(signal, kernel, output):

    _libfftconvolve.fftconvolve_batch(signal.ctypes.data_as(ctypes.c_void_p),
                                      ctypes.c_int(len(signal)),
                                      kernel.ctypes.data_as(ctypes.c_void_p),
                                      ctypes.c_int(len(kernel)),
                                      output.ctypes.data_as(ctypes.c_void_p))


fwplan = np.zeros(1, dtype=np.uint32)
bwplan = np.zeros(1, dtype=np.uint32)


@timeit()
def fftconvolve_gpu_fast(signal, signalLen, kernel, kernelLen, output):

    _libfftconvolve.fftconvolve_fast(int(signal.gpudata),
                                     ctypes.c_int(signalLen),
                                     int(kernel.gpudata),
                                     ctypes.c_int(kernelLen),
                                     int(output.gpudata),
                                     fwplan.ctypes.data_as(ctypes.c_void_p),
                                     bwplan.ctypes.data_as(ctypes.c_void_p))


@timeit()
def fftconvolve_cpu(signal, kernel):
    return scipy.signal.fftconvolve(signal, kernel)


if __name__ == "__main__":
    n_turns = 100
    n_points = int(10000)

    if(len(sys.argv) > 1):
        n_turns = int(sys.argv[1])
    if(len(sys.argv) > 2):
        n_points = int(sys.argv[2])

    np.random.seed(0)
    signal = np.random.randn(n_points)
    signal = np.array(signal, dtype=np.float64)
    kernel = np.random.randn(n_points)
    kernel = np.array(kernel, dtype=np.float64)

    result_gpu = np.zeros(len(signal) + len(kernel) - 1, dtype=np.float64)
    result_cpu = np.zeros(len(signal) + len(kernel) - 1, dtype=np.float64)

    d_signal = gpuarray.to_gpu(np.concatenate(
        (signal, np.zeros(len(kernel) - 1))))
    d_kernel = gpuarray.to_gpu(np.concatenate((
        kernel, np.zeros(len(signal) - 1))))
    d_result = gpuarray.to_gpu(result_cpu)
    # d_signal = gpuarray.to_gpu(signal)
    # d_out_signal = gpuarray.empty((out_signal.size,), dtype=np.complex128)
    # plan = np.zeros(1, dtype=np.uint32)

    for i in range(n_turns):
        fftconvolve_gpu_fast(d_signal, len(signal),
                             d_kernel, len(kernel), d_result)

    for i in range(n_turns):
        fftconvolve_gpu(signal, kernel, result_gpu)

    for i in range(n_turns):
        fftconvolve_gpu_batch(signal, kernel, result_gpu)

    for i in range(n_turns):
        result_cpu = fftconvolve_cpu(signal, kernel)

    # print('CPU')
    # print(result_cpu)

    # print('GPU')
    # print(result_gpu)

    # print('GPU Fast')
    # print(d_result.get())

    if np.allclose(result_cpu, result_gpu, atol=1e-05):
        print('CPU and GPU agree')
    else:
        print('Test 1 failed')

    if np.allclose(result_cpu, d_result.get(), atol=1e-05):
        print('CPU and GPU fast agree')
    else:
        print('Test 2 failed')

    report(skip=1)

    print('Done!')
