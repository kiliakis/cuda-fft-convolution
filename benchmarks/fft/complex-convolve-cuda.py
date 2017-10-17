# simlpe test to use the cuda shared library
import ctypes
import numpy as np
import os
import sys
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

from pyprof.timing import timeit, report
import scipy.fftpack

_libfftconvolve = ctypes.cdll.LoadLibrary(
    os.path.dirname(os.path.abspath(__file__)) + '/../../src/libfftconvolve.so')

_libfftconvolve.matrixconvolve_v2.restype = None
_libfftconvolve.matrixconvolve_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p
]


_libfftconvolve.matrixconvolve_v1.restype = None
_libfftconvolve.matrixconvolve_v1.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p
]


@timeit()
def matrixconvolve_gpu_v2(signal, kernel, output):
    _libfftconvolve.matrixconvolve_v2(signal.ctypes.data_as(ctypes.c_void_p),
                                      ctypes.c_int(len(signal)),
                                      kernel.ctypes.data_as(ctypes.c_void_p),
                                      ctypes.c_int(len(kernel)),
                                      output.ctypes.data_as(ctypes.c_void_p),
                                      int(matrixconvolve_gpu_v2.d_in.gpudata),
                                      matrixconvolve_gpu_v2.fwplan1.ctypes.data_as(
                                          ctypes.c_void_p),
                                      matrixconvolve_gpu_v2.fwplan2.ctypes.data_as(
                                          ctypes.c_void_p),
                                      matrixconvolve_gpu_v2.bwplan.ctypes.data_as(
                                          ctypes.c_void_p))


@timeit()
def matrixconvolve_gpu_v1(signal, kernel, output):
    _libfftconvolve.matrixconvolve_v1(signal.ctypes.data_as(ctypes.c_void_p),
                                      ctypes.c_int(len(signal)),
                                      kernel.ctypes.data_as(ctypes.c_void_p),
                                      ctypes.c_int(len(kernel)),
                                      output.ctypes.data_as(ctypes.c_void_p),
                                      int(matrixconvolve_gpu_v1.d_in.gpudata),
                                      matrixconvolve_gpu_v1.fwplan.ctypes.data_as(
                                          ctypes.c_void_p),
                                      matrixconvolve_gpu_v1.bwplan.ctypes.data_as(
                                          ctypes.c_void_p))


@timeit()
def matrixconvolve_cpu_v0(signal, kernel):
    from scipy.signal import fftconvolve
    return ((fftconvolve(signal.real, kernel.real) -
             fftconvolve(signal.imag, kernel.imag)) +
            1j * (fftconvolve(signal.imag, kernel.real) +
                  fftconvolve(signal.real, kernel.imag)))


@timeit()
def matrixconvolve_cpu_v2(signal, kernel):
    from scipy.signal import fftconvolve
    return fftconvolve(signal, kernel)


@timeit()
def matrixconvolve_cpu_v1(signal, kernel):
    from scipy.fftpack import fft, ifft
    real_size = len(signal) + len(kernel) - 1
    return (ifft(fft(signal, real_size) * fft(kernel, real_size)))


if __name__ == "__main__":
    n_turns = 100
    n_points = int(10000)

    if(len(sys.argv) > 1):
        n_turns = int(sys.argv[1])
    if(len(sys.argv) > 2):
        n_points = int(sys.argv[2])

    np.random.seed(0)
    signal = np.random.randn(n_points) + np.random.randn(n_points) * 1j
    signal = np.array(signal, dtype=np.complex128)
    kernel = np.random.randn(n_points) + np.random.randn(n_points) * 1j
    kernel = np.array(kernel, dtype=np.complex128)

    real_size = len(signal) + len(kernel) - 1
    complex_size = real_size // 2 + 1

    result_gpu_v1 = np.zeros(len(signal) + len(kernel) - 1, dtype=np.complex128)
    result_gpu_v2 = np.zeros(len(signal) + len(kernel) - 1, dtype=np.complex128)
    result_cpu_v0 = np.zeros(len(signal) + len(kernel) - 1, dtype=np.complex128)
    result_cpu_v2 = np.zeros(len(signal) + len(kernel) - 1, dtype=np.complex128)

    matrixconvolve_gpu_v1.fwplan = np.zeros(1, np.uint32)
    matrixconvolve_gpu_v1.bwplan = np.zeros(1, np.uint32)
    matrixconvolve_gpu_v1.d_in = gpuarray.empty(2 * real_size, np.complex128)

    matrixconvolve_gpu_v2.fwplan1 = np.zeros(1, np.uint32)
    matrixconvolve_gpu_v2.fwplan2 = np.zeros(1, np.uint32)
    matrixconvolve_gpu_v2.bwplan = np.zeros(1, np.uint32)
    matrixconvolve_gpu_v2.d_in = gpuarray.empty(2 * real_size, np.complex128)

    from pycuda.tools import PageLockedMemoryPool
    mempool = PageLockedMemoryPool()
    pinned_signal = mempool.allocate(signal.shape, dtype=np.complex128)
    pinned_kernel = mempool.allocate(kernel.shape, dtype=np.complex128)

    ctypes.memmove(pinned_signal.ctypes.data_as(ctypes.c_void_p),
                   signal.ctypes.data_as(ctypes.c_void_p),
                   signal.size * 2 * ctypes.sizeof(ctypes.c_double))

    ctypes.memmove(pinned_kernel.ctypes.data_as(ctypes.c_void_p),
                   kernel.ctypes.data_as(ctypes.c_void_p),
                   kernel.size * 2 * ctypes.sizeof(ctypes.c_double))

    # matrixconvolve_gpu_v2.fwplan = np.zeros(1, np.uint32)
    # matrixconvolve_gpu_v2.d_in = gpuarray.empty(2 * real_size, np.complex128)

    # fft_signal = scipy.fftpack.fft(signal, real_size)
    # fft_kernel = scipy.fftpack.fft(kernel, real_size)

    # print('signal', fft_signal)
    # print('kernel', fft_kernel)
    # print('inverse', scipy.fftpack.ifft(fft_signal * fft_kernel))

    # for i in range(n_turns):
    #     result_cpu_v0 = matrixconvolve_cpu_v0(signal, kernel)
    # print('End matrixconvolve_cpu_v0')

    # for i in range(n_turns):
    #     result_cpu_v1 = matrixconvolve_cpu_v1(signal, kernel)
    # print('End matrixconvolve_cpu_v1')

    # for i in range(n_turns):
    #     result_cpu_v2 = matrixconvolve_cpu_v2(signal, kernel)
    # print('End matrixconvolve_cpu_v2')
    _libfftconvolve.reset_timer()
    for i in range(n_turns):
        matrixconvolve_gpu_v1(signal, kernel, result_gpu_v1)
    print('End matrixconvolve_gpu_v1')
    _libfftconvolve.report_timer()

    _libfftconvolve.reset_timer()
    for i in range(n_turns):
        matrixconvolve_gpu_v2(pinned_signal, pinned_kernel, result_gpu_v2)
    print('End matrixconvolve_gpu_v2')
    _libfftconvolve.report_timer()

    # for i in range(n_turns):
    #     matrixconvolve_gpu_v2(signal, kernel, result_gpu_v2)
    # print('End matrixconvolve_gpu_v2')

    # print('CPU')
    # print(result_cpu_v0)

    # print('GPU')
    # print(result_gpu_v0)

    # if np.allclose(result_cpu_v0, result_cpu_v2, atol=1e-05):
    #     print('CPU-v0 and CPU-v1 agree')
    # else:
    #     print('CPU-v0 and CPU-v1 disagree')

    # if np.allclose(result_cpu_v2, result_gpu_v0, atol=1e-05):
    #     print('CPU-v2 and GPU-v0 agree')
    # else:
    #     print('CPU-v2 and GPU-v0 disagree')

    if np.allclose(result_gpu_v1, result_gpu_v2, atol=1e-05):
        print('GPU-v2 and GPU-v1 agree')
    else:
        print('GPU-v2 and GPU-v1 disagree')

    # if np.allclose(result_cpu_v2, result_gpu_v2, atol=1e-05):
    #     print('CPU-v2 and GPU-v2 agree')
    # else:
    #     print('CPU-v2 and GPU-v2 disagree')

    report(skip=1)

    print('Done!')
    pinned_signal.base.free()
    pinned_kernel.base.free()
    mempool.free_held()
