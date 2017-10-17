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

_libfftconvolve.convolve_complex.restype = None
_libfftconvolve.convolve_complex.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p
]


# @timeit()
def convolve_complex_gpu(signal, kernel, output):
    _libfftconvolve.convolve_complex(signal.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(len(signal)),
                                     kernel.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(len(kernel)),
                                     output.ctypes.data_as(ctypes.c_void_p),
                                     int(convolve_complex_gpu.d_in.gpudata),
                                     convolve_complex_gpu.fwplan.ctypes.data_as(
                                     ctypes.c_void_p),
                                     convolve_complex_gpu.bwplan.ctypes.data_as(
                                     ctypes.c_void_p))


# @timeit()
def convolve_complex_cpu_v1(signal, kernel):
    from scipy.signal import fftconvolve
    return fftconvolve(signal, kernel)


# @timeit()
def convolve_complex_cpu_v2(signal, kernel):
    from numpy import convolve
    return convolve(signal, kernel)


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

    result_gpu = np.zeros(len(signal) + len(kernel) - 1, dtype=np.complex128)
    result_cpu_v1 = np.zeros(len(signal) + len(kernel) - 1, dtype=np.complex128)
    result_cpu_v2 = np.zeros(len(signal) + len(kernel) - 1, dtype=np.complex128)

    convolve_complex_gpu.fwplan = np.zeros(1, np.uint32)
    convolve_complex_gpu.bwplan = np.zeros(1, np.uint32)
    convolve_complex_gpu.d_in = gpuarray.empty(2 * real_size, np.complex128)

    # from pycuda.tools import PageLockedMemoryPool
    # mempool = PageLockedMemoryPool()
    # pinned_signal = mempool.allocate(signal.shape, dtype=np.complex128)
    # pinned_kernel = mempool.allocate(kernel.shape, dtype=np.complex128)

    # ctypes.memmove(pinned_signal.ctypes.data_as(ctypes.c_void_p),
    #                signal.ctypes.data_as(ctypes.c_void_p),
    #                signal.size * 2 * ctypes.sizeof(ctypes.c_double))

    # ctypes.memmove(pinned_kernel.ctypes.data_as(ctypes.c_void_p),
    #                kernel.ctypes.data_as(ctypes.c_void_p),
    #                kernel.size * 2 * ctypes.sizeof(ctypes.c_double))

    # _libfftconvolve.reset_timer()
    for i in range(n_turns):
        convolve_complex_gpu(signal, kernel, result_gpu)
    print('End of convolve_complex_gpu')
    # _libfftconvolve.report_timer()

    for i in range(n_turns):
        result_cpu_v1 = convolve_complex_cpu_v1(signal, kernel)
    print('End of convolve_complex_cpu_v1')

    for i in range(n_turns):
        result_cpu_v2 = convolve_complex_cpu_v2(signal, kernel)
    print('End of convolve_complex_cpu_v2')

    if np.allclose(result_cpu_v1, result_gpu, atol=1e-05):
        print('CPU-v1 and GPU agree')
    else:
        print('CPU-v1 and GPU disagree')

    if np.allclose(result_cpu_v1, result_cpu_v2, atol=1e-05):
        print('CPU-v2 and CPU-v1 agree')
    else:
        print('CPU-v2 and CPU-v1 disagree')

    # report(skip=1)

    print('Done!')
    # pinned_signal.base.free()
    # pinned_kernel.base.free()
    # mempool.free_held()
