#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import csv
from matplotlib import colors

prefix = '/home/kiliakis/work/git/cpu-gpu-bench'

input_file = prefix + '/results/fft-times-v1.csv'
image_names = [prefix + '/results/fft/cufft-speedups.pdf',
               prefix + '/results/fft/cufft-throughput.pdf']
y_columns = ['turn_time', 'throughput(MP/s)']


def annotate(ax, A, B, **kwargs):
    for x, y in zip(A, B):
        ax.annotate('%.1f' % y, xy=(x, y), textcoords='data', **kwargs)


def group_by(header, list, key_names, prefixes):
    d = {}
    for r in list:
        key = []
        for k in key_names:
            key += [prefixes[key_names.index(k)] + r[header.index(k)]]
        key = ''.join(key)
        if key not in d:
            d[key] = []
        d[key] += [r]

    return d


def keep_only(dir, header, to_keep):
    d = {}
    for k, values in dir.items():
        if k not in d:
            d[k] = []
        for h in to_keep:
            c = header.index(h)
            l = []
            for v in values:
                l.append(v[c])
            d[k].append(l)
    return d


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000
    # add more suffixes if you need them
    return '%d%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def string_between(string, before, after):
    temp = string.split(before)[1]
    temp = temp.split(after)[0]
    return temp


def plot_speedup(data, image_name):
    plt.figure()
    plt.grid(True, which='major', alpha=0.5)
    plt.xlabel('log10(points)')
    plt.ylabel('Speedup')
    plt.title('FFT benchmark')

    header = data[0].tolist()
    data = np.array(data[1:])
    header = [x[1:] if x[0] == '_' else x for x in header]
    dir = group_by(header, data.tolist(), ['version'], [''])
    dir = keep_only(dir, header, ['n_particles',
                                  'turn_time', 'throughput(kp/s)'])
    # print(dir)
    base = np.array(dir['fftnumpy'][1], dtype=float)
    for version in dir.keys():
        x = dir[version][0]
        y = base / np.array(dir[version][1], dtype=float)
        x = np.log10(np.array(x, dtype=float))
        p = plt.plot(x, y, linestyle='-', marker='.',
                     label=version + '_vs_fftnumpy')
        if (version != 'fftnumpy'):
            annotate(plt.gca(), x[-4:], y[-4:],
                     size='9', color=p[0].get_color())

    base = np.array(dir['fftwcpp'][1], dtype=float)
    for version in ['cufft']:
        x = dir[version][0]
        y = base / np.array(dir[version][1], dtype=float)
        x = np.log10(np.array(x, dtype=float))
        p = plt.plot(x, y, linestyle='-', marker='.',
                     label=version + '_vs_fftwcpp')
        if (version is not 'fftnumpy'):
            annotate(plt.gca(), x[-3:], y[-3:],
                     size='9', color=p[0].get_color())

    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.tight_layout()
    plt.savefig(image_name, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_throughput(data, image_name):
    plt.figure()
    plt.grid(True, which='major', alpha=0.5)
    plt.xlabel('log10(points)')
    plt.ylabel('Throughput (MPoints/sec)')
    # plt.title('Scalability')

    header = data[0].tolist()
    data = np.array(data[1:])
    header = [x[1:] if x[0] == '_' else x for x in header]
    dir = group_by(header, data.tolist(), ['version'], [''])
    dir = keep_only(dir, header, ['n_particles',
                                  'turn_time', 'throughput(kp/s)'])
    # print(dir)
    # base = np.array(dir['fftnumpy'][1], dtype=float)
    for version in dir.keys():
        x = dir[version][0]
        y = np.array(dir[version][2], dtype=float)
        x = np.log10(np.array(x, dtype=float))
        p = plt.plot(x, y, linestyle='-', marker='.',
                     label=version)
        if (version != 'fftnumpy'):
            annotate(plt.gca(), x[-4:], y[-4:],
                     size='9', color=p[0].get_color())

    # base = np.array(dir['fftwcpp'][1], dtype=float)
    # for version in ['cufft']:
    #     x = dir[version][0]
    #     y = base / np.array(dir[version][1], dtype=float)
    #     x = np.log10(np.array(x, dtype=float))
    #     p = plt.plot(x, y, linestyle='-', marker='.',
    #                  label=version + '_vs_fftwcpp')
    #     if (version is not 'fftnumpy'):
    #         annotate(plt.gca(), x[-3:], y[-3:],
    #                  size='9', color=p[0].get_color())

    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.tight_layout()
    plt.savefig(image_name, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    data = np.genfromtxt(input_file, dtype=str, delimiter='\t')
    plot_speedup(data, image_names[0])
    plot_throughput(data, image_names[1])
