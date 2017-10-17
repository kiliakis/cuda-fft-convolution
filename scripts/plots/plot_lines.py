#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import csv
from matplotlib import colors


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


def plot_lines(data, image_folder):
    plt.figure()
    plt.grid(True, which='major', alpha=0.5)
    # plt.yticks(range(0, 101, 10))
    plt.xlabel('Threads')
    plt.ylabel('Speedup')
    plt.title('Scalability')
    # plt.grid(True, which='minor', alpha=0.4)
    # plt.minorticks_on()

    header = data[0].tolist()
    data = np.array(data[1:], dtype=float)
    print(header)
    header = [x[1:] if x[0] == '_' else x for x in header]
    print(data)
    # width = 0.35
    ind = np.unique(data[:, 0])
    d = {}
    for r in data:
        c = header.index('n_particles')
        if r[c] not in d:
            d[r[c]] = [[], []]
        d[r[c]][0].append(r[-2])
        d[r[c]][1].append(r[-1])

    for k, v in d.items():
        print(v[0])
        plt.errorbar(ind, v[0][0]/v[0], linestyle='-', marker='.',
                     label=human_format(float(k))+' Particles')
        # bottom = np.zeros(len(data))
        # data[:, header.index('track')] -= data[:, header.index('drift')] + \
        #     data[:, header.index("rf_voltage_calculation")]
        # header[header.index('track')] = 'linear_interp_kick'

        # a, b = header.index('slice'), header.index('rf_voltage_calculation')
        # data[:, [a, b]] = data[:, [b, a]]
        # header[a], header[b] = header[b], header[a]

        # c = header.index('n_slices')
        # y = 100.0 - np.sum(100*data[:, c+1:] / data[:, c][:, None], axis=1)
        # plt.bar(ind, y, width, bottom=bottom, label='other')
        # bottom += y
        # y = (data[0, c] - np.sum(data[0, c+1:])) / \
        # (data[:, c] - np.sum(data[:, c+1:], axis=1))
        # plt.plot(ind, y, linestyle='-', marker='.', label='other')
    # for i in range(c+1, len(data[0])):
    #     print(header[i])
    #     y = data[0, i] / data[:, i]
    #     print(y)
    #     plt.plot(ind, y, linestyle='-', marker='.', label=header[i])
    plt.plot(ind, ind, '--', label='Ideal speedup')
    # parallel = np.sum(data[0, -3:])/data[0, c]
    # plt.plot(ind, 1/((1-parallel) + (parallel/ind)),
    #          '--', label='Amdhal\'s law')
    # plt.ylim(ymax=25)
    # plt.xticks(ind, np.array(data[:, 0], dtype=int))
    # plt.legend(loc=[1.01, 0.4], fancybox=True, framealpha=0.5)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.tight_layout()
    plt.savefig(image_folder, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("You must specify the input file and image name")
        exit(-1)
    input_file = sys.argv[1]
    image_folder = sys.argv[2]
    data = np.genfromtxt(input_file, dtype=str, delimiter='\t')
    # print(np.array(data[1:], dtype=float))
    plot_lines(data, image_folder)
    # plot_bars(data, image_folder)
