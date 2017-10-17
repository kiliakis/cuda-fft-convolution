#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# import os
import numpy as np
# import sys
# import csv
# from matplotlib import colors
from plot.plotting_utilities import *
# from cycler import cycler

# plt.rc('axes', prop_cycle=(cycler('color', ['blue', 'green', 'red', 'cyan',
#                                             'magenta', 'black', 'purple',
#                                             'pink', 'brown', 'orange', 'teal', 'coral',
#                                             'lightblue', 'lime', 'lavender', 'turquoise',
#                                             'darkgreen', 'tan', 'salmon', 'gold',
#                                             'darkred', 'darkblue'])))
home = '/afs/cern.ch/work/k/kiliakis/git/cpu-gpu-bench/'
input_file = home + 'results/fftconvolve/matrixconvolve-breakdown-p100.csv'
image_name = [home + 'results/fftconvolve/matrixconvolve-pies-p100.pdf']
key_names = ['n_points']
prefixes = ['']
to_keep = ['function', 'turn_time']
show = 0
# xlabel = ['Threads']
# ylabel = ['Run time %']
title = ['FFT Convolution execution time breakdown']

names = {
}

serie = ['alloc_input', 'alloc_complex', 'memset', 'memcpy', 'create_plans',
         'forward', 'multiply', 'backward', 'copy_back', 'finalize', 'convolve']


def plot_bars(data, image_name):
    # plt.figure(figsize=(6.5, 4))
    plt.figure()
    # plt.grid(True, which='major', alpha=0.5)
    # plt.yticks(range(0, 101, 10))
    # plt.xlabel(xlabel.pop())
    # plt.ylabel(ylabel.pop())
    plt.title(title.pop())

    header = data[0].tolist()
    # data = data[1:]
    # print(header)
    # print(data)
    data = group_by(header, data[1:], key_names, prefixes)
    data = keep_only(header, data, to_keep)
    print(data)
    # width = .5
    # ind = np.arange(len(data))
    # bottom = np.zeros(len(data))
    rows = int(np.sqrt(len(data)))
    cols = (len(data) + rows - 1) // rows
    print(rows, cols)
    grid = GridSpec(rows, cols)
    # grid.update()
    # return
    i = 0
    for points, values in data.items():
        row = i // cols
        col = i % cols
        print('row, col : %d, %d' % (row, col))
        plt.subplot(grid[row, col], aspect=1)
        plt.xlabel('%s points' % human_format(int(points)))
        i += 1
        labels = values[0]
        sizes = values[1]
        total_time_idx = labels.index('total_time')
        total_time = float(sizes[total_time_idx])
        labels.remove('total_time')
        sizes.remove(str(total_time))

        sizes = np.array(values[1], float)
        # sizes = []
        # matrixconvolve_idx = labels.index('matrixconvolve_gpu')
        # sizes[matrixconvolve_idx] -= total_time
        # labels[matrixconvolve_idx] = 'python_call'
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0., 1., len(sizes)))
        explode = [0] * len(sizes)

        explode[labels.index('forward')] = .08
        explode[labels.index('backward')] = .08
        # explode[serie.index('memset')] = .05
        # explode[serie.index('copy_back')] = .05
        # explode[serie.index('finalize')] = .05
        patches, texts, autotexts = plt.pie(sizes, shadow=False, colors=colors,
                                            counterclock=False,
                                            autopct='%1.1f%%',
                                            textprops={'fontsize': '8'}, startangle=90,
                                            explode=explode)
        for t in autotexts:
            if(float(t.get_text().split('%')[0]) < 4):
                t.set_text('')
        autotexts[0].set_color('w')
        plt.axis('equal')
    # plt.subplot(grid[0, 0])
    # plt.legend(labels, loc='best', fancybox=True, framealpha=0.5)
    plt.legend(labels, loc='upper center', bbox_to_anchor=(-0.6, 2.4), ncol=5,
               fancybox=True, fontsize=8, framealpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(image_name.pop(), bbox_inches='tight')

    plt.close()


if __name__ == '__main__':
    # if len(sys.argv) < 3:
    #     print("You must specify the input file and image folder")
    #     exit(-1)
    # input_file = sys.argv[1]
    # image_folder = sys.argv[2]
    data = np.genfromtxt(input_file, dtype=str, delimiter='\t')
    plot_bars(data, image_name)
