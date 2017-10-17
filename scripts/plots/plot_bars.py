#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os
from plot.plotting_utilities import *

hatces = ['x', '\\', '/']
colors = ['.25', '.5', '.75']

home = '/afs/cern.ch/work/k/kiliakis/git/cpu-gpu-bench/'
res_file = home + 'results/fftconvolve/matrixconvolve-bench-v2.csv'
images_dir = home + 'results/fftconvolve/'
image_name = 'matrixconvolve-bench-v3.pdf'

y_label = 'Normalized Thoughput'
x_label = 'Signal Size'
# plot_title = 'FFT Convolution'
# plot_title = 'lin_interp_kick'
title = 'Matrix FFT Convolution'
grouping = ['function']
prefixes = ['']
keeponly = ['n_points', 'turn_time']
x_lims = []
y_lims = []

names = {'matrixconvolve_cpu_v2': 'CPU Complex Convolution',
         'matrixconvolve_cpu_v0': 'CPU Real Convolution',
         'matrixconvolve_gpu_v1': 'Tesla K20X Complex Convolution',
         'matrixconvolve_gpu_p100': 'Pascal P100 Complex Convolution'}

opacity = 0.85
width = 0.35
# start = -width/2
start = 0

show = 0

if __name__ == '__main__':

    data = np.genfromtxt(res_file, dtype=str)
    header = data[0].tolist()
    data = data[1:]
    all_plots = group_by(header, data, grouping, prefixes)
    # version = file.split('.csv')[0]
    all_plots = keep_only(header, all_plots, keeponly)
    print(all_plots)

    plt.figure()
    plt.grid(True, which='major', alpha=0.5)
    # plt.tick_params(labelright=True)
    plt.xlabel(x_label)
    if(x_lims):
        plt.xlim(x_lims)
    if(y_lims):
        plt.ylim(y_lims)

    plt.title(title)
    plt.ylabel(y_label)

    normalize = np.array(all_plots['matrixconvolve_gpu_v1'][1], float)

    for plot_name, plots in all_plots.items():
        if 'gpu' not in plot_name:
            continue
        N = len(plots[0])+1
        ind = np.linspace(0, N, N)
        # plt.figure(figsize=(6, 3))

        # define the starting position and the way the bars will be arranged

        # plots.sort(key=lambda a: a[0])
        label = plot_name
        if label in names:
            label = names[label]
        x = np.array(plots[0], dtype=int)
        y = normalize / np.array(plots[1], dtype=float)
        y = np.append(y, [np.mean(y)])
        p = plt.bar(ind + start, y, width, label=label, alpha=opacity)
        autolabel(plt.gca(), p, rounding=2, fontsize=8)
        start += width
        x = [human_format(i) for i in x]
        plt.xticks(ind+width/2, x + ['Average'])

    # normalize = np.array(all_plots['matrixconvolve_cpu_v2'][1], float)
    # label = 'P100 / CPU Complex'
    # y = normalize / np.array(all_plots['matrixconvolve_gpu_p100'][1], float)
    # y = np.append(y, [np.mean(y)])
    # p = plt.bar(ind + start - width, y, width, label=label, alpha=opacity)

    # autolabel(plt.gca(), p, rounding=2, fontsize=8.5)

    plt.legend(loc='center left', fancybox=True, framealpha=0.5, fontsize=10,
               bbox_to_anchor=(0., 0.6))
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(images_dir + image_name, dpi=300)
    plt.close()

    # autolabel(simple)
    # autolabel(batched)
