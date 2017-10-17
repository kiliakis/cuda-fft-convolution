#!/usr/bin/python
import os
import sys
import csv
import numpy as np


def string_between(string, before, after):
    temp = string.split(before)[1]
    temp = temp.split(after)[0]
    return temp


# header = ['version', 'function', 'n_threads', 'n_bunches', 'n_particles',
#           'n_turns', 'n_slices', 'turn_time', 'stdev',
#           'throughput(Mp/s)']

header = ['function', 'n_points',
          'n_turns', 'turn_time', 'stdev']


def read_data(input_dir, out_file):
    records = []
    for dirs, subdirs, files in os.walk(input_dir):
        for file in files:
            if('.txt' not in file):
                continue
            # record = []
            # version = string_between(file, 'ver-', '-')
            # threads = '1'
            turns = '200'
            # try:
            #     slices = string_between(file, 'slices-', '.')
            # except Exception as e:
            #     slices = '0'
            n_p = string_between(file, 'n_p-', '-')
            # n_bunches = '0'
            # record = [version, threads, n_bunches, n_p, turns, slices]
            # l = {'kick': [], 'drift': []}
            # l = {'histo': []}
            # header = []
            for line in open(os.path.join(dirs, file), 'r'):
                line = line.lower()
                if ('average time' not in line):
                    continue
                time = string_between(line, 'time:', 'ms').strip()
                name = line.split('[')[1].split(']')[0].split('.')[-1]
                records.append([name, n_p, turns, time, '0'])
                # l['histo'].append(float(time))
                # if 'kick' in line:
                #     l['kick'].append(float(time))
                # elif 'drift' in line:
                #     l['drift'].append(float(time))
            # for k, v in l.items():
            #     mean_time = np.mean(v)
            #     throughput = int(n_p) / mean_time / 1e3
            #     record = [version, k, threads, n_bunches, n_p, turns,
            #               slices, mean_time, np.std(v), throughput]
            #     records.append(record)

    records.sort(key=lambda a: (int(a[1]), a[0], int(a[2])))
    writer = csv.writer(open(out_file, 'w'),
                        lineterminator='\n', delimiter='\t')
    writer.writerow(header)
    writer.writerows(records)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("You must specify the input directory and the output file")
        exit(-1)
    input_dir = sys.argv[1]
    out_file = sys.argv[2]

    read_data(input_dir, out_file)
