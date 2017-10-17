import subprocess
import os
# import time

print("\nPython simulation\n")

# blond_dir = '/afs/cern.ch/work/k/kiliakis/git/cpu-gpu-bench/'
blond_dir = '/home/kiliakis/git/cpu-gpu-bench/'

exe_dir = blond_dir + 'benchmarks/fft/'
# exe_list = ['matrixconvolve-cuda.py']
exe_list = ['matrixconvolve-cuda.py']
# outfiles = blond_dir + 'results/fftconvolve/matrixconvolve-bench-v1/'
outfiles = blond_dir + 'results/fftconvolve/matrixconvolve-breakdown-p100/'

if not os.path.exists(outfiles):
    os.makedirs(outfiles)

n_particles_list = ['100000', '200000', '500000',
                    '1000000', '2000000', '5000000']
# n_particles_list = ['5000000']
n_turns_list = ['100']
repeats = 1
os.chdir(exe_dir)

total_sims = len(n_particles_list) * len(exe_list) * len(n_turns_list) * repeats

current_sim = 0
for exe in exe_list:
    for n_turns in n_turns_list:
        for n_particles in n_particles_list:
            name = 'n_p-' + n_particles + '-'
            # if not os.path.exists(outfiles + exe + '/'):
            # os.makedirs(outfiles + exe + '/')
            # res = open(outfiles + name+'.res', 'w')
            stdout = open(outfiles + name+'.txt', 'w')
            for i in range(0, repeats):
                print(n_particles, i)
                # start = time.time()
                subprocess.call(['python', exe, n_turns, n_particles],
                                stdout=stdout,
                                stderr=stdout,
                                env=os.environ.copy()
                                )
                # end = time.time()
                current_sim += 1
                # res.write(str(end-start)+'\n')
                print("%lf %% is completed" %
                      (100.0 * current_sim / total_sims))
