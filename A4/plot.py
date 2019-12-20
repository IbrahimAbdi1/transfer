# MOVE TO BUILD FOLDER AND CHANGE PROGRAM ARGUMENTS PRIOR TO RUNNING
import matplotlib
matplotlib.use('Agg')
import subprocess
import re
import numpy as np
from matplotlib import pyplot as plt

ITER = 10

if __name__ == "__main__":
    cpu, k1, k2, k3, k4, k5 = [], [], [], [], [], []
    for i in range(ITER):
        proc = subprocess.Popen(["./main", "-i", "test0.pgm", "-o", "please.pgm"], stdout=subprocess.PIPE)
        proc.stdout.readline()
        output_split = re.sub(' +',' ',proc.stdout.readline().decode()).split(' ')
        cpu.append([float(output_split[1]), float(output_split[4]), float(output_split[5])])
        output_split = re.sub(' +',' ',proc.stdout.readline().decode()).split(' ')
        k1.append([float(output_split[3]), float(output_split[4]), float(output_split[5])])
        output_split = re.sub(' +',' ',proc.stdout.readline().decode()).split(' ')
        k2.append([float(output_split[3]), float(output_split[4]), float(output_split[5])])
        output_split = re.sub(' +',' ',proc.stdout.readline().decode()).split(' ')
        k3.append([float(output_split[3]), float(output_split[4]), float(output_split[5])])
        output_split = re.sub(' +',' ',proc.stdout.readline().decode()).split(' ')
        k4.append([float(output_split[3]), float(output_split[4]), float(output_split[5])])
        output_split = re.sub(' +',' ',proc.stdout.readline().decode()).split(' ')
        k5.append([float(output_split[3]), float(output_split[4]), float(output_split[5])])
    kernel_names = ["CPU", "K1", "K2", "K3", "K4", "K5"]
    kernels = [cpu, k1, k2, k3, k4, k5]
    sums = []
    pro_times = []

    for i in range(6):
        sum = 0
        pro_time = 0
        for run in kernels[i]:
            pro_time += run[0]
            sum += np.sum(run)
        sums.append(sum/ITER)
        pro_times.append(pro_time/ITER)
    plt.xlabel("Process Method")
    plt.ylabel("Time (ms)")
    plt.title("Total Time For CPU and Kernel Methods. Average over 10 runs (1mb)")
    plt.bar(kernel_names, sums)
    plt.savefig('etime_witht.png',dpi=800)
    plt.clf()
    plt.title("Execution Time For CPU and Kernel Methods. Average over 10 runs (1mb)")
    plt.xlabel("Process Method")
    plt.ylabel("Time (ms)")
    plt.bar(kernel_names, pro_times)
    plt.savefig('etime_withoutt.png',dpi=800)
    plt.clf()
    if pro_times[2] < pro_times[4]:
        print("Kernel 2 is faster than Kernel 4")
    else:
        print("Kernel 4 is faster than Kernel 2")
    if pro_times[4] * 0.9 > pro_times[5]:
        print("GPU Time for Kernel 5 is 10% faster than Kernel 4\n")
    elif sums[4] * 0.9 > sums[5]:
        print("Overall Execution Time for Kernel 5 is 10% faster than Kernel 4\n")
    else:
        print("Kernel 5 is not 10% faster than Kernel 4")

    if pro_times[2] * 0.9 > pro_times[5]:
        print("GPU Time for Kernel 5 is 10% faster than Kernel 2\n")
    elif sums[2] * 0.9 > sums[5]:
        print("Overall Execution Time for Kernel 5 is 10% faster than Kernel 2\n")
    else:
        print("Kernel 5 is not 10% faster than Kernel 2")
    
