#!/usr/bin/env python3

import csv
import math
import os
import sys

import matplotlib.pyplot as plt

def parse_benchmarks(benchmarks):
    results = {}

    # parse .csv data
    for i, bm in enumerate(benchmarks):
        name, data = bm

        if name not in results:
            results[name] = {}

        for row in data:
            n_clusters = int(row['clusters'])
            if n_clusters not in results[name]:
                results[name][n_clusters] = {}

            dim = int(row['dim'])
            time = 1000 * float(row['time'])
            if dim not in results[name][n_clusters]:
                results[name][n_clusters][dim] = [time]
            else:
                results[name][n_clusters][dim].append(time)

    return results

def plot_comparison(name1, data1, name2, data2):
    # determine median cluster size
    n_clusters1 = sorted(data1)[len(data1) // 2]
    n_clusters2 = sorted(data2)[len(data2) // 2]
    if n_clusters1 != n_clusters2:
        raise ValueError("incompatible data")

    # prepare data
    def parse_runtimes(runtimes):
        dims = []
        times = []
        for dim in runtimes:
            dims.append(dim)
            times.append(runtimes[dim])

        times = [t for _, t in sorted(zip(dims, times))]
        dims = sorted(dims)

        return dims, times

    dims1, times1 = parse_runtimes(data1[n_clusters1])
    dims2, times2 = parse_runtimes(data2[n_clusters2])

    if dims1 != dims2:
        raise ValueError("incompatible data")

    dims = dims1

    # common plot settings
    plt.rc('text', usetex=True)
    labelsize = 20
    labelpad = 20

    l = lambda x : r'${d} \times {d}$'.format(d=int(x))
    xticklabels = list(map(l, dims))

    # create boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    def bplot(ax, name, n_clusters, dims, times):
        bplot = ax.boxplot(times, patch_artist=True)

        elems = ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']
        for elem in elems:
            plt.setp(bplot[elem], color='lightslategray')

        plt.setp(bplot['boxes'], facecolor='lightsteelblue')
        plt.setp(bplot['fliers'],
                 markerfacecolor='lightsteelblue',
                 markeredgecolor='lightslategray')

        ax.plot(range(1, len(dims) + 1),
                [sum(t) / len(t) for t in times],
                color='gray', linestyle='--')

        ax.set_title('{} ({} Clusters)'.format(name, n_clusters))

        xlabel = r'$\frac{A_{image}}{pixels^2}$'
        ax.set_xlabel(xlabel, size=labelsize)

        ylabel = r'$\frac{t_{exec}}{\mu s}$'
        ax.set_ylabel(ylabel, size=labelsize, labelpad=labelpad, rotation=0)
        ax.yaxis.set_label_coords(0.0, 1.02)

        ax.set_xticklabels(xticklabels, rotation=-45, ha='left')

        ax.grid(True)

    bplot(ax1, name1, n_clusters1, dims, times1)
    bplot(ax2, name2, n_clusters2, dims, times2)

    # save plots to file
    fig.set_size_inches(10, 5)

    fname = 'report/resources/{}_{}_boxplots.svg'.format(name1, name2)
    plt.savefig(fname)

    fig.clear()

    # create speedup plot
    speedups = []
    for t1, t2 in zip(times1, times2):
        avg1 = sum(t1) / len(t1)
        avg2 = sum(t2) / len(t2)
        speedups.append(avg1 / avg2)

    plt.plot(dims, speedups, 'o')
    plt.axhline(y=(sum(speedups) / len(speedups)), linestyle='--')

    plt.title('Speedup {} vs. {}'.format(name2, name1))
    plt.xlabel(r'$\frac{A_{image}}{pixels^2}$', size=labelsize)

    plt.xticks(dims, xticklabels, rotation=-45, ha='left')

    plt.grid(True)

    # save plots to file
    fig.set_size_inches(10, 5)

    fname = 'report/resources/{}_{}_speedup.svg'.format(name1, name2)
    plt.savefig(fname)

    fig.clear()

if __name__ == '__main__':
    benchmarks = []

    # gather benchmark .csv files
    for root, dirs, files in os.walk(sys.argv[1]):
        csv_files = []
        for filename in files:
            if filename.endswith('.csv'):
                csv_files.append(filename)

        for csv_file in sorted(csv_files):
            benchmark_name = csv_file[:csv_file.find('.csv')]
            with open(os.path.join(root, csv_file)) as benchmark_csv:
                benchmark_data = []
                reader = csv.DictReader(benchmark_csv)
                for row in reader:
                    benchmark_data.append(row)

                benchmarks.append((benchmark_name, benchmark_data))

    # create plots
    results = parse_benchmarks(benchmarks)
    plot_comparison("C", results['C'], "OpenMP", results['OpenMP_double'])
