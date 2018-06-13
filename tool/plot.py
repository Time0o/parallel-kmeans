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


def parse_runtimes(runtimes):
    dims = []
    times = []
    for dim in runtimes:
        dims.append(dim)
        times.append(runtimes[dim])

    times = [t for _, t in sorted(zip(dims, times))]
    dims = sorted(dims)

    return dims, times


def plot_simple(dims, results):
    for name, runtimes in results:
        medians = [rt[len(rt) // 2] for rt in runtimes]
        plt.plot(dims, medians, '--o', label=name)

    xticklabels = map(lambda x : r'${d} \times {d}$'.format(d=int(x)), dims)
    plt.xticks(dims, list(xticklabels), rotation=-45, ha='left')
    plt.xlabel(r'$\frac{A_{image}}{pixels^2}$', size=15)
    plt.ylabel(r'$\frac{t_{exec}}{\mu s}$', size=15)

    plt.grid(True)
    plt.legend()


def plot_speedup(dims, runtimes_ref, runtimes_variant):
    rt_avgs_ref = [sum(rt) / len(rt) for rt in runtimes_ref]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (label, runtimes) in enumerate(runtimes_variant):
        speedups = []
        for j, rt in enumerate(runtimes):
            speedups.append(rt_avgs_ref[j] / (sum(rt) / len(rt)))

        color = colors[i]
        plt.plot(dims, speedups, 'x', label=label, color=color)
        speedup_avg = sum(speedups) / len(speedups)
        plt.axhline(y=speedup_avg, color=color, linestyle='--')

    xticklabels = map(lambda x : r'${d} \times {d}$'.format(d=int(x)), dims)
    plt.xticks(dims, list(xticklabels), rotation=-45, ha='left')
    plt.xlabel(r'$\frac{A_{image}}{pixels^2}$', size=15)

    plt.grid(True)
    plt.legend()


def plot_boxplot(dims, runtimes):

    def color_boxplot(bplot, facecolor, edgecolor):
        elems = ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']
        for elem in elems:
            plt.setp(bplot[elem], color=edgecolor)

        plt.setp(bplot['boxes'], facecolor=facecolor)
        plt.setp(bplot['fliers'], markerfacecolor=facecolor)
        plt.setp(bplot['fliers'], markeredgecolor=edgecolor)

    fig, ax = plt.subplots()

    # reference boxplots
    bplot = ax.boxplot(runtimes, patch_artist=True)
    color_boxplot(bplot, 'lightsteelblue', 'lightslategray')

    # reference center line
    xticks = range(1, len(dims) + 1)
    medians = [rt[len(rt) // 2] for rt in runtimes]
    ax.plot(xticks, medians, color='gray', linestyle='--')

    # reference formatting
    xticklabels = map(lambda x : r'${d} \times {d}$'.format(d=int(x)), dims)
    ax.set_xticklabels(xticklabels, rotation=-45, ha='left')
    xlabel = r'$\frac{A_{image}}{pixels^2}$'
    ax.set_xlabel(xlabel, size=15)

    ylabel = r'$\frac{t_{exec}}{\mu s}$'
    ax.set_ylabel(ylabel, size=15, rotation=0)
    ax.yaxis.set_label_coords(0.0, 1.02)

    ax.grid(True)


def save_plot(filename):
    fname = 'report/resources/{}.svg'.format(filename)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.gcf().clear()


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

    # median cluster size
    k = sorted(results['C'])[len(results['C']) // 2]

    # dimensions
    dims, _ = parse_runtimes(results['C'][k])

    # results
    _, c_runtimes = parse_runtimes(results['C'][k])
    _, cuda_runtimes = parse_runtimes(results['CUDA'][k])
    _, omp1_runtimes = parse_runtimes(results['OpenMP_single'][k])
    _, omp2_runtimes = parse_runtimes(results['OpenMP_double'][k])
    _, omp3_runtimes = parse_runtimes(results['OpenMP_triple'][k])
    _, omp4_runtimes = parse_runtimes(results['OpenMP_quad'][k])

    # simple plot
    plot_simple(dims, [('C', c_runtimes),
                       ('CUDA C', cuda_runtimes),
                       ('C + OpenMP (1 core)', omp1_runtimes),
                       ('C + OpenMP (2 cores)', omp2_runtimes),
                       ('C + OpenMP (3 cores)', omp3_runtimes),
                       ('C + OpenMP (4 cores)', omp4_runtimes)])

    save_plot('All_plot')

    # speedup plots
    plot_speedup(dims, c_runtimes, [('C + OpenMP (1 core)', omp1_runtimes),
                                    ('C + OpenMP (2 cores)', omp2_runtimes),
                                    ('C + OpenMP (3 cores)', omp3_runtimes),
                                    ('C + OpenMP (4 cores)', omp4_runtimes)])

    save_plot('C_OMP_speedup')

    # boxplots
    plot_boxplot(dims, c_runtimes)
    save_plot('C_boxplot')

    plot_boxplot(dims, cuda_runtimes)
    save_plot('CUDA_boxplot')

    plot_boxplot(dims, omp1_runtimes)
    save_plot('OpenMP_single_boxplot')

    plot_boxplot(dims, omp2_runtimes)
    save_plot('OpenMP_double_boxplot')

    plot_boxplot(dims, omp3_runtimes)
    save_plot('OpenMP_triple_boxplot')

    plot_boxplot(dims, omp4_runtimes)
    save_plot('OpenMP_quad_boxplot')
