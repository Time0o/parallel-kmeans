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


def plot_speedup(title, dims, runtimes_ref, variant_runtimes):
    rt_avgs_ref = []
    for rt_ref in runtimes_ref:
        rt_avgs_ref.append(sum(rt_ref) / len(rt_ref))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (label, runtimes) in enumerate(variant_runtimes):
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

    plt.title(title)
    plt.grid(True)
    plt.legend()


def boxplots(name_ref, dims_ref, runtimes_ref, name_variants, variant_runtimes):

    def color_boxplot(bplot, facecolor, edgecolor):
        elems = ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']
        for elem in elems:
            plt.setp(bplot_ref[elem], color=edgecolor)

        plt.setp(bplot_ref['boxes'], facecolor=facecolor)
        plt.setp(bplot_ref['fliers'], markerfacecolor=facecolor)
        plt.setp(bplot_ref['fliers'], markeredgecolor=edgecolor)

    fig, (ax_ref, ax) = plt.subplots(1, 2)

    # reference boxplots
    bplot_ref = ax_ref.boxplot(runtimes_ref, patch_artist=True)
    color_boxplot(bplot_ref, 'lightsteelblue', 'lightslategray')

    # reference center line
    xticks = range(1, len(dims_ref) + 1)
    runtime_avgs = [sum(t) / len(t) for t in runtimes_ref]
    ax_ref.plot(xticks, runtime_avgs, color='gray', linestyle='--')

    # reference formatting
    xticklabels = map(lambda x : r'${d} \times {d}$'.format(d=int(x)), dims_ref)
    ax_ref.set_xticklabels(xticklabels, rotation=-45, ha='left')
    xlabel = r'$\frac{A_{image}}{pixels^2}$'
    ax_ref.set_xlabel(xlabel, size=15)

    ylabel = r'$\frac{t_{exec}}{\mu s}$'
    ax_ref.set_ylabel(ylabel, size=15, rotation=0)
    ax_ref.yaxis.set_label_coords(0.0, 1.02)

    ax_ref.set_title(name_ref)
    ax_ref.grid(True)

    # TODO


def plot_comparisons(name_ref, data_ref, name_variants, variants):
    # determine median cluster size
    n_clusters = sorted(data_ref)[len(data_ref) // 2]

    # parse runtime data
    dims_ref, runtimes_ref = parse_runtimes(data_ref[n_clusters])

    variant_runtimes = []
    for label, data in variants:
        dims, runtimes = parse_runtimes(data[n_clusters])
        if dims != dims_ref:
            raise ValueError("incompatible data")

        variant_runtimes.append((label, runtimes))

    # create boxplots
    boxplots(name_ref, dims_ref, runtimes_ref, name_variants, variant_runtimes)

    # save plots
    fname = 'report/resources/{}_{}_boxplots.svg'.format(name_ref, name_variants)
    plt.savefig(fname)
    plt.gcf().clear()

    # create speedup plot
    title = 'Speedup {} vs. {}'.format(name_ref, name_variants)
    plot_speedup(title, dims_ref, runtimes_ref, variant_runtimes)

    # save plot
    fname = 'report/resources/{}_{}_speedup.svg'.format(name_ref, name_variants)
    plt.savefig(fname)
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

    omp = []
    omp.append(("1 core", results["OpenMP_single"]))
    omp.append(("2 cores", results["OpenMP_double"]))
    omp.append(("3 cores", results["OpenMP_triple"]))
    omp.append(("4 cores", results["OpenMP_quad"]))
    plot_comparisons("C", results['C'], "OpenMP", omp)
