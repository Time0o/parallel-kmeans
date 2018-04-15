#!/usr/bin/env python3

import csv
import math
import os
import sys

import matplotlib.pyplot as plt

def plot_benchmarks(benchmarks):
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

    it = iter(sorted(results))

    subplot_dim = math.ceil(math.sqrt(len(results)))
    for y in range(subplot_dim):
        for x in range(subplot_dim):
            p = y * subplot_dim + x + 1
            if p == len(results) + 1:
                break

            # generate boxplots for median cluster size
            ax = plt.subplot(subplot_dim, subplot_dim, p)
            name = next(it)
            tmp = sorted(results[name])
            n_clusters = tmp[len(tmp) // 2]

            dims = []
            times = []
            for dim in results[name][n_clusters]:
                dims.append(dim)
                times.append(results[name][n_clusters][dim])

            # plot results
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
            ax.set_xlabel(r'$\frac{A_{image}}{pixels^2}$', size=15, labelpad=20)
            ax.set_ylabel(r'$\frac{t_{exec}}{\mu s}$', size=15, labelpad=20,
                          rotation=0)

            ax.set_xticklabels(
                map(lambda x : '{d}x{d}'.format(d=int(x)), dims),
                rotation=-45, ha='left')

            ax.grid(True)

            # save plot to file
            extent = ax.get_window_extent()
            extent = extent.transformed(plt.gcf().dpi_scale_trans.inverted())
            plt.savefig('report/resources/{}_boxplot.svg'.format(name),
                        bbox_inches=extent.expanded(1.5, 1.8))

    # display all plots
    plt.tight_layout()
    plt.show()

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

    # plot data
    plot_benchmarks(benchmarks)
