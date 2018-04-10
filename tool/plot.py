#!/usr/bin/env python3

import csv
import math
import os
import sys

import matplotlib.pyplot as plt

def plot_benchmarks(benchmarks):
    results = {}
    for i, bm in enumerate(benchmarks):
        name, data = bm

        for row in data:
            clusters = int(row['clusters'])

            if clusters not in results:
                results[clusters] = {}

            if name not in results[clusters]:
                results[clusters][name] = ([], [])

            results[clusters][name][0].append(int(row['dim']))
            results[clusters][name][1].append(1000 * float(row['time']))

    it = iter(sorted(results.items()))

    subplot_dim = math.ceil(math.sqrt(len(results)))
    for y in range(subplot_dim):
        for x in range(subplot_dim):
            p = y * subplot_dim + x + 1
            if p == len(results):
                break

            ax = plt.subplot(subplot_dim, subplot_dim, p)

            clusters, data = next(it)
            for name in data:
                dim, time = data[name]
                ax.plot(dim, time, label=name)

            ax.set_title('{} clusters'.format(clusters))
            ax.set_xlabel('image dimensions')
            ax.set_xticklabels(
                map(lambda x : '{d}x{d}'.format(d=int(x)), ax.get_xticks()))
            ax.set_ylabel('execution time in $\mu s$')
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    benchmarks = []

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

    plot_benchmarks(benchmarks)
