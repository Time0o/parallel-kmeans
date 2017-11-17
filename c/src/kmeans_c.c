#include <math.h> /* pow(), sqrt() */
#include <stdlib.h> /* malloc(), free() */
#include <string.h> /* memset() */

#include <omp.h>

#include "kmeans_config.h"
#include "kmeans_c.h"

void kmeans_c(double **data_points, int n_data_points,
              double **centroids, int n_centroids, int *labels)
{
    for (int iter = 0; iter < KMEANS_MAX_ITER; ++iter) {

        /* assign labels to data points */
        for (int d = 0; d < n_data_points; ++d) {
            double min_dist = -1.0;
            for (int c = 0; c < n_centroids; ++c) {
                double *dp = data_points[d];
                double *centr = centroids[c];

                double dist = sqrt(pow((dp[0] - centr[0]), 2) +
                                   pow((dp[1] - centr[1]), 2) +
                                   pow((dp[2] - centr[2]), 2));

                if (min_dist < 0 || dist < min_dist) {
                    labels[d] = c;
                    min_dist = dist;
                }
            }
        }

        /* null centroids */
        for (int c = 0; c < n_centroids; ++c) {
            centroids[c][0] = 0.0;
            centroids[c][1] = 0.0;
            centroids[c][2] = 0.0;
        }

        int *points_per_cluster = calloc(n_centroids, sizeof(int));

        /* sum data points for each centroid */
        for (int d = 0; d < n_data_points; ++d) {
            int label = labels[d];
            double *dp = data_points[d];
            double *centr = centroids[label];

            centr[0] += dp[0];
            centr[1] += dp[1];
            centr[2] += dp[2];

            points_per_cluster[label]++;
        }

        /* normalize centroids */
        for (int c = 0; c < n_centroids; ++c) {
            double *centr = centroids[c];
            int n_points = points_per_cluster[c];

            if (n_points != 0) {
                double norm_fact =  1.0 / n_points;
                centr[0] *= norm_fact;
                centr[1] *= norm_fact;
                centr[2] *= norm_fact;
            } else {
                int new_cluster = rand() % n_data_points;
                centr[0] = data_points[new_cluster][0];
                centr[1] = data_points[new_cluster][1];
                centr[2] = data_points[new_cluster][2];
            }
        }

        free(points_per_cluster);
    }

    for (int d = 0; d < n_data_points; ++d) {
        double *dp = data_points[d];
        double *centr = centroids[labels[d]];
        dp[0] = centr[0];
        dp[1] = centr[1];
        dp[2] = centr[2];
    }
}
