#include <opencv2/core/core.hpp>

#include "kmeans_config.h"
#include "kmeans_wrapper.h"

extern "C" {
#include "kmeans_c.h"
}

namespace kmeans {

void CWrapper::exec(cv::Mat const &image, cv::Mat const &initial_clusters) {

    // transform data points into **double format
    int n_data_points = image.rows * image.cols;

    double **data_points = new double*[n_data_points];
    for (int d = 0; d < n_data_points; ++d)
        data_points[d] = new double[3];

    for (int y = 0; y < image.rows; ++y) {
        for(int x = 0; x < image.cols; ++x) {
            int idx = y * image.cols + x;
            for (int i = 0; i < 3; ++i)
                data_points[idx][i] = image.at<cv::Vec3b>(y, x)[i];
        }
    }

    // transform cluster centers into **double format
    int n_clusters = initial_clusters.rows;

    double **clusters = new double*[n_clusters];
    for (int c = 0; c < n_clusters; ++c) {
        clusters[c] = new double[3];
        for (int i = 0; i < 3; ++i)
            clusters[c][i] = initial_clusters.at<cv::Vec3b>(c, 0)[i];
    }

    // allocate space for labels
    std::vector<int> labels(n_data_points);

    // perform calculations
    start_timer();
    kmeans_c(data_points, n_data_points, clusters, n_clusters, &labels[0]);
    stop_timer();

    // rebuild image from results
    result = cv::Mat(image.size(), image.type());
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int label = labels[y * image.cols + x];
            for (int i = 0; i < 3; ++i)
                result.at<cv::Vec3b>(y, x)[i] = clusters[label][i];
        }
    }

    // free heap memory
    for (int d = 0; d < n_data_points; ++d)
        delete[] data_points[d];
    delete[] data_points;

    for (int c = 0; c < n_clusters; ++c)
        delete[] clusters[c];
    delete[] clusters;
}

void OpenCVWrapper::exec(cv::Mat const &image, cv::Mat const &initial_clusters) {

    cv::Mat clusters = initial_clusters.clone();

    // construct input data points vector
    cv::Mat data_points(image.rows * image.cols, 3, CV_32F);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            for (int channel = 0; channel < 3; ++channel)
                data_points.at<float>(y *  image.cols + x, channel) =
                    image.at<cv::Vec3b>(y, x)[channel];
        }
    }

    // allocate space for labels
    cv::Mat labels;

    // specify termination criteria
    cv::TermCriteria term(CV_TERMCRIT_ITER, KMEANS_MAX_ITER, 0);

    // perform calculations
    start_timer();
    cv::kmeans(data_points, clusters.rows, labels, term, 1, 0, clusters);
    stop_timer();

    // rebuild image from results
    result = cv::Mat(image.size(), image.type());
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int idx = labels.at<size_t>(y * image.cols +  x, 0);
            for (int i = 0; i < 3; ++i)
                result.at<cv::Vec3b>(y, x)[i] = clusters.at<float>(idx, i);
        }
    }
}

}
