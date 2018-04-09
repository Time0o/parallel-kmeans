#include <opencv2/core/core.hpp>

#include "kmeans_config.h"
#include "kmeans_wrapper.h"

extern "C" {
#include "kmeans_c.h"
}

namespace kmeans {

void CWrapper::exec(cv::Mat const &image, int n_clusters) {

    // transform data points into **double format
    int n_data_points = image.rows * image.cols;

    std::vector<double> data_points(3 * n_data_points);
    for (int y = 0; y < image.rows; ++y) {
        for(int x = 0; x < image.cols; ++x) {
            int idx = 3 * (y * image.cols + x);
            for (int i = 0; i < 3; ++i)
                data_points[idx + i] = image.at<cv::Vec3b>(y, x)[i];
        }
    }

    // allocate space for centroids
    std::vector<double> centroids(3 * n_clusters);

    // allocate space for labels
    std::vector<int> labels(n_data_points);

    // perform calculations
    start_timer();
    kmeans_c(&data_points[0], n_data_points,
             &centroids[0], n_clusters, &labels[0]);
    stop_timer();

    // rebuild image from results
    result = cv::Mat(image.size(), image.type());
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int idx = 3 * labels[y * image.cols + x];
            for (int i = 0; i < 3; ++i)
                result.at<cv::Vec3b>(y, x)[i] = centroids[idx + i];
        }
    }
}

void C2DWrapper::exec(cv::Mat const &image, int n_clusters) {

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

    // allocate space for centroids
    double **centroids = new double*[n_clusters];
    for (int c = 0; c < n_clusters; ++c)
        centroids[c] = new double[3];

    // allocate space for labels
    std::vector<int> labels(n_data_points);

    // perform calculations
    start_timer();
    kmeans_c2d(data_points, n_data_points, centroids, n_clusters, &labels[0]);
    stop_timer();

    // rebuild image from results
    result = cv::Mat(image.size(), image.type());
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int label = labels[y * image.cols + x];
            for (int i = 0; i < 3; ++i)
                result.at<cv::Vec3b>(y, x)[i] = centroids[label][i];
        }
    }

    // free heap memory
    for (int d = 0; d < n_data_points; ++d)
        delete[] data_points[d];
    delete[] data_points;

    for (int c = 0; c < n_clusters; ++c)
        delete[] centroids[c];
    delete[] centroids;
}

void OpenCVWrapper::exec(cv::Mat const &image, int n_clusters) {

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

    // transform cluster centers into **double format
    cv::Mat centroids;

    // specify termination criteria
    cv::TermCriteria term(CV_TERMCRIT_ITER, KMEANS_MAX_ITER, 0);

    // perform calculations
    start_timer();
    cv::kmeans(data_points, n_clusters, labels, term, 1,
               cv::KMEANS_RANDOM_CENTERS, centroids);
    stop_timer();

    // rebuild image from results
    result = cv::Mat(image.size(), image.type());
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int idx = labels.at<int>(y * image.cols +  x, 0);
            for (int i = 0; i < 3; ++i)
                result.at<cv::Vec3b>(y, x)[i] = centroids.at<float>(idx, i);
        }
    }
}

}
