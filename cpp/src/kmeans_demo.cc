#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "kmeans_config.h"
#include "kmeans_wrapper.h"

int main(int argc, char **argv)
{
    cv::Mat image, initial_clusters;
    int n_clusters;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " IMAGE CLUSTERS\n";
        return 1;
    }

    // load image
    image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Failed to load image file '" << argv[1] << "'\n";
        return 2;
    }

    // parse number of clusters
    try {
        size_t idx;
        n_clusters = std::stoi(argv[2], &idx);

        if (idx != strlen(argv[2]))
            throw std::invalid_argument("trailing garbage");
        else if (n_clusters <= 0 || n_clusters > KMEANS_MAX_CLUSTERS)
            throw std::invalid_argument(
                "must be in range [1, " +
                std::to_string(KMEANS_MAX_CLUSTERS) + "]");

    } catch (std::exception const &e) {
        std::cerr << "Failed to parse number of clusters: " << e.what() << "\n";
        return 3;
    }

    // initialize clusters to random color values
    initial_clusters = cv::Mat(n_clusters, 1, CV_8UC3);
    for (int c = 0; c < n_clusters; ++c) {
        int row = rand() % image.rows;
        int col = rand() % image.cols;
        initial_clusters.at<cv::Vec3b>(c, 0) = image.at<cv::Vec3b>(row, col);
    }

    // initialize implementation variants
    std::vector<std::pair<char const *, kmeans::Wrapper *>> impl;

    kmeans::CWrapper c_wrapper;
    impl.push_back(std::make_pair("Pure C", &c_wrapper));

    kmeans::OpenCVWrapper opencv_wrapper;
    impl.push_back(std::make_pair("OpenCV", &opencv_wrapper));

    // setup results display window
    const int margin = 10;

    cv::Mat win_mat(
        cv::Size((image.cols + margin) * impl.size(), image.rows + 50),
        CV_8UC3, cv::Scalar(0, 0, 0)
    );

    int offs = margin / 2;
    for (auto const &pane: impl) {
        char const *title = std::get<0>(pane);

        // get result and execution time for each wrapper
        kmeans::Wrapper *wrapper = std::get<1>(pane);
        wrapper->exec(image, initial_clusters);
        cv::Mat result = wrapper->get_result();
        double exec_time = wrapper->get_exec_time();

        result.copyTo(win_mat(cv::Rect(offs, 0, image.cols, image.rows)));

        // construct subtitles
        std::ostringstream subtext;
        subtext << title << " (" << std::setprecision(2) << exec_time
                             << " sec.)";

        cv::putText(win_mat, subtext.str(),
                    cvPoint(offs + margin, image.rows + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255),
                    1.0, CV_AA);

        offs += image.cols + margin;
    }

    image.release();
    initial_clusters.release();

    // display window
    std::string disp_title(argv[1]);
    cv::namedWindow(disp_title, cv::WINDOW_AUTOSIZE);
    cv::imshow(disp_title, win_mat);
    cv::waitKey(0);
    cv::destroyWindow(disp_title);

    win_mat.release();

    return 0;
}
