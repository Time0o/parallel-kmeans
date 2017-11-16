#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

extern "C" {
  #include "kmeans.h"
}

#define MAX_CLUSTERS 10
#define MAX_ITER 10000
#define EPS_STATIONARY 0.0001

int main(int argc, char **argv)
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " IMAGE N_CLUSTERS\n"; 
    return 1;
  }
  
  // load image
  cv::Mat image = cv::imread(argv[1]);
  if (image.empty()) {
    std::cerr << "Failed to load image file '" << argv[1] << "'\n";
    return 2;
  }

  int n_clusters;
  try {
    size_t idx;
    n_clusters = std::stoi(argv[2], &idx);

    if (idx != strlen(argv[2]))
      throw std::invalid_argument("trailing garbage");
    else if (n_clusters <= 0 || n_clusters > MAX_CLUSTERS)
      throw std::invalid_argument("must be in range [1,  "
                                  + std::to_string(MAX_CLUSTERS) + "]");

  } catch (std::exception const &e) {
    std::cerr << "Could not evaluate number of clusters: " << e.what() << "\n";
    return 3;
  }

  // for measuring execution time
  clock_t start, end;
  double exec_time_opencv;

//=============================================================================

  // OpenCV reference implementation
  cv::Mat opencv_data_points(image.rows * image.cols, 3, CV_32F);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      for (int channel = 0; channel < 3; ++channel)
        opencv_data_points.at<float>(y *  image.cols + x, channel) =
          image.at<cv::Vec3b>(y, x)[channel];
    }
  }

  cv::Mat opencv_labels, opencv_centers;
  cv::TermCriteria opencv_term(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, MAX_ITER,
                               EPS_STATIONARY);

  start = clock();
  cv::kmeans(opencv_data_points, n_clusters, opencv_labels, opencv_term, 1,
             cv::KMEANS_RANDOM_CENTERS, opencv_centers);
  end = clock();
  exec_time_opencv = (double)(end - start) / CLOCKS_PER_SEC;

  cv::Mat opencv_clustered(image.size(), image.type());
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
     size_t idx = opencv_labels.at<size_t>(y * image.cols +  x, 0);
     opencv_clustered.at<cv::Vec3b>(y, x)[0] =
       opencv_centers.at<float>(idx, 0);
     opencv_clustered.at<cv::Vec3b>(y, x)[1] =
       opencv_centers.at<float>(idx, 1);
     opencv_clustered.at<cv::Vec3b>(y, x)[2] =
       opencv_centers.at<float>(idx, 2);
    }
  }

  // C reference implementation
  // TODO

  // C reference implementation using OpenMP
  // TODO

  // CUDA implementation
  // TODO

//=============================================================================

  // display results
  std::vector<cv::Mat> win_panes { opencv_clustered };
  std::vector<std::string> pane_titles { "OpenCV" };
  std::vector<double> exec_times { exec_time_opencv };

  int disp_width = image.size().width;
  int disp_height = image.size().height;
  int const win_margin = 10;

  cv::Mat win_mat(
    cv::Size((disp_width + win_margin) * win_panes.size(), disp_height + 50),
    CV_8UC3, cv::Scalar(0, 0, 0)
  );

  int offs = win_margin / 2;
  for (size_t i = 0; i < win_panes.size(); ++i) {
    win_panes[i].copyTo(win_mat(cv::Rect(offs, 0, disp_width, disp_height)));

    std::ostringstream title_text;
    title_text << pane_titles[i] << " (" << std::setprecision(4)
               << exec_times[i] << " sec.)";

    cv::putText(win_mat, title_text.str(),
                cvPoint(offs + win_margin, disp_height + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255),
                1.0, CV_AA);

    offs += disp_width + win_margin;
  }

  std::string disp_title(argv[1]);
  disp_title += " (clustered)";

  cv::namedWindow(disp_title, cv::WINDOW_AUTOSIZE);
  cv::imshow(disp_title, win_mat);
  cv::waitKey(0);
  cv::destroyWindow(disp_title);

  return 0;
}
