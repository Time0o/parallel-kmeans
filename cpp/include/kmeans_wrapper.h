#pragma once

#include <omp.h>
#include <opencv2/core/core.hpp>

namespace kmeans {

class Wrapper
{
public:
    Wrapper() {}
    virtual ~Wrapper() {}

    virtual void exec(cv::Mat const &image, cv::Mat const &initial_clusters) = 0;

    cv::Mat get_result() { return result; };
    double get_exec_time() { return _exec_time; };

protected:
    void start_timer() { _start_time = omp_get_wtime(); };
    void stop_timer() { _exec_time = (double) (omp_get_wtime() - _start_time); }
    cv::Mat result;

private:
    double _start_time;
    double _exec_time;
};

class CWrapper : public Wrapper
{
public:
    CWrapper() {}
    void exec(cv::Mat const &image, cv::Mat const &initial_clusters);
};

class OpenCVWrapper : public Wrapper
{
public:
    OpenCVWrapper() {}
    void exec(cv::Mat const &image, cv::Mat const &initial_clusters);
};

}
