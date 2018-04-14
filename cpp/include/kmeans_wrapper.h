#pragma once

#include <omp.h>
#include <opencv2/core/core.hpp>

extern "C" {
#include "kmeans.h"
}

class KmeansWrapper
{
public:
    virtual void exec(cv::Mat const &image, size_t n_centroids) = 0;

    cv::Mat get_result() { return result; };
    double get_exec_time() { return _exec_time; };

    virtual ~KmeansWrapper() {}

protected:
    void start_timer() { _start_time = omp_get_wtime(); };
    void stop_timer() { _exec_time = (double) (omp_get_wtime() - _start_time); }
    cv::Mat result;

private:
    double _start_time;
    double _exec_time;
};

class KmeansOpenCVWrapper : public KmeansWrapper
{
public:
    KmeansOpenCVWrapper() {}
    void exec(cv::Mat const &image, size_t n_clusters);
};

class KmeansCWrapper : public KmeansWrapper
{
public:
    KmeansCWrapper(
        void (*impl)(struct pixel *, size_t, struct pixel *, size_t, size_t *))
            : impl(impl) {}

    void exec(cv::Mat const &image, size_t n_clusters);

protected:
    void (*impl)(struct pixel *, size_t, struct pixel *, size_t, size_t *);
};

class KmeansOMPWrapper : public KmeansCWrapper
{
public:
    KmeansOMPWrapper() : KmeansCWrapper(kmeans_omp) {}
};

class KmeansPureCWrapper : public KmeansCWrapper
{
public:
    KmeansPureCWrapper() : KmeansCWrapper(kmeans) {}
};
