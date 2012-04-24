/*
 * segment.h
 *
 *  Created on: Feb 14, 2012
 *      Author: norton
 */

#pragma once

#include <image.h>
#include <misc.h>
#include <segment-image.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iomanip>
#include <vector>

#include <boost/thread/mutex.hpp>

#include <sys/types.h>
#include <sys/socket.h>

#define print_mat(gaus, type)                       \
    std::cout << std::fixed;                        \
    std::cout << #gaus << "= \n\n";                 \
    for(int ii = 0; ii < gaus.rows; ii++) {         \
      std::cout << "  ";                            \
      for(int jj = 0; jj < gaus.cols; jj++)  {      \
        std::cout << gaus.at<type>(ii, jj) << " ";  \
      }                                             \
      std::cout << "\n";                            \
    }                                               \
    std::cout << std::endl

#define EPSILON 1e-10
#define PI 3.1415926535

namespace depth {

  typedef unsigned char  uc;
  typedef unsigned short us;

  /* utility */
  std::string type_to_string(const cv::Mat& img);

  /* segmentation */
  void cv_to_img(image<rgb>* dst, cv::Mat& src);
  void img_to_cv(cv::Mat& dst, image<rgb>* src);
  void segment(cv::Mat& src, cv::Mat& dst);

  /* doog filters */
  cv::Mat createGaussian(double x0, double y0, double sigX, double sigY, int size);
  cv::Mat createDOOG(double sigma, double r, double theta, int size);
  std::vector<cv::Mat> getFilters(double sigma, double r, double step, int size);
  std::vector<cv::Mat> laws();

  template<typename _t>
  double mean(_t begin, _t end) {
    double sum;
    int count = end - begin;

    for(; begin != end; begin++) {
      sum += *begin;
    }

    return sum / count;
  }

  template<typename _t>
  double median(_t begin, _t end) {
    int diff = end - begin;

    if((diff)%2 == 0) {
      return (*(begin + (diff/2)) + *(end - (diff/2)))/2.0;
    }

    return *(begin + (diff/2));
  }

  double v_sum(const std::vector<double>& vec);
  std::vector<double> v_pow(const std::vector<double>& vec, int pow);
  std::vector<double> operator*(const std::vector<double>& lhs,
      const std::vector<double>& rhs);

  template<typename _t, typename iter_t>
  double append(std::vector<_t>& vec, iter_t begin, iter_t end) {
    std::for_each(begin, end, [&vec](_t& elem) { vec.push_back(elem); });
  }
}



