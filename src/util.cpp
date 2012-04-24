/*
 * segment.cpp
 *
 *  Created on: Feb 14, 2012
 *      Author: norton
 */

/* local includes */
#include <util.h>

/* stl includes */
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <map>

/* networking includes */
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/stat.h>

/* ************************************************************************** */
/* *** utility ************************************************************** */
/* ************************************************************************** */

/**
 * Gets a string representation of the type of the matrix
 *
 * @param img the matrix to get the type for
 * @return string representation of the type
 */
std::string depth::type_to_string(const cv::Mat& img) {
#define CASE_STMT(type) case type: return #type

  switch(img.type()) {
    CASE_STMT(CV_8UC1);
    CASE_STMT(CV_8UC2);
    CASE_STMT(CV_8UC3);
    CASE_STMT(CV_8UC4);

    CASE_STMT(CV_16UC1);
    CASE_STMT(CV_16UC2);
    CASE_STMT(CV_16UC3);
    CASE_STMT(CV_16UC4);

    CASE_STMT(CV_16SC1);
    CASE_STMT(CV_16SC2);
    CASE_STMT(CV_16SC3);
    CASE_STMT(CV_16SC4);

    CASE_STMT(CV_32SC1);
    CASE_STMT(CV_32SC2);
    CASE_STMT(CV_32SC3);
    CASE_STMT(CV_32SC4);

    CASE_STMT(CV_32FC1);
    CASE_STMT(CV_32FC2);
    CASE_STMT(CV_32FC3);
    CASE_STMT(CV_32FC4);

    CASE_STMT(CV_64FC1);
    CASE_STMT(CV_64FC2);
    CASE_STMT(CV_64FC3);
    CASE_STMT(CV_64FC4);
  }

#undef CASE_STMT
}

/* ************************************************************************** */
/* *** segmenation ********************************************************** */
/* ************************************************************************** */

/**
 * The segmentation code that I'm using uses its own image class. This
 * transfers the data from the opencv image format into the image format used
 * by the segmentation algorithm.
 *
 * @param src  the image to get the data from
 * @return  a new image pointer that can be segmented
 */
void depth::cv_to_img(image<rgb>* img, cv::Mat& src) {

  if(src.rows != img->height() || src.cols != img->width()) {
    throw std::exception();
  }

  rgb* dst = img->data;
  for(auto iter  = src.begin<cv::Vec3b>();
           iter != src.end<cv::Vec3b>(); iter++, dst++) {
    dst->r = (*iter)[0];
    dst->g = (*iter)[1];
    dst->b = (*iter)[2];
  }
}

/**
 * The segmentation code that I'm using uses its own image class. This
 * transfers the data from the image formated used during semgentation back into
 * the image format used by opencv.
 *
 * @param dst  the opencv image to put the data in
 * @param img  the segmentation image format to get the data from
 */
void depth::img_to_cv(cv::Mat& dst, image<rgb>*  img) {

  if(dst.rows != img->height() || dst.cols != img->width()) {
    throw std::exception();
  }

  rgb* src = img->data;
  for(auto iter  = dst.begin<cv::Vec3b>();
           iter != dst.end<cv::Vec3b>(); iter++, src++) {
    (*iter)[0] = src->r;
    (*iter)[1] = src->g;
    (*iter)[2] = src->b;
  }
}

/**
 * performs some form of image segmentation on the input image. This currently
 * uses the super pixel algorithm to perform the image segmentation.
 *
 * @param src  the image that should be segmented
 * @param dst  the destination that the image will be saved to
 */
void depth::segment(cv::Mat& src, cv::Mat& dst) {
  image<rgb>* input = NULL;
  image<rgb>* seg   = NULL;

  const float sigma = 3.0;
  const float k     = 100.0;
  const int   min   = 100;

  int num_ccs;

  input = new image<rgb>(src.cols, src.rows);
  cv_to_img(input, src);
  seg = segment_image(input, sigma, k, min, &num_ccs);
  img_to_cv(dst, seg);

  delete input;
  delete seg;
}

/* ************************************************************************** */
/* *** segmenation ********************************************************** */
/* ************************************************************************** */

#define norm_to_sum(sum, mat)                             \
    std::for_each(mat.begin<double>(), mat.end<double>(), \
        [&sum](double& d) { sum += d; } );                \
    std::for_each(mat.begin<double>(), mat.end<double>(), \
        [&sum](double& d) { d /= sum; } )

/**
 * TODO
 *
 * @param x0
 * @param y0
 * @param sigX
 * @param sigY
 * @param size
 * @return
 */
cv::Mat depth::createGaussian(double x0, double y0, double sigX, double sigY, int size) {
  int radius = (size - 1)/2;
  double sum = 0;
  cv::Mat ret(size, size, CV_64F);

  for(int x = -radius; x <= radius; x++) {
    for(int y = -radius; y <= radius; y++) {
      ret.at<double>(x + radius, y + radius) =
          1 / (2 * PI * sigX * sigY) *
          std::exp(
              (-1 * std::pow(x - x0, 2)) / (2 * std::pow(sigX, 2)) -
              std::pow(y - y0, 2) / (2 * std::pow(sigY, 2)));
    }
  }

  norm_to_sum(sum, ret);
  return ret;
}

/**
 * TODO
 *
 * @param sigma
 * @param r
 * @param theta
 * @param size
 * @return
 */
cv::Mat depth::createDOOG(double sigma, double r, double theta, int size) {
  double sigmaX = r*sigma;
  double sigmaY = sigma;
  double sum;

  const double a = -1;
  const double b = 2;
  const double c = 1;

  /* get the gaussians */
  cv::Mat g1 = createGaussian(0,  sigma, sigmaX, sigmaY, size);
  cv::Mat g2 = createGaussian(0,      0, sigmaX, sigmaY, size*4 + 1);
  cv::Mat g3 = createGaussian(0, -sigma, sigmaX, sigmaY, size);
  cv::Mat ret(g2.size(), g2.type());

  /* rotate the middle gaussian */
  cv::warpAffine(
      g2,
      ret,
      cv::getRotationMatrix2D(
          cv::Point2f(double(g2.rows)/2.0 - 0.5, double(g2.cols)/2.0 - 0.5),
          theta,
          1),
      g2.size(),
      cv::INTER_CUBIC);
  int i1 = (4 * size+1)/2 - size/2;
  int i2 = i1 + size;
  ret = cv::Mat(ret, cv::Range(i1, i2), cv::Range(i1, i2));
  norm_to_sum(sum, ret);

  /* add all the filters together */
  ret = (a * g1) + (b * ret) + (c * g3);
  norm_to_sum(sum, ret);

  return ret;
}

/**
 * TODO
 *
 * @return
 */
std::vector<cv::Mat> depth::getFilters(double sigma, double r, double step, int size) {
  std::vector<cv::Mat> ret;

  for(double theta = 0; theta < 180; theta += step) {
    ret.push_back(createDOOG(sigma, r, theta, size));
  }

  return ret;
}

std::vector<cv::Mat> depth::laws() {
  static cv::Matx33f L3L3(1,  2,  1,  2,  4,  2,  1,  2,  1);
  static cv::Matx33f L3E3(1,  0, -1,  2,  0, -2,  1,  0, -1);
  static cv::Matx33f L3S3(1, -1,  1,  2, -2,  2,  1, -1,  1);

  static cv::Matx33f E3L3 = L3E3.t();
  static cv::Matx33f E3E3(1,  0, -1,  0,  0,  0, -1,  0,  1);
  static cv::Matx33f E3S3(1, -1,  1,  0,  0,  0, -1,  1, -1);

  static cv::Matx33f S3L3 = L3S3.t();
  static cv::Matx33f S3E3 = E3S3.t();
  static cv::Matx33f S3S3(1, -1,  1, -1,  1, -1,  1, -1,  1);

  static std::vector<cv::Mat> ret;

  if(ret.size() == 0) {
    ret.push_back(cv::Mat(L3L3));
    ret.push_back(cv::Mat(L3E3));
    ret.push_back(cv::Mat(L3S3));

    ret.push_back(cv::Mat(E3L3));
    ret.push_back(cv::Mat(E3E3));
    ret.push_back(cv::Mat(E3S3));

    ret.push_back(cv::Mat(S3L3));
    ret.push_back(cv::Mat(S3E3));
    ret.push_back(cv::Mat(S3S3));
  }

  return ret;
}

/* ************************************************************************** */
/* *** other **************************************************************** */
/* ************************************************************************** */

double depth::v_sum(const std::vector<double>& vec) {
  double sum = 0;

  std::for_each(vec.begin(), vec.end(), [&sum](double d) { sum += d; } );

  return sum;
}

std::vector<double> depth::v_pow(const std::vector<double>& vec, int exp) {
  std::vector<double> ret(vec.size());

  for(int i = 0; i < vec.size(); i++) {
    ret[i] = pow(vec[i], exp);
  }

  return ret;
}

std::vector<double> depth::operator*(const std::vector<double>& lhs,
    const std::vector<double>& rhs) {
  std::vector<double> ret(lhs.size());

  for(int i = 0; i < lhs.size(); i++) {
    ret[i] = lhs[i] * rhs[i];
  }

  return ret;
}



