/*
 * ground.cpp
 *
 *  Created on: Mar 14, 2012
 *      Author: norton
 */

#include <ground.h>

depth::ground_truth::ground_truth(std::shared_ptr<processed_image> img) :
    img(img),
    _labels(img->npixels()) { }

depth::ground_truth::ground_truth(std::shared_ptr<processed_image> img,
    fs::path file) :
    img(img),
    _labels(img->npixels()) {
  std::ifstream istr(file.c_str());
  std::string name;
  int curr;

  istr >> name;
  for(int i = 0; i < img->npixels(); i++) {
    istr >> curr;
    _labels[i] = (label)curr;
  }
}

void depth::ground_truth::display(int delay) {
  cv::Mat result = cv::Mat::zeros(img->source().size(), CV_8UC3);
  cv::Mat dst(img->source().size(), CV_8UC3);

  img->source().copyTo(dst);

  cv::namedWindow("segment", CV_WINDOW_AUTOSIZE);
  cvSetMouseCallback("segment", ground_truth::mouse_callback, this);

  paint_ground(dst, result);

  cv::imshow("source", dst);
  cv::imshow("segment", img->segimg());
  cv::imshow("ground",  result);
  cv::waitKey(delay);
}

void depth::ground_truth::mouse_callback(int event, int x, int y, int flags,
    void* im) {
  ground_truth* truth = (ground_truth*)im;

  switch(event) {
    case /* move   */ 0:                            break;
    case /* l_down */ 1: truth->mouse_handle(y, x); break;
    case /* r_down */ 2:                            break;
    case /* m_down */ 3:                            break;
    case /* l_up   */ 4:                            break;
    case /* r_up   */ 5:                            break;
    case /* m_up   */ 6:                            break;
  }
}

void depth::ground_truth::mouse_handle(int x, int y) {
  int segm = img->segment().at<int>(x, y);
  cv::Mat result = cv::Mat::zeros(img->source().size(), CV_8UC3);
  cv::Mat dst(img->source().size(), CV_8UC3);

  img->source().copyTo(dst);

  _labels.at(segm) = (label)((int(_labels.at(segm)) + 1) % 4);
  paint_ground(dst, result);

  cv::imshow("source", dst);
  cv::imshow("ground", result);
}

void depth::ground_truth::paint_ground(cv::Mat& dst, cv::Mat& simp) {
  for(int i = 0; i < simp.rows; i++) {
    for(int j = 0; j < simp.cols; j++) {
      int l = _labels[img->segment().at<int>(i, j)];

      switch(l) {
        case unlabled:
          simp.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0  );
          break;

        case ground:
          dst.at<cv::Vec3b>(i, j)[0] = 255;
          simp.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
          break;

        case vertical:
          dst.at<cv::Vec3b>(i, j)[1] = 0;
          simp.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
          break;

        case sky:
          dst.at<cv::Vec3b>(i, j)[2] = 0;
          simp.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
          break;
      }
    }
  }
}
