/*
 * ground.cpp
 *
 *  Created on: Mar 14, 2012
 *      Author: norton
 */

#include <ground.h>

const int depth::ground_truth::margin = 50;

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
  cv::Mat show;
  cv::Mat oth = img->source().clone();

  cv::namedWindow("segment", CV_WINDOW_AUTOSIZE);
  cv::namedWindow("source" , CV_WINDOW_AUTOSIZE);
  cvSetMouseCallback("segment", ground_truth::mouse_callback, this);
  cvSetMouseCallback("source" , ground_truth::mouse_callback, this);

  paint_ground(show, oth, img->source());

  cv::imshow("segment", show);
  cv::imshow("source" , oth);
  cv::waitKey(delay);
}

void depth::ground_truth::mouse_callback(int event, int x, int y, int flags,
    void* im) {
  ground_truth* truth = (ground_truth*)im;

  switch(event) {
    case /* move   */ 0:                               break;
    case /* l_down */ 1: truth->mouse_handle(y, x, 0); break;
    case /* r_down */ 2: truth->mouse_handle(y, x, 1); break;
    case /* m_down */ 3: truth->mouse_handle(y, x, 2); break;
    case /* l_up   */ 4:                               break;
    case /* r_up   */ 5:                               break;
    case /* m_up   */ 6:                               break;
  }
}

void depth::ground_truth::mouse_handle(int y, int x, int type) {
  static int state = 0;
  cv::Mat dst;
  cv::Mat oth;

  switch(state) {
    case 0:
      if(type == 0) {
        _lines.push_back(cv::Point(x, y));
        state = 1;
      } else if(type == 2) {
        _points[cv::Point(x - margin/2, y - margin/2)] = ground;
      }

      break;

    case 1:
      if(type == 0) {
        _lines.push_back(cv::Point(-1, -1));
        state = 0;
      } else if(type == 1) {
        _lines.push_back(cv::Point(x, y));
      }

      break;
  }

  paint_ground(dst, oth, img->source());
  cv::imshow("segment", dst);
  cv::imshow("source", oth);
}

void flood(cv::Mat& dst, cv::Mat& filled, int x, int y,
    depth::ground_truth::label type) {
  if(x < 0 || x >= dst.cols || y < 0 || y > dst.rows)
    return;
  if(filled.at<uint8_t>(y, x))
    return;
  if(dst.at<cv::Vec3b>(y, x) == cv::Vec3b(255, 255, 255))
    return;

  switch(type) {
    case depth::ground_truth::none:
      break;

    case depth::ground_truth::ground:
      dst.at<cv::Vec3b>(y, x)[0] = 255;
      break;

    case depth::ground_truth::vertical:
      dst.at<cv::Vec3b>(y, x)[1] = 255;
      break;

    case depth::ground_truth::sky:
      dst.at<cv::Vec3b>(y, x)[2] = 255;
      break;
  }

  filled.at<uint8_t>(y, x) = 1;

  flood(dst, filled, x + 1, y    , type);
  flood(dst, filled, x - 1, y    , type);
  flood(dst, filled, x    , y + 1, type);
  flood(dst, filled, x    , y - 1, type);
}

void depth::ground_truth::paint_ground(cv::Mat& dst, cv::Mat& oth, cv::Mat& src) {
  dst = cv::Mat::zeros(src.rows + margin, src.cols + margin, src.type());
  oth = src.clone();
  cv::Point last(-1, -1);

  for(int i = 0; i < src.rows; i++) {
    for(int j = 0; j < src.cols; j++) {
      dst.at<cv::Vec3b>(i + margin/2, j + margin/2) = src.at<cv::Vec3b>(i, j);
    }
  }

  for(const cv::Point& curr : _lines) {
    cv::Point o1 = last, o2 = curr;
    o1.x -= margin/2; o1.y -= margin/2;
    o2.x -= margin/2; o2.y -= margin/2;

    if(last.x != -1 && curr.x != -1) {
      cv::line(dst, last, curr, cv::Scalar(255, 255, 255));
      cv::line(oth, o1,   o2,   cv::Scalar(255, 255, 255));
    }

    last = curr;
  }

  for(std::pair<cv::Point, label>& lab : _points) {



  }
}

bool operator<(cv::Point& lhs, cv::Point& rhs) {
  int32_t l_total = 0;
  int32_t r_total = 0;

  l_total = lhs.x | (lhs.y << 16);
  r_total = rhs.x | (rhs.y << 16);

  return l_total < r_total;
}
