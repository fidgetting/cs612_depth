/*
 * knearest.h
 *
 *  Created on: Apr 17, 2012
 *      Author: norton
 */

#pragma once

#include <string>

#include <opencv2/opencv.hpp>

namespace depth {

  class knearest {
    public:

      knearest() { }
      knearest(cv::Mat& _data, cv::Mat& _resp, uint8_t _k);
      virtual ~knearest() { }

      virtual void load(std::string filename, std::string name);
      virtual void save(std::string filename, std::string name);

      virtual float predict(cv::Mat& point) const;

    private:

      cv::Mat _data;
      cv::Mat _resp;
      uint8_t _k;
      uint8_t _n;
  };

}
