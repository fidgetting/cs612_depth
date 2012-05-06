/*
 * knearest.cpp
 *
 *  Created on: Apr 17, 2012
 *      Author: norton
 */

#include <cfloat>
#include <set>

#include <knearest.h>

depth::knearest::knearest(cv::Mat& _data, cv::Mat& _resp, uint8_t _k) :
_data(_data), _resp(_resp), _k(_k) {
  std::set<int> classes;

  for(int i = 0; i < _resp.rows; i++) {
    if(classes.find(_resp.at<int32_t>(i, 0)) == classes.end()) {
      classes.insert(_resp.at<int32_t>(i, 0));
    }
  }

  _n = classes.size();
}

void depth::knearest::load(std::string filename, std::string name) {
  cv::FileStorage global(filename, cv::FileStorage::READ);
  cv::FileNode file = global.root()[name];
  int rows, cols;
  std::string type;

  _k = int(file["knearest"]);
  _n = int(file["nclasses"]);

  cv::FileNode datanode = file["data"];
  cv::FileNode respnode = file["response"];

  rows = int(datanode["rows"]);
  cols = int(datanode["cols"]);
  type = std::string(datanode["dt"]);

  _data = cv::Mat::zeros(rows, cols, CV_32FC1);
  datanode["data"].readRaw(type, _data.data, rows * cols);

  rows = int(respnode["rows"]);
  cols = int(respnode["cols"]);
  type = std::string(respnode["dt"]);

  _resp = cv::Mat::zeros(rows, cols, CV_32FC1);
  respnode["data"].readRaw(type, _resp.data, rows * cols);
}

void depth::knearest::save(std::string filename, std::string name) {
  cv::FileStorage file(filename, cv::FileStorage::WRITE);
  CvFileStorage* store = *file;

  /* start the classifier section */
  cvStartWriteStruct(store, name.c_str(), CV_NODE_MAP);
  cvWriteInt(store, "knearest", _k);
  cvWriteInt(store, "nclasses", _n);

  /* start the data section */
  cvStartWriteStruct(store, "data", CV_NODE_MAP, "opencv-matrix");
  cvWriteInt(store, "rows", _data.rows);
  cvWriteInt(store, "cols", _data.cols);
  cvWriteString(store, "dt", "f");

  cvStartWriteStruct(store, "data", CV_NODE_SEQ);
  cvWriteRawData(store, _data.data, _data.rows * _data.cols, "f");
  cvEndWriteStruct(store);
  /* end the data section */
  cvEndWriteStruct(store);

  /* start the response section */
  cvStartWriteStruct(store, "response", CV_NODE_MAP, "opencv-matrix");
  cvWriteInt(store, "rows", _resp.rows);
  cvWriteInt(store, "cols", _resp.cols);
  cvWriteString(store, "dt", "i");

  cvStartWriteStruct(store, "data", CV_NODE_SEQ);
  cvWriteRawData(store, _resp.data, _resp.rows * _resp.cols, "i");
  cvEndWriteStruct(store);
  /* end the data section */
  cvEndWriteStruct(store);

  /* end the classifier section */
  cvEndWriteStruct(store);
}

float depth::knearest::predict(cv::Mat& point) const {
  std::vector<std::pair<int, double> > best;
  std::vector<uint8_t> nfound(_n, 0);
  double dist;
  int max = 0;

  for(int i = 0; i < _k; i++) {
    std::pair<int, double> pair(-1, DBL_MAX);
    best.push_back(pair);
  }

  for(int i = 0; i < _data.rows; i++) {
    dist = 0;

    /* calculate simple Manhattan distance */
    for(int j = 0; j < _data.cols; j++)
      dist += pow(point.at<float>(0, j) - _data.at<float>(i, j), 2);
    dist = sqrt(dist);

    for(std::pair<int, double>& curr : best) {
      if(dist < curr.second) {
        curr.first = i;
        curr.second = dist;
        break;
      }
    }
  }

  for(std::pair<int, double>& curr : best) {
    /*std::cout << "[" << curr.first << ", " << curr.second << ", "
        << _resp.at<int32_t>(curr.first) << "] ";*/
    nfound[_resp.at<int32_t>(curr.first)]++;
  }

  for(int i = 1; i < nfound.size(); i++) {
    if(nfound[i] > nfound[max]) {
      max = i;
    }
  }

  return max;
}

