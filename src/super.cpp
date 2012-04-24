/*
 * super.cpp
 *
 *  Created on: Feb 20, 2012
 *      Author: norton
 */

#include <super.h>

#include <algorithm>
#include <cmath>
#include <exception>

/* ************************************************************************** */
/* *** super_pixel ********************************************************** */
/* ************************************************************************** */

/**
 * TODO
 *
 * @param lhs
 * @param rhs
 * @return
 */
bool operator<(std::pair<int, int> lhs, std::pair<int, int> rhs) {
  return (lhs.first + (lhs.second << 16)) < (rhs.first + (rhs.second << 16));
}

depth::super_pixel::super_pixel(const processed_image* src) :
    _src(src), _sp_num(-1), _n_pix(0), _mean_r(0), _mean_g(0), _mean_b(0),
    _mean_hue(0), _mean_sat(0), _hist_hue(hue_buckets), _hist_sat(sat_buckets),
    _mean_x(0), _mean_y(0), _laws_resp(law_buckets) { }

/**
 * TODO
 *
 * @param src
 * @param sp_num
 */
depth::super_pixel::super_pixel(const processed_image* src, int32_t sp_num) :
    _src(src), _sp_num(sp_num), _n_pix(0), _mean_r(0), _mean_g(0), _mean_b(0),
    _mean_hue(0), _mean_sat(0), _hist_hue(hue_buckets), _hist_sat(sat_buckets),
    _mean_x(0), _mean_y(0), _laws_resp(law_buckets) {

  /* do the easy stuff first:
   *   mean      red, green, blue
   *   mean      hue, saturation
   *   mean      x, y
   *   histogram hue, saturation
   */
  std::set<std::pair<int, int> > pixels;
  double _row_norm = 1.0 / double(_src->rows());
  double _col_norm = 1.0 / double(_src->cols());

  cv::Mat hsv(_src->source().size(), CV_8UC3);
  cv::cvtColor(_src->source(), hsv, CV_RGB2HSV);

  for(int i = 0; i < _src->rows(); i++) {
    for(int j = 0; j < _src->cols(); j++) {
      if(_src->segment().at<int>(i, j) == _sp_num) {
        pixels.insert(std::pair<int, int>(i, j));

        _mean_x += j;
        _mean_y += j;

        _mean_r += _src->source().at<cv::Vec3b>(i, j)[0];
        _mean_g += _src->source().at<cv::Vec3b>(i, j)[1];
        _mean_b += _src->source().at<cv::Vec3b>(i, j)[2];

        _mean_hue += hsv.at<cv::Vec3b>(i, j)[0];
        _mean_sat += hsv.at<cv::Vec3b>(i, j)[1];

        double hue = (hsv.at<cv::Vec3b>(i, j)[0] / 256.0) * hue_buckets;
        double sat = (hsv.at<cv::Vec3b>(i, j)[1] / 256.0) * sat_buckets;

        _hist_hue[int(floor(hue))]++;
        _hist_sat[int(floor(sat))]++;

        _n_pix++;
      }
    }
  }

  _mean_x /= _n_pix;
  _mean_y /= _n_pix;

  _mean_r /= _n_pix;
  _mean_g /= _n_pix;
  _mean_b /= _n_pix;

  _mean_hue /= _n_pix;
  _mean_sat /= _n_pix;

  for(double& d : _hist_hue) d /= _n_pix;
  for(double& d : _hist_sat) d /= _n_pix;
}

/**
 * TODO
 *
 * @param row
 */
void depth::super_pixel::pack(cv::Mat& row) const {
  int idx = base;

  if(row.rows != 1 || row.cols != n_dim) {
    throw std::exception();
  }

  row.at<float>(0 ) = _n_pix;
  row.at<float>(1 ) = _mean_r;
  row.at<float>(2 ) = _mean_g;
  row.at<float>(3 ) = _mean_b;
  row.at<float>(4 ) = _mean_x;
  row.at<float>(5 ) = _mean_y;
  row.at<float>(6 ) = _mean_hue;
  row.at<float>(7 ) = _mean_sat;

  for(int i = 0; i < hue_buckets; i++)
    row.at<float>(idx++) = _hist_hue[i];
  for(int i = 0; i < sat_buckets; i++)
    row.at<float>(idx++) = _hist_sat[i];
}

/**
 * TODO
 *
 * @param lhs
 * @param rhs
 * @return
 */
bool depth::operator <(const depth::super_pixel& lhs,
    const depth::super_pixel& rhs) {
  return lhs.sp_num() < rhs.sp_num();
}

/**
 * TODO
 *
 * @param ostr
 * @param pix
 * @return
 */
#define printf_vec(ostr, vec)                                 \
    ostr << #vec << "[";                                      \
    for(auto iter = vec.begin(); iter != vec.end(); iter++) { \
      ostr << std::setw(6) << std::right << *iter;            \
      if(iter != pix.hist_hue().end() - 1) {                  \
        ostr << ", ";                                         \
      }                                                       \
    }                                                         \
    ostr << "] "

/**
 * TODO
 *
 * @param ostr
 * @param pix
 * @return
 */
std::ostream& depth::operator<<(std::ostream& ostr, const super_pixel& pix) {
  ostr << std::fixed
       << std::setfill(' ')
       << std::setprecision(3);

  ostr << "size: " << std::setw(6) << std::right << pix.n_pix() <<
          " rgb["  << pix.mean_r()   <<
          ", "     << pix.mean_g()   <<
          ", "     << pix.mean_b()   <<
          "] hs["  << pix.mean_hue() <<
          ", "     << pix.mean_sat() << "] ";

  printf_vec(ostr, pix.hist_hue());
  printf_vec(ostr, pix.hist_sat());

  ostr << " xy[" << pix.mean_x() << ", " << pix.mean_y() << "] ";

  printf_vec(ostr, pix.laws_resp());

  return ostr;
}

#undef printf_vec

/* ************************************************************************** */
/* *** region *************************************************************** */
/* ************************************************************************** */

/**
 * TODO
 *
 * @param src
 */
depth::region::region(const processed_image* src) :
  super_pixel(src), _subs() { }

/**
 * Expands the region to include the new super pixel
 *
 * @param pix  the super pixel to add the region
 */
void depth::region::push(const super_pixel& pix) {
  _subs.push_back(pix);

  _mean_x   = ((_mean_x   * _n_pix) + pix.mean_x()  ) / (_n_pix + pix.n_pix());
  _mean_y   = ((_mean_y   * _n_pix) + pix.mean_y()  ) / (_n_pix + pix.n_pix());
  _mean_r   = ((_mean_r   * _n_pix) + pix.mean_r()  ) / (_n_pix + pix.n_pix());
  _mean_g   = ((_mean_g   * _n_pix) + pix.mean_g()  ) / (_n_pix + pix.n_pix());
  _mean_b   = ((_mean_b   * _n_pix) + pix.mean_b()  ) / (_n_pix + pix.n_pix());
  _mean_hue = ((_mean_hue * _n_pix) + pix.mean_hue()) / (_n_pix + pix.n_pix());
  _mean_sat = ((_mean_sat * _n_pix) + pix.mean_sat()) / (_n_pix + pix.n_pix());

  auto hue_s = pix.hist_hue().begin();
  for(auto hue_t = _hist_hue.begin();
      hue_t != _hist_hue.end(); hue_t++, hue_s++) {
    *hue_t = ((*hue_t * _n_pix) + *hue_t) / (_n_pix + pix.n_pix());
  }

  auto sat_s = pix.hist_sat().begin();
  for(auto sat_t = _hist_sat.begin();
      sat_t != _hist_sat.end(); sat_t++, sat_s++) {
    *sat_t = ((*sat_t * _n_pix) + *sat_t) / (_n_pix + pix.n_pix());
  }

  _n_pix  += pix.n_pix();
}

/* ************************************************************************** */
/* *** processed_image ****************************************************** */
/* ************************************************************************** */

/**
 * Processes an image and creates 3 things from it:
 *   1. a segmented image that will be used to get super pixels
 *   2. an image that uniquely numbers and maps each super pixel
 *   3. an adjacency matrix for the super pixels
 *
 * @param image_name the file to load the image from
 */
depth::processed_image::processed_image(const std::string& image_name) :
    _name(image_name),
    _source(cv::imread(image_name.c_str())),
    _spimg  (_source.size(), CV_8UC3),
    _segment(_source.size(), CV_32SC1),
    _adj_mat() {
  cv::Mat dst(_source.size(), CV_32SC1);

  std::map<int, int> idx_map;
  int idx_gen = 0, x, y;

  /* make sure we can acutally load the image */
  if(!valid())
    return;

  /* start by segmenting the image */
  depth::segment(_source, _spimg);

  /* convert to something that can be easily mapped */
  std::transform(_spimg.begin<cv::Vec3b>(), _spimg.end<cv::Vec3b>(),
      dst.begin<int>(), [](const cv::Vec3b& in)
      { return int(in[0]) + (int(in[1]) << 8) + (int(in[2]) << 16); } );

  /* index the segmentations */
  for(int i = 0; i < _spimg.rows; i++) {
    for(int j = 0; j < _spimg.cols; j++) {
      int curr = dst.at<int>(i, j);

      if(idx_map.find(curr) == idx_map.end()) {
        idx_map[curr] = idx_gen++;
      }

      _segment.at<int>(i, j) = idx_map[curr];
    }
  }

  /* get the edges for the image */
  _adj_mat = cv::Mat::eye(idx_map.size(), idx_map.size(), CV_8UC1);
  for(int i = 1; i < _segment.rows - 1; i++) {
    for(int j = 1; j < _segment.cols - 1; j++) {
      int x = _segment.at<int>(i, j);
      int y;

      if((y = _segment.at<int>(i, j + 1)) != x) {
        _adj_mat.at<uc>(x, y) = 255;
        _adj_mat.at<uc>(y, x) = 255;
      }

      if((y = _segment.at<int>(i, j - 1)) != x) {
        _adj_mat.at<uc>(x, y) = 255;
        _adj_mat.at<uc>(y, x) = 255;
      }

      if((y = _segment.at<int>(i + 1, j)) != x) {
        _adj_mat.at<uc>(x, y) = 255;
        _adj_mat.at<uc>(y, x) = 255;
      }

      if((y = _segment.at<int>(i - 1, j)) != x) {
        _adj_mat.at<uc>(x, y) = 255;
        _adj_mat.at<uc>(y, x) = 255;
      }
    }
  }

  for(int i = 0; i < npixels(); i++) {
    _sps.push_back(super_pixel(this, i));
  }

  comp_laws();
}

/**
 * show the different images produced when the processed image was created.
 *
 * @param delay  how long should this function block, default of -1
 */
void depth::processed_image::display(int delay) {
  if(valid()) {

    cv::namedWindow("segment", CV_WINDOW_AUTOSIZE);
    cvSetMouseCallback("segment", processed_image::mouse_callback, this);

    cv::imshow("source",  _source);
    cv::imshow("segment", _spimg);
    cv::imshow("bw",      create_bw());
    cv::imshow("adj",     _adj_mat);
    cv::waitKey(delay);
  }
}

/**
 * Function that handles a mouse button being clicked on the segmented image
 *
 * @param event   the type of mouse click that was used
 * @param x       the x location of the mouse click inside the image
 * @param y       the y location of the mouse click inside the image
 * @param flags   special flags associated with the mouse event
 * @param unused  data pointer is not used in this function
 */
void depth::processed_image::mouse_callback(int event, int x, int y, int flags,
    void* im) {
  processed_image* pim = (processed_image*)im;

  switch(event) {
    case /* move   */ 0:                          break;
    case /* l_down */ 1: pim->mouse_handle(y, x); break;
    case /* r_down */ 2:                          break;
    case /* m_down */ 3:                          break;
    case /* l_up   */ 4:                          break;
    case /* r_up   */ 5:                          break;
    case /* m_up   */ 6:                          break;
  }
}

/**
 * TODO
 *
 * @param x
 * @param y
 */
void depth::processed_image::mouse_handle(int x, int y) {
  int     pix = _segment.at<int>(x, y);
  cv::Mat row = _adj_mat.row(pix);
  cv::Mat dst = create_bw();
  std::set<int> oth;

  for(int i = 0; i < row.cols; i++) {
    if(i != pix && row.at<unsigned char>(i)) {
      oth.insert(i);
    }
  }

  for(int i = 0; i < _segment.rows; i++) {
    for(int j = 0; j < _segment.cols; j++) {
      int curr = _segment.at<int>(i, j);

      if(curr == pix) {
        dst.at<double>(i, j) = 1.0;
      } else if(oth.find(curr) != oth.end()) {
        dst.at<double>(i, j) = 0.5;
      } else {
        dst.at<double>(i, j) = 0.0;
      }
    }
  }

  cv::imshow("bw", dst);
}

/**
 * TODO
 *
 * @return
 */
cv::Mat depth::processed_image::create_bw() {
  cv::Mat ret = cv::Mat::zeros(_source.size(), CV_64FC1);
  double inc = 1.0 / double(_adj_mat.rows);

  auto src = _segment.begin<int>();
  for(auto dst = ret.begin<double>(); dst != ret.end<double>(); dst++, src++) {
    (*dst) = inc * (*src);
  }

  return ret;
}

void depth::processed_image::comp_laws() {
  std::vector<cv::Mat> masks = depth::laws();
  cv::Mat source = cv::Mat::zeros(_source.size(), CV_64F);
  cv::Mat resp;
  cv::Mat disp;

  std::transform(_source.begin<cv::Vec3b>(), _source.end<cv::Vec3b>(),
      source.begin<double>(), [](cv::Vec3b p)
      { return double(int(p[0]) + int(p[1]) + int(p[2])) / 255.0; } );

  for(int k = 0; k < masks.size(); k++) {
    cv::Mat kernel = masks[k];
    cv::filter2D(source, resp, -1, kernel);

    resp.copyTo(disp);
    double min = *std::min_element(disp.begin<double>(), disp.end<double>());
    double max = *std::max_element(disp.begin<double>(), disp.end<double>());
    std::for_each(disp.begin<double>(), disp.end<double>(),
        [&min, &max](double& d){ d = (d - min) / (max - min); });

    for(int i = 0; i < _source.rows; i++) {
      for(int j = 0; j < _source.cols; j++) {
        _sps[_segment.at<int>(i, j)].laws_resp()[k] += disp.at<double>(i, j);
      }
    }
  }

  for(super_pixel& sp : _sps) {
    for(double& d : sp.laws_resp()) {
      d /= sp.n_pix();
    }
  }
}

/* ************************************************************************** */
/* *** get lines ************************************************************ */
/* ************************************************************************** */

/*void depth::line::lines(depth::processed_image& img,
    std::vector<super_pixel>& sps) {
  cv::Mat gray;
  cv::Mat dx, dy;
  cv::Mat conv;
  cv::Mat poss;
  cv::Mat imcanny;
  std::vector<cv::Mat>   objs;
  std::vector<point_set> directions;

  cv::cvtColor(img.source(), gray, CV_RGB2GRAY);
  cv::GaussianBlur(gray, conv, cv::Size(7, 7), 1,5);
  cv::Sobel(conv, dx, -1, 1, 0);
  cv::Sobel(conv, dy, -1, 0, 1);

  cv::Mat used = cv::Mat::zeros(imcanny.size(), CV_8UC1);
  double min = sqrt((gray.rows * gray.rows) + (gray.cols * gray.cols)) * 0.01;

  for(int i = 0; i < super_pixel::lin_buckets; i++)
    directions.push_back(point_set());

  for(int i = 0; i < imcanny.rows; i++) {
    for(int j = 0; j < imcanny.cols; j++) {
      if(imcanny.at<uc>(i, j)) {
        int index = int(floor((
            atan(dy.at<uc>(i, j) / (dx.at<uc>(i, j) + EPSILON)) + PI/2) /
            PI * super_pixel::lin_buckets)) % super_pixel::lin_buckets;
        directions[index].push_back(point(i, j));
      }
    }
  }

  for(int k = 0; k < super_pixel::lin_buckets; k++) {
    std::vector<point_set> groups;
    point_set possible;

    poss = cv::Mat::zeros(imcanny.size(), CV_8UC1);
    objs.clear();

    if(k > 0)
      append(possible, directions[k - 1].begin(), directions[k - 1].end());
    if(k < super_pixel::lin_buckets - 1)
      append(possible, directions[k + 1].begin(), directions[k + 1].end());
    append(possible, directions[k].begin(), directions[k].end());

    std::for_each(possible.begin(), possible.end(), [&poss](const point& curr)
        { poss.at<uc>(curr.first, curr.second) = 255; } );

    for(int i = 0; i < poss.rows; i++) {
      for(int j = 0; j < poss.cols; j++) {
        if(poss.at<uc>(i, j) && !used.at<uc>(i, j)) {
          point_set set;
          get_group(poss, point(i, j), set);
          groups.push_back(set);
        }
      }
    }

    for(point_set& ps : groups) {
      if(ps.size() > min && ps.size() < poss.rows * poss.cols) {
        for(point p : ps) {
          used.at<uc>(p.first, p.second) = 1;
        }

        point rel = ps[0];
        sps[img.segment().at<int>(rel.first, rel.second)].hist_line()[k]++;
        sps[img.segment().at<int>(rel.first, rel.second)].n_line()++;
      }
    }
  }
}

void depth::line::get_group(cv::Mat& img, point curr, point_set& dst) {
  dst.push_back(curr);

  img.at<uc>(curr.first, curr.second) = 0;

  int maxI = curr.first  < img.rows - 1 ? curr.first + 1  : img.rows - 1;
  int maxJ = curr.second < img.cols - 1 ? curr.second + 1 : img.cols - 1;

  for(int i = curr.first; i <= maxI; i++) {
    for(int j = curr.second; j <= maxJ; j++) {
      if(img.at<uc>(i, j)) {
        get_group(img, point(i, j), dst);
      }
    }
  }
}*/
