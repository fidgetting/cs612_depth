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

using std::placeholders::_1;
using std::placeholders::_2;

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
    _mean_x(0), _mean_y(0), _laws_resp(law_buckets * col_spaces),
    _grad_hist(gra_buckets * col_spaces) { }

/**
 * TODO
 *
 * @param src
 * @param sp_num
 */
depth::super_pixel::super_pixel(const processed_image* src, int32_t sp_num) :
    _src(src), _sp_num(sp_num), _n_pix(0), _mean_r(0), _mean_g(0), _mean_b(0),
    _mean_hue(0), _mean_sat(0), _hist_hue(hue_buckets), _hist_sat(sat_buckets),
    _mean_x(0), _mean_y(0), _laws_resp(law_buckets * col_spaces),
    _grad_hist(gra_buckets * col_spaces) {

  std::for_each(_src->segment().begin<int>(), _src->segment().end<int>(),
      [&](int i){ if(i == _sp_num) _n_pix++; });
#ifdef INC_COLORLOC
  /* do the easy stuff first:
   *   mean      red, green, blue
   *   mean      hue, saturation
   *   mean      x, y
   *   histogram hue, saturation
   */
  double _row_norm = 1.0 / double(_src->rows());
  double _col_norm = 1.0 / double(_src->cols());

  cv::Mat hsv(_src->source().size(), CV_8UC3);
  cv::cvtColor(_src->source(), hsv, CV_RGB2HSV);

  for(int i = 0; i < _src->rows(); i++) {
    for(int j = 0; j < _src->cols(); j++) {
      if(_src->segment().at<int>(i, j) == _sp_num) {
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
#endif
}

/**
 * TODO
 *
 * @param row
 */
void depth::super_pixel::pack(cv::Mat& row) {
  int idx = base;

  if(row.rows != 1 || row.cols != n_dim) {
    throw std::exception();
  }

#ifdef INC_COLORLOC
  //row.at<float>(0 ) = _n_pix;
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
#endif

  for(int i = 0; i < law_buckets; i++)
    row.at<float>(idx++) = _laws_resp[i];
  for(int i = 0; i < gra_buckets; i++)
    row.at<float>(idx++) = _grad_hist[i];
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
 * TODO
 *
 * @param dst
 */
void depth::region::pack(cv::Mat& dst) {
  /* zero everything */
  _mean_r = _mean_g = _mean_b = 0;
  _mean_x = _mean_y = 0;
  _mean_hue = _mean_sat = 0;

  std::for_each(_hist_hue.begin(),  _hist_hue.end(),
      [] (double& d) { d = 0; } );
  std::for_each(_hist_sat.begin(),  _hist_sat.end(),
      [] (double& d) { d = 0; } );
  std::for_each(_laws_resp.begin(), _laws_resp.end(),
      [] (double& d) { d = 0; } );
  std::for_each(_grad_hist.begin(), _grad_hist.end(),
      [] (double& d) { d = 0; } );

  /* add up everything for average */
  for(super_pixel& sp : _subs) {
#ifdef INC_COLORLOV
    _mean_r += sp.mean_r();
    _mean_g += sp.mean_g();
    _mean_b += sp.mean_b();
    _mean_hue += sp.mean_hue();
    _mean_sat += sp.mean_sat();
    _mean_x += sp.mean_x();
    _mean_y += sp.mean_y();

    comb(sp.hist_hue().begin(), sp.hist_hue().end(), _hist_hue.begin(),
        [] (const double& src, double& dst) { dst += src; } );
    comb(sp.hist_sat().begin(), sp.hist_sat().end(), _hist_sat.begin(),
        [] (const double& src, double& dst) { dst += src; } );
#endif

    comb(sp.laws_resp().begin(), sp.laws_resp().end(), _laws_resp.begin(),
        [] (const double& src, double& dst) { dst += src; } );
    comb(sp.grad_hist().begin(), sp.grad_hist().end(), _grad_hist.begin(),
        [] (const double& src, double& dst) { dst += src; } );

    _n_pix += sp.n_pix();
  }

  /* divide through by the number of pixels */
  _mean_r   /= _subs.size();
  _mean_g   /= _subs.size();
  _mean_b   /= _subs.size();
  _mean_hue /= _subs.size();
  _mean_sat /= _subs.size();
  _mean_x   /= _subs.size();
  _mean_y   /= _subs.size();

  std::for_each(_hist_hue.begin(),  _hist_hue.end(),
      [&](double& d){ d /= _subs.size(); } );
  std::for_each(_hist_sat.begin(),  _hist_sat.end(),
      [&](double& d){ d /= _subs.size(); } );
  std::for_each(_laws_resp.begin(), _laws_resp.end(),
      [&](double& d){ d /= _subs.size(); } );
  std::for_each(_grad_hist.begin(), _grad_hist.end(),
      [&](double& d){ d /= _subs.size(); } );

  super_pixel::pack(dst);
}

/**
 * Expands the region to include the new super pixel
 *
 * @param pix  the super pixel to add the region
 */
void depth::region::push(const super_pixel& pix) {
  _subs.push_back(pix);
}

/**
 * TODO
 */
void depth::region::clear() {
  _subs.clear();
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
depth::processed_image::processed_image(const cv::Mat& img) :
    _name(""), _source(img.clone()),
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

  comp_channel(std::bind<void>(&processed_image::comp_laws, this, _1, _2),
      super_pixel::law_buckets);
  comp_channel(std::bind<void>(&processed_image::comp_grad, this, _1, _2),
      super_pixel::gra_buckets);
}

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

  comp_channel(std::bind<void>(&processed_image::comp_laws, this, _1, _2),
      super_pixel::law_buckets);
  comp_channel(std::bind<void>(&processed_image::comp_grad, this, _1, _2),
      super_pixel::gra_buckets);
}

/**
 * show the different images produced when the processed image was created.
 *
 * @param delay  how long should this function block, default of -1
 */
void depth::processed_image::display(int delay) {
  if(valid()) {

    cv::Mat vert = cv::Mat::zeros(_source.size(), CV_8UC3);
    cv::Mat horz = cv::Mat::zeros(_source.size(), CV_8UC3);

    for(int i = 0; i < _source.rows; i++) {
      for(int j = 0; j < _source.cols; j++) {
        if(_spimg.at<cv::Vec3b>(i, j)[1]) {
          vert.at<cv::Vec3b>(i, j) = _source.at<cv::Vec3b>(i, j);
        } else {
          horz.at<cv::Vec3b>(i, j) = _source.at<cv::Vec3b>(i, j);
        }
      }
    }

    cv::imshow("vertical", vert);
    cv::imshow("horizontal", horz);
    cv::imshow("source", _source);

    cv::imwrite("horz.png", horz);
    cv::imwrite("vert.png", vert);
    cv::imwrite("comb.png", _source);

    cv::waitKey(delay);
  }
}

/**
 * TODO
 *
 * @param vert
 * @param horz
 */
void depth::processed_image::write(cv::VideoWriter& vert_v,
    cv::VideoWriter& horz_v) {
  if(valid()) {
    cv::Mat vert = cv::Mat::zeros(_source.size(), CV_8UC3);
    cv::Mat horz = cv::Mat::zeros(_source.size(), CV_8UC3);

    for(int i = 0; i < _source.rows; i++) {
      for(int j = 0; j < _source.cols; j++) {
        if(_spimg.at<cv::Vec3b>(i, j)[1]) {
          vert.at<cv::Vec3b>(i, j) = _source.at<cv::Vec3b>(i, j);
        } else {
          horz.at<cv::Vec3b>(i, j) = _source.at<cv::Vec3b>(i, j);
        }
      }
    }

    vert_v << vert;
    horz_v << horz;
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

  std::cout << pix << std::endl;

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

void depth::processed_image::comp_laws(cv::Mat source, int off) {
  std::vector<cv::Mat> masks = depth::laws();
  cv::Mat resp;
  cv::Mat disp;

  for(int k = 0; k < masks.size(); k++) {
    cv::Mat kernel = masks[k];
    cv::filter2D(source, resp, -1, kernel);

    resp.copyTo(disp);
    double min = *std::min_element(disp.begin<float>(), disp.end<float>());
    double max = *std::max_element(disp.begin<float>(), disp.end<float>());
    std::for_each(disp.begin<float>(), disp.end<float>(),
        [&min, &max] (float& d) { d = (d - min) / (max - min); });

    for(int i = 0; i < _source.rows; i++) {
      for(int j = 0; j < _source.cols; j++) {
        _sps[_segment.at<int>(i, j)].laws_resp()[off + k] += disp.at<float>(i, j);
      }
    }
  }
}

void depth::processed_image::comp_grad(cv::Mat img, int off) {
  cv::Mat dx, dy;

  cv::Sobel(img, dx, -1, 1, 0);
  cv::Sobel(img, dy, -1, 0, 1);

  for(int i = 0; i < _source.rows; i++) {
    for(int j = 0; j < _source.cols; j++) {
      int bucket = round(((atan(dy.at<float>(i, j) /
          (dx.at<float>(i, j) + EPSILON)) + (PI / 2.0)) / PI) *
          super_pixel::gra_buckets);

      _sps[_segment.at<int>(i, j)].grad_hist()[off + bucket]++;
    }
  }
}
