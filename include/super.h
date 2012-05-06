/*
 * super.h
 *
 *  Created on: Feb 20, 2012
 *      Author: norton
 */

#ifndef SUPER_H_
#define SUPER_H_

#include <util.h>

#include <iostream>
#include <set>
#include <vector>
#include <memory>

namespace depth {

  typedef std::pair<int, int> point;
  typedef std::vector<point>  point_set;

  class processed_image;

  /* ************************************************************************ */
  /* *** super pixel ******************************************************** */
  /* ************************************************************************ */

  class super_pixel {
    public:

#ifdef INC_COLORLOC
      const static int base = 8;
      const static int hue_buckets = 5;
      const static int sat_buckets = 3;
#else
      const static int base = 0;
      const static int hue_buckets = 0;
      const static int sat_buckets = 0;
#endif
      const static int col_spaces = 3;
      const static int law_buckets = 9;
      const static int gra_buckets = 18;

      const static int n_dim = base + hue_buckets + sat_buckets
          + (law_buckets * col_spaces)
          + (gra_buckets * col_spaces);

      super_pixel(const processed_image* src);
      super_pixel(const processed_image* src, int32_t _sp_num);
      virtual ~super_pixel() { }

      virtual void pack(cv::Mat& dst);

      inline int sp_num() const { return _sp_num; }

      inline    int n_pix()    const { return _n_pix;    }
      inline double mean_r()   const { return _mean_r;   }
      inline double mean_g()   const { return _mean_g;   }
      inline double mean_b()   const { return _mean_b;   }
      inline double mean_hue() const { return _mean_hue; }
      inline double mean_sat() const { return _mean_sat; }

      inline const std::vector<double>& hist_hue() const { return _hist_hue; }
      inline const std::vector<double>& hist_sat() const { return _hist_sat; }

      inline double mean_x() const { return _mean_x; }
      inline double mean_y() const { return _mean_y; }

      inline       std::vector<double>& laws_resp()       { return _laws_resp; }
      inline const std::vector<double>& laws_resp() const { return _laws_resp; }
      inline       std::vector<double>& grad_hist()       { return _grad_hist; }
      inline const std::vector<double>& grad_hist() const { return _grad_hist; }

      inline const processed_image* source() const { return _src; }

    protected:

      /* the source images */
      const processed_image* _src;
      int32_t _sp_num;

      /* color */
      double           _n_pix;
      double           _mean_r;
      double           _mean_g;
      double           _mean_b;
      double           _mean_hue;
      double           _mean_sat;
      std::vector<double> _hist_hue;
      std::vector<double> _hist_sat;

      /* textures: DOOG filters */
      // TODO

      /* location and shape */
      double           _mean_x;
      double           _mean_y;

      /* geometry */
      std::vector<double> _laws_resp;
      std::vector<double> _grad_hist;
  };

  std::set<super_pixel> get_super(processed_image& src);
  bool operator<(const super_pixel& lhs, const super_pixel& rhs);

  std::ostream& operator<<(std::ostream& ostr, const super_pixel& pix);

  /* ************************************************************************ */
  /* *** region ************************************************************* */
  /* ************************************************************************ */

  class region : public super_pixel{
    public:

      region(const processed_image* src);

      virtual void pack(cv::Mat& dst);

      void push(const super_pixel& pix);
      void clear();

    protected:

      std::vector<super_pixel> _subs;
  };

  /* ************************************************************************ */
  /* *** processed image **************************************************** */
  /* ************************************************************************ */

  /*struct line {
      line(double x1, double y1, double x2, double y2, double theta, double r) :
        x1(x1), y1(y1), x2(x2), y2(y2), theta(theta), r(r) { }

      static void lines(processed_image& img,
          std::vector<super_pixel>& sps);
      static void get_group(cv::Mat& img, point curr, point_set& dst);

      double x1, y1;
      double x2, y2;
      double theta;
      double r;
  };*/

  class processed_image {
    public:

      processed_image();
      processed_image(const cv::Mat& img);
      processed_image(const std::string& image_name);

      template<typename model_t>
      void predict(const std::shared_ptr<model_t> model);
      template<typename model_t>
      cv::Mat pair(std::shared_ptr<model_t> model);

      void display(int delay = -1);
      void write(cv::VideoWriter& vert_v, cv::VideoWriter& horz_v);

      template<typename _func_t>
      void comp_channel(_func_t func, int off);

      void comp_laws(cv::Mat img, int off);
      void comp_grad(cv::Mat img, int off);

      inline bool valid() { return _source.data != NULL; }

      inline std::string& name()       { return _name; }
      inline std::string  name() const { return _name; }

      inline       cv::Mat& source()        { return _source;  }
      inline const cv::Mat& source()  const { return _source;  }
      inline       cv::Mat& segimg()        { return _spimg;   }
      inline const cv::Mat& segimg()  const { return _spimg;   }
      inline       cv::Mat& segment()       { return _segment; }
      inline const cv::Mat& segment() const { return _segment; }
      inline       cv::Mat& adj_mat()       { return _adj_mat; }
      inline const cv::Mat& adj_mat() const { return _adj_mat; }

      inline       std::vector<super_pixel>& sps()       { return _sps; }
      inline const std::vector<super_pixel>& sps() const { return _sps; }

      inline int rows() const { return _source.rows; }
      inline int cols() const { return _source.cols; }

      inline int npixels() const { return _adj_mat.rows; }

    private:

      static void mouse_callback(int event, int x, int y, int flags, void* im);

      void    mouse_handle(int x, int y);
      cv::Mat create_bw();

      std::string _name;
      cv::Mat _source;
      cv::Mat _spimg;
      cv::Mat _segment;
      cv::Mat _adj_mat;
      std::vector<super_pixel> _sps;
  };

  /* ************************************************************************ */
  /* *** TODO NEEDS WORK **************************************************** */
  /* ************************************************************************ */

  template<typename _func_t>
  void processed_image::comp_channel(_func_t func, int off) {
    cv::Mat src = cv::Mat::zeros(_source.size(), CV_32FC3);
    cv::Mat R   = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat G  = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat B  = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat Hue = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat Sat = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat dst;

    std::transform(_source.begin<cv::Vec3b>(), _source.end<cv::Vec3b>(),
        src.begin<cv::Vec3f>(), [] (cv::Vec3b& in)
        { return cv::Vec3f(in[0] / 255.0, in[1]/255.0, in[2]/255.0); } );

    std::transform(src.begin<cv::Vec3f>(), src.end<cv::Vec3f>(),
        R.begin<float>(), [] (cv::Vec3f& in) { return in[0]; } );
    std::transform(src.begin<cv::Vec3f>(), src.end<cv::Vec3f>(),
        G.begin<float>(), [] (cv::Vec3f& in) { return in[1]; } );
    std::transform(src.begin<cv::Vec3f>(), src.end<cv::Vec3f>(),
        B.begin<float>(), [] (cv::Vec3f& in) { return in[2]; } );

    /*cv::cvtColor(src, dst, CV_RGB2HSV);
    std::transform(dst.begin<cv::Vec3f>(), dst.end<cv::Vec3f>(),
        Hue.begin<float>(), [] (cv::Vec3f& in) { return in[0]; } );
    std::transform(dst.begin<cv::Vec3f>(), dst.end<cv::Vec3f>(),
        Sat.begin<float>(), [] (cv::Vec3f& in) { return in[1]; } );*/

    func(R,   off * 0);
    func(G,   off * 1);
    func(B,   off * 2);
    //func(Hue, off * 0);
    //func(Sat, off * 1);

    for(super_pixel& sp : _sps)
      for(double& d : sp.laws_resp())
        d /= sp.n_pix();
    for(super_pixel& sp : _sps)
      for(double& d : sp.grad_hist())
        d /= sp.n_pix();
  }

  /**
   * TODO
   *
   * @param model
   */
  template<typename model_t>
  void processed_image::predict(const std::shared_ptr<model_t> model) {
    cv::Mat input(1, super_pixel::n_dim, CV_32FC1);
    std::map<int, int> labels;
    region re(this);

    for(int i = 0; i < npixels(); i++) {
      re.clear();

      re.push(_sps[i]);
      for(int j = 0; j < npixels(); j++) {
        if(_adj_mat.at<uint8_t>(i, j)) {
          re.push(_sps[j]);
        }
      }

      re.pack(input);

      //std::cout << i << " => ";
      labels[i] = int(model->predict(input));
      //std::cout << " => " << labels[i] << std::endl;
    }

    for(int i = 0; i < _source.rows; i++) {
      for(int j = 0; j < _source.cols; j++) {
        switch(labels[_segment.at<int>(i, j)]) {
          case 0:
            _spimg.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            break;

          case -1:
            _spimg.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
            break;

          case 1:
            _spimg.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
            break;

          case 3:
            _spimg.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            break;
        }
      }
    }
  }

  /**
   * Gets a matrix of pair-wise super pixel likelihoods
   *
   * @param model
   */
  template<typename model_t>
  cv::Mat processed_image::pair(std::shared_ptr<model_t> model) {
    cv::Mat input(1, super_pixel::n_dim * 2, CV_32FC1);
    cv::Mat a_in = input.colRange(0, super_pixel::n_dim);
    cv::Mat b_in = input.colRange(super_pixel::n_dim, super_pixel::n_dim * 2);
    cv::Mat pair_wise(npixels(), npixels(), CV_64F);

    for(int i = 0; i < _sps.size() - 1; i++) {
      for(int j = i + 1; j < _sps.size(); j++) {
        if(_adj_mat.at<uc>(i, j)) {
          _sps[i].pack(a_in);
          _sps[j].pack(b_in);

          pair_wise.at<double>(i, j) =
              pair_wise.at<double>(j, i) =
                  model->predict(input);
          std::cout << pair_wise.at<double>(i, j) << " " << std::flush;
        }
      }
    }

    std::cout << std::endl;
    return pair_wise;
  }
}

#endif /* SUPER_H_ */
