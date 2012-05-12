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

      const static int col_spaces = 3;
      const static int law_buckets = 9;
      const static int gra_buckets = 18;

      const static int n_dim = (law_buckets * col_spaces)
              + (gra_buckets * col_spaces);

      super_pixel(const processed_image* src);
      super_pixel(const processed_image* src, int32_t _sp_num);
      virtual ~super_pixel() { }

      virtual void pack(cv::Mat& dst);
      virtual void pack_with(cv::Mat& dst, super_pixel& sp);

      inline int sp_num() const { return _sp_num; }

      inline    int n_pix()    const { return _n_pix;    }

      inline       std::vector<double>& laws_resp()       { return _laws_resp; }
      inline const std::vector<double>& laws_resp() const { return _laws_resp; }
      inline       std::vector<double>& grad_hist()       { return _grad_hist; }
      inline const std::vector<double>& grad_hist() const { return _grad_hist; }

      inline const processed_image* source() const { return _src; }

      inline double& label()       { return _label; }
      inline double  label() const { return _label; }

    protected:

      /* the source images */
      const processed_image* _src;
      int32_t _sp_num;

      /* color */
      double           _n_pix;

      /* geometry */
      std::vector<double> _laws_resp;
      std::vector<double> _grad_hist;

      /* the final label of the super pixel */
      double _label;
  };

  std::set<super_pixel> get_super(processed_image& src);
  bool operator<(const super_pixel& lhs, const super_pixel& rhs);

  /* ************************************************************************ */
  /* *** region ************************************************************* */
  /* ************************************************************************ */

  class region : public super_pixel{
    public:

      region(const processed_image* src);

      virtual void pack(cv::Mat& dst);

      void push(const super_pixel& pix);
      void clear();

      inline       std::vector<super_pixel>& subs()       { return _subs; }
      inline const std::vector<super_pixel>& subs() const { return _subs; }

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

      template<typename region_t, typename pair_t>
      void pair(const std::shared_ptr<region_t> region,
          const std::shared_ptr<pair_t> pair);

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
#ifdef USE_REGION
      re.clear();

      re.push(_sps[i]);
      for(int j = 0; j < npixels(); j++) {
        if(_adj_mat.at<uint8_t>(i, j)) {
          re.push(_sps[j]);
        }
      }

      re.pack(input);
#else
      _sps[i].pack(input);
#endif

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
   * TODO
   *
   * @param model
   */
  template<typename region_t, typename pair_t>
  void processed_image::pair(const std::shared_ptr<region_t> m_region,
      const std::shared_ptr<pair_t> m_pair) {
    const static int repeat = 10;

    cv::Mat input(1, super_pixel::n_dim, CV_32FC1);
    std::vector<super_pixel> ran_sps(_sps);
    std::vector<int> npregion = {3, 4, 5, 7, 9, 11, 15, 20, 25};

    region* best;
    double best_f, sum;
    int np_sum = 0;

    std::for_each(npregion.begin(), npregion.end(),
        [&np_sum](int& i) { np_sum += i; } );

    /* begin by placing the super pixels in regions */
    for(int& np : npregion) {
      for(int i = 0; i < repeat; i++) {
        std::vector<region> regions(np, this);
        std::random_shuffle(ran_sps.begin(), ran_sps.end());

        auto iter = ran_sps.begin();
        for(region& re : regions) {
          re.push(*(iter++));
        }

        for(; iter != ran_sps.end(); iter++) {
          best_f = 0;
          best = NULL;

          for(region& re : regions) {
            re.pack_with(input, *iter);

            sum = m_pair->predict(input, cv::Mat(), cv::Range::all(),
                false, true);

            if(best == NULL || sum > best_f) {
              best = &re;
              best_f = sum;
            }
          }

          best->push(*iter);
        }

        for(region& re : regions) {
          re.pack(input);
          sum = m_region->predict(input, cv::Mat(), cv::Range::all(),
              false, true);

          for(super_pixel& sp : re.subs()) {
            _sps[sp.sp_num()].label() += sum;
          }
        }
      }
    }

    for(super_pixel& sp : _sps) {
      sp.label() /= double(np_sum);

      int assign = m_region;
    }

    for(super_pixel& sp : _sps) {
      std::cout << sp.sp_num() << "[" << sp.label() << ", " <<
          (sp.label() < 0 ? "horz" : "vert") << "]" << std::endl;
    }

    for(int i = 0; i < _source.rows; i++) {
      for(int j = 0; j < _source.cols; j++) {
        _spimg.at<cv::Vec3b>(i, j) = _sps[_segment.at<int>(i, j)] < 0 ?
            cv::Vec3b(255, 0, 0) : cv::Vec3b(0, 255, 0);
      }
    }
  }
}

#endif /* SUPER_H_ */
