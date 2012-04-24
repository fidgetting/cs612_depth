/*
 * ground.hpp
 *
 *  Created on: Mar 14, 2012
 *      Author: norton
 */

#pragma once

#include <super.h>

#include <fstream>
#include <memory>
#include <vector>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

namespace depth {

  /**
   * TODO
   */
  class ground_truth {
    public:

      enum label {
        unlabled = 0,
        ground = 1,
        vertical = 2,
        sky = 3
      };

      ground_truth(std::shared_ptr<processed_image> img);
      ground_truth(std::shared_ptr<processed_image> img, fs::path file);

      void display(int delay = -1);

      const std::vector<label> labels() const { return _labels; }
      const std::shared_ptr<processed_image> get_img() const { return img; }

      label at(int idx) const { return _labels.at(idx); }

    private:

      static void mouse_callback(int event, int x, int y, int flags, void* im);

      void mouse_handle(int x, int y);
      void paint_ground(cv::Mat& dst, cv::Mat& simp);

      std::shared_ptr<processed_image> img;
      std::vector<label> _labels;
  };

  /**
   * TODO
   *
   * @param train_data
   * @return
   */
  template<typename model_t, typename params_t>
  std::shared_ptr<model_t>
  train_region(std::vector<std::shared_ptr<ground_truth> >& train_data,
      params_t& params) {
    int total_pix = 0;
    int curr_row = 0;

    /* put together the classifier for the type of the super pixel */
    for(const std::shared_ptr<ground_truth>& g : train_data) {
      total_pix += g->get_img()->npixels();
    }

    cv::Mat train_mat(total_pix, depth::super_pixel::n_dim, CV_32FC1);
    cv::Mat respo_mat(total_pix, 1, CV_32SC1);

    for(const std::shared_ptr<ground_truth>& g : train_data) {
      for(const super_pixel& sp : g->get_img()->sps()) {
        respo_mat.at<int>(curr_row) = int(g->at(sp.sp_num()));

        cv::Mat row = train_mat.row(curr_row++);

        sp.pack(row);
      }
    }

    respo_mat.at<int>(0, 0) = 0;

    CvMat train = train_mat;
    CvMat respo = respo_mat;

    return std::shared_ptr<model_t>(
        new model_t(train_mat, respo_mat, 1));
    /*return std::shared_ptr<model_t>(
        new model_t(&train, &respo, 0, 0, params));*/
  }

  /**
   * TODO
   *
   * @param train_data
   * @return
   */
  template<typename model_t, typename params_t>
  std::shared_ptr<model_t>
  train_pair(std::vector<std::shared_ptr<ground_truth> >& train_data,
      params_t& params) {
    int total_samples = 0;
    int curr_row = 0;

    for(const std::shared_ptr<ground_truth>& g : train_data) {
      int npix = g->get_img()->npixels();
      /*int curr_img = 0;

      std::for_each(g->get_img()->adj_mat().begin<uc>(),
          g->get_img()->adj_mat().end<uc>(),
          [&](uc c){ if(c) curr_img++; } );

      total_samples += (curr_img / 2);*/
      for(int i = 0; i < npix - 1; i++) {
        for(int j = i + 1; j < npix; j++) {
          if(g->get_img()->adj_mat().at<uc>(i, j)) {
            total_samples++;
          }
        }
      }
    }

    cv::Mat train_mat(total_samples, depth::super_pixel::n_dim * 2, CV_32FC1);
    cv::Mat respo_mat(total_samples, 1, CV_32SC1);

    for(const std::shared_ptr<ground_truth>& g : train_data) {
      auto pix = g->get_img()->sps();
      int npix = g->get_img()->npixels();

      for(int i = 0; i < npix - 1; i++) {
        for(int j = i + 1; j < npix; j++) {
          if(g->get_img()->adj_mat().at<uc>(i, j)) {
            respo_mat.at<int>(curr_row) =
                (g->at(pix[i].sp_num()) == g->at(pix[j].sp_num())) ? 1 : -1;

            cv::Mat row = train_mat.row(curr_row++);
            cv::Mat i_mat = row.colRange(0, super_pixel::n_dim);
            cv::Mat j_mat = row.colRange(super_pixel::n_dim, row.cols);

            pix[i].pack(i_mat);
            pix[j].pack(j_mat);
          }
        }
      }
    }

    CvMat train = train_mat;
    CvMat respo = respo_mat;

    return std::shared_ptr<model_t>(
        new model_t(train_mat, respo_mat, 1));
    /*return std::shared_ptr<model_t>(
        new model_t(&train, CV_ROW_SAMPLE, &respo, NULL, NULL, NULL, NULL,
            params));*/
  }
}



