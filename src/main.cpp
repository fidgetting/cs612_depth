/*
 * main.cpp
 *
 *  Created on: Feb 14, 2012
 *      Author: norton
 */

/* local includes */
#include <ground.h>
#include <super.h>
#include <util.h>
#include <knearest.h>

/* std library */
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/* open cv */
#include <opencv2/opencv.hpp>

/* boost includes */
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/timer.hpp>

#define MODEL_R CvBoost
#define MODEL_P depth::knearest

int main(int argc, char** argv) {
  std::vector<std::string> file_list;
  std::vector<std::string> video_list;
  std::shared_ptr<MODEL_R> m_region;
  std::shared_ptr<MODEL_P> m_pair;
  po::options_description opt("Options");
  po::variables_map vm;
  std::string directory;
  std::string truth;
  std::string model;
  boost::timer watch;
  bool p_help  = false;
  bool m_truth = false;

  /* parameters for the region classifier */
  /*CvSVMParams   region_p;

  region_p.kernel_type = CvSVM::RBF;
  region_p.degree = pow(2,  5);
  region_p.C      = pow(2, -5);*/
  CvBoostParams region_p;

  region_p.boost_type     = CvBoost::DISCRETE;
  region_p.split_criteria = CvBoost::DEFAULT;

  /* parameters for paired classifier */
  CvBoostParams pair_p;

  pair_p.boost_type     = CvBoost::DISCRETE;
  pair_p.split_criteria = CvBoost::DEFAULT;

  opt.add_options()
      ("help,h",
        "produce help message")
      ("video,v",
        po::value<std::vector<std::string> >(&video_list),
        "analyze this video")
      ("file,f",
        po::value<std::vector<std::string> >(&file_list),
        "analyze this file")
      ("directory,d",
        po::value<std::string>(&directory)->default_value(""),
        "analyze all the files in this directory")
      ("truth,t",
        po::value<std::string>(&truth)->default_value(""),
        "load ground truth information from this file")
      ("make,m",
        "make the group truth maps")
      ("model,M",
        po::value<std::string>(&model)->default_value(""),
        "the file that the trained model is saved in");


  po::store(po::command_line_parser(argc, argv).options(opt).run(), vm);
  po::notify(vm);

  if((p_help = vm.count("help"))) {
    opt.print(std::cout);
    return 0;
  }

  m_truth = vm.count("make");

  std::vector<std::shared_ptr<depth::ground_truth> > samples;
  if(truth.length() != 0) {
    std::cout << "LOAD:" << std::endl;
    fs::path src_dir(truth.substr(0, truth.find(':')));
    fs::path dst_dir(truth.substr(truth.find(':') + 1));

    for(auto iter = fs::directory_iterator(src_dir);
        iter != fs::directory_iterator(); iter++) {
      fs::path curr = (*iter).path();
      fs::path save = dst_dir / fs::basename(curr);

      if(m_truth) {
        if(!fs::is_regular_file(save)) {
          std::shared_ptr<depth::processed_image> img =
              std::make_shared<depth::processed_image>(curr.string());
          depth::ground_truth ground(img);

          ground.display();

          std::ofstream ostr(save.string());
          ostr << save << " ";
          for(depth::ground_truth::label l : ground.labels()) {
            ostr << int(l) << " ";
          }
        }
      } else {
        if(fs::is_regular_file(save)) {
          std::cout << "load: " << curr << std::flush;
          watch.restart();
          std::shared_ptr<depth::processed_image> img =
              std::make_shared<depth::processed_image>(curr.string());
          samples.push_back(
              std::make_shared<depth::ground_truth>(img, save));
          std::cout << " " << watch.elapsed() << std::endl;
        }
      }
    }

    if(samples.size() == 0) {
      return 0;
    }
  }

  if(samples.size() != 0) {
    std::cout << "\nTRAINING: region" << std::flush;
    watch.restart();
    m_region = depth::train_region<MODEL_R>(samples, region_p);
    m_region->save(model.c_str(), "region");
    std::cout << " " << watch.elapsed() << std::endl;

  } else {
    m_region = std::make_shared<MODEL_R>();
    m_region->load(model.c_str(), "region");
  }

  if(directory.size() != 0) {
    for(auto iter = fs::directory_iterator(directory);
        iter != fs::directory_iterator(); iter++) {
      std::cout << (*iter).path() << std::endl;
      depth::processed_image pi((*iter).path().c_str());

      pi.predict(m_region);
      pi.display();
    }
  }

  if(file_list.size() != 0) {
    for(std::string& str : file_list) {
      std::cout << str << std::endl;
      depth::processed_image pi(str);

      pi.predict(m_region);
      pi.display();
    }
  }

  for(std::string& str : video_list) {
    cv::Mat src;
    cv::VideoCapture vid(str);
    int frame = 1;

    cv::Size frame_size(
        vid.get(CV_CAP_PROP_FRAME_HEIGHT),
        vid.get(CV_CAP_PROP_FRAME_WIDTH));

    while(vid.read(src)) {
      watch.restart();
      depth::processed_image pi(src);

      pi.predict(m_region);
      pi.display(100);
      std::cout << "[" << frame << ", " << watch.elapsed() << "] "
          << std::flush;
      if((frame++) % 10 == 0) {
        std::cout << std::endl;
      }
    }
  }

  return 0;
}

