/*------------------------------------------------------*/
/* Copyright 2013, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/

#undef __STRICT_ANSI__
#include <fstream>
#include <string>
#include <iomanip>
#include <sys/time.h>
#include <map>

#include "/thirdPary/mods/io_mods.h"

#include "/thirdPary/mods/detectors/mser/extrema/extrema.h"
#include "/thirdPary/mods/detectors/helpers.h"
#include "/thirdPary/mods/matching/siftdesc.h"
#include "/thirdPary/mods/synth-detection.hpp"

#include "/thirdPary/mods/detectors/affinedetectors/scale-space-detector.hpp"
#include "/thirdPary/mods/detectors/detectors_parameters.hpp"
#include "/thirdPary/mods/descriptors_parameters.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "/thirdPary/mods/matching.hpp"

#include "/thirdPary/mods/configuration.hpp"
#include "/thirdPary/mods/imagerepresentation.h"
#include "/thirdPary/mods/correspondencebank.h"


//#define SCV

#ifdef SCV
#include "scv/scv_entrypoint.hpp"
#endif

#ifdef WITH_ORSA
#include "orsa.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

const int nn_n = 50; //number of nearest neighbours retrieved to get 1st inconsistent

//inline long getMilliSecs()
//{
//  timeval t;
//  gettimeofday(&t, NULL);
//  return t.tv_sec*1000 + t.tv_usec/1000;
//}

int WxBSdet_desc(string path, string config,string iters,std::vector<float> &RootSIFTdesc, std::vector<float> &HalfRootSIFTdesc)
{

  long c_start = getMilliSecs();
  double time1;
  TimeLog TimingLog;
  logs log1;
  /// Parameters reading
  configs Config1;
  int VERB = Config1.OutputParam.verbose;
  /// Ground truth homography reading
  log1.VerifMode =  Config1.CLIparams.ver_type;
  conf1.CLIparams.img1_fname = path;
  //conf1.CLIparams.k1_fname = argv[2];
  conf1.CLIparams.config_fname = config;
  conf1.CLIparams.iters_fname = iters;
  
  /// Input images reading
  cv::Mat img1;
  SynthImage tilt_img1;
  tilt_img1.id=0;
  //tilt_img2.id=1000;

      img1 = cv::imread(Config1.CLIparams.img1_fname,Config1.LoadColor); // load grayscale; Try RGB?

  if(!img1.data) {
    std::cerr <<  "Could not open or find the image1 " << Config1.CLIparams.img1_fname << std::endl;
    return 1;
  }
    cv::resize(img1,img1,cv::Size(640,480));
  /// Data structures preparation
  ImageRepresentation ImgRep1;
  if (Config1.CLIparams.doCLAHE)
  {
      long clahe_start = getMilliSecs();

      Ptr<CLAHE> clahe = createCLAHE();
      clahe->setClipLimit(4);
      cv::Mat img1_clahe, img2_clahe;

      cv::Mat gray_in_img;
      if (img1.channels() == 3)
        {
          cv::Mat gray_img1;
          //cv::cvtColor(img1, gray_img1, CV_BGR2GRAY);
          std::vector<cv::Mat> RGB_planes(3);
          cv::Mat in_32f;
          img1.convertTo(in_32f,CV_32FC3);
          cv::split(in_32f, RGB_planes);
          // gray_img1 = cv::Mat::zeros(img1.cols, img1.rows,CV_32FC1);
          gray_img1 = (RGB_planes[0] + RGB_planes[1] + RGB_planes[2]) / 3.0 ;
          gray_img1.convertTo(gray_in_img,CV_8UC1);
        } else {
          gray_in_img = img1;
        }

      clahe->apply(gray_in_img,img1_clahe);
      ImgRep1 = ImageRepresentation(img1_clahe,Config1.CLIparams.img1_fname);



      double time2 = ((double)(getMilliSecs() - clahe_start))/1000;
      if (VERB) std::cerr << " CLAHE done in "  << time2<< " seconds" << endl;

  }
  else
  {
    ImgRep1 = ImageRepresentation(img1,Config1.CLIparams.img1_fname);
  }

  int final_step = 0;
  int curr_matches = 0;

  /// Affine regions detection
  std::cerr << "View synthesis, detection and description..." << endl;

  /// Main program loop
  //for (int step=0; (step < Config1.Matchparam.maxSteps)
  //                 && (curr_matches < Config1.Matchparam.minMatches); step++, final_step++)
  //{
  for (int step=0; step<4; step++, final_step++)
  {    
    double parallel_curr_start = getMilliSecs();

        if (VERB)
        {
          std::cerr << "Iteration " << step << std::endl;
          for (unsigned int det=0; det < DetectorNames.size(); det++)
          {
            unsigned int n_synths = Config1.ItersParam[step][DetectorNames[det]].size();
            if (n_synths > 0)
              std::cerr << DetectorNames[det] << ": " << n_synths << " synthesis will be done." << std::endl;
          }
        }

    ImgRep1.SynthDetectDescribeKeypoints(Config1.ItersParam[step],
                                               Config1.DetectorsPars,
                                               Config1.DescriptorPars,
                                               Config1.DomOriPars);
    TimeLog img1time = ImgRep1.GetTimeSpent();
    //std::cerr << "Writing files... " << endl;
    //ImgRep1.SaveDescriptorsBenchmark(Config1.CLIparams.k1_fname);
    //TODO store all descs
    for (std::map<std::string, AffineRegionVectorMap>::const_iterator
           reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end();  ++reg_it) {
          for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
               desc_it != reg_it->second.end(); ++desc_it) {

                if (desc_it->first == "None") {
                  continue;
                }

                if(desc_it->first == "RootSIFT"){
                    int num_keys = desc_it->second.size();
                    for (int i = 0; i < num_keys ; i++ ) {
                    AffineRegion ar = desc_it->second[i];
                    cout<<ar.desc.vec.size()<<" ";
                    for (int ddd = 0; ddd < ar.desc.vec.size(); ++ddd){
                        RootSIFTdesc.push_back(ar.desc.vec[ddd]);
                        //kpfile << ar.desc.vec[ddd] << " ";
                        }
                    //kpfile << std::endl;
                    }
                }
                else if(desc_it->first == "HalfRootSIFT"){
                    int num_keys = desc_it->second.size();
                    for (int i = 0; i < num_keys ; i++ ) {
                    AffineRegion ar = desc_it->second[i];
                    cout<<ar.desc.vec.size()<<" ";
                    for (int ddd = 0; ddd < ar.desc.vec.size(); ++ddd){
                        HalfRootSIFTdesc.push_back(ar.desc.vec[ddd]);
                        //kpfile << ar.desc.vec[ddd] << " ";
                        }
                    //kpfile << std::endl;
                    }
                }
            }
        }

  }

  //log1.UnorientedReg1 = ImgRep1.GetRegionsNumber();
  //log1.OrientReg1 = ImgRep1.GetDescriptorsNumber() - ImgRep1.GetDescriptorsNumber("None");
  //log1.FinalStep = final_step;
  std::cerr << "Done in " << 4 << " iterations" << endl;
  std::cerr << "*********************" << endl;

  return 0;
}


