#pragma once
#include <iostream>
#include "cv.hpp"
#include "iindex.hpp"
#include <host.h>
#include <kdtree.h>
#include <src/generic-driver.h>
#include "detectors/structures.hpp"
#include "generic.h"
#include "sift.h"
#include "mathop.h"
#include <kmeans.h>

#include "io_mods.h"

using namespace std;
using namespace cv;

namespace BoW{
    struct param{
        int numWords;
        int maxImgsForVocab;
        string img1;
        string img2;
        string out1;
        string out2;
        string k1;
        string k2;
        string matching;
        string log;
        string logonly;
        string ver_type;
        string gt;
        string config;
        string iter;
    };
    struct model{
        int vocabSize;
        VlKDForest* RootSIFTkdtree;
        VlKDForest* HalfRootSIFTkdtree;
        //vocab;
        //kdtree;
    };
    struct nodes{
        VlKDForestNeighbor * bin =(VlKDForestNeighbor *)vl_malloc(1*sizeof(VlKDForestNeighbor));
        AffineRegion region;
    };

    class BagOfWords_WxBS{
    public:
    //private:
    param params;
    model models;
    invertedIndex index;
    vector<vector<nodes>> regionVector;
    //VlSiftFilt* sift;
    VlKMeans * kmeans;
    TimeLog TimingLog;
    logs log1;
    /// Parameters reading
    configs Config1;

    //public:
    BagOfWords_WxBS(){
        int width = 640;
        int height = 480;
        int noctaves = 5;
        //int noctaves = log2(min(width,height));
        int nlevels = 3;
        int o_min= 0;
        
        vl_size ntrees = 3;
        //sift = vl_sift_new(width, height, noctaves, nlevels, o_min);
        kmeans = vl_kmeans_new (VL_TYPE_FLOAT, VlDistanceL2) ;
        models.RootSIFTkdtree =  vl_kdforest_new(VL_TYPE_FLOAT,128,ntrees,VlDistanceL2);
        models.HalfRootSIFTkdtree =  vl_kdforest_new(VL_TYPE_FLOAT,64,ntrees,VlDistanceL2);
        params.img1="";
        params.img2="";
        params.out1="";
        params.out2="";
        params.k1="";
        params.k2="";
        params.matching="";
        params.log="";
        params.gt="";
        params.logonly="0";
        params.ver_type="0";
    };
    ~BagOfWords_WxBS(){
        //vl_sift_delete(sift);
        vl_kmeans_delete(kmeans);
        vl_kdforest_delete(models.RootSIFTkdtree);
        vl_kdforest_delete(models.HalfRootSIFTkdtree);
    };
    
    void getDefaultParam(int numWords, int maxImgsForVocab){
    params.numWords = numWords;
    params.maxImgsForVocab = maxImgsForVocab;
    };
    void setImgNum(int numImg){
        index.numImgs = numImg;
        models.vocabSize = params.numWords;
    }

    void saveVocab(string name);
    void loadVocab(string name);

    void saveIndex(string name);
    void loadIndex(string name);

    int computeVocab(string imgsDir, int numImg);
    void buildInvIndex(string imgsDir,int numImg,int flag);
    void imageSearchUsingBoW(string I,int topn);
    void imageSearchUsingWxBSMatcher(string I,int topn);
    void visualizeMatching();
    vector<nodes> computeImageRep(string I,int &num, int flag);
    int findCorrespondFeatures(vector<nodes> d1,vector<nodes>d2, vector<nodes> &out1,vector<nodes> &out2,multimap<int,int> matchlist);
    
    double calScore(vector<nodes> d1,vector<nodes> d2, double* H);
    
    int WxBSdet_desc(string path, std::vector<AffineRegion> &RootSIFTdesc, std::vector<AffineRegion> &HalfRootSIFTdesc);
    int getWxBSParam(string config,string iters){
        char ** argv_ = (char**)malloc(14*sizeof(char*));
        for(int i=0;i<14;i++)
           argv_[i] = (char*)malloc(100*sizeof(char));
        strcpy(argv_[1], params.img1.c_str());
        strcpy(argv_[2], params.img2.c_str());
        strcpy(argv_[3], params.out1.c_str());
        strcpy(argv_[4], params.out2.c_str());
        strcpy(argv_[5], params.k1.c_str());
        strcpy(argv_[6], params.k2.c_str());
        strcpy(argv_[7], params.matching.c_str());
        strcpy(argv_[8], params.log.c_str());
        strcpy(argv_[9], params.logonly.c_str());
        strcpy(argv_[10], params.ver_type.c_str());
        strcpy(argv_[11], params.gt.c_str());
        strcpy(argv_[12], config.c_str());
        strcpy(argv_[13], iters.c_str());
        // do stuff
        
        if (getCLIparam(Config1,14,argv_)) return 1;
        //free(argv_[0]);
        for(int i=0;i<14;i++)
            free(argv_[i]);
        free(argv_);

        return 1;
    }

    enum detector_type {DET_HESSIAN = 0,
                    DET_DOG = 1,
                    DET_HARRIS = 2,
                    DET_MSER = 3,
                    DET_ORB = 4,
                    DET_FAST = 5,
                    DET_SURF = 6,
                    DET_STAR = 7,
                    DET_BRISK = 8,
                    DET_KAZE = 9,
                    DET_FOCI = 10,
                    DET_CAFFE = 11,
                    DET_READ = 12,
                    DET_WAVE = 13,
                    DET_WASH = 14,
                    DET_SFOP = 15,
                    DET_TILDE = 16,
                    DET_TILDE_PLUGIN= 17,
                    DET_SADDLE = 18,
                    DET_TOS_MSER = 19,
                    DET_MIK_MSER = 20,
                    DET_UNKNOWN = 1000};
    };

}