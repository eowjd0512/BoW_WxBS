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
    vector<vector<nodes>> RSIFTregionVector;
    vector<vector<nodes>> HRSIFTregionVector;
    //VlSiftFilt* sift;
    VlKMeans * kmeans;
    TimeLog TimingLog;
    logs log1;
    /// Parameters reading
    configs Config1;
    string descname;
    int iternum;
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
    void loadIndex(string name[]);

    void saveGeneralInfo(string name,int startnum);
    void loadGeneralInfo(string name);
    void loadGeneralInfo(string name[]);

    void saveKeyPoints(string name);
    void loadKeyPoints(string name);
    void loadKeyPoints(string name[]);
    void grouping(string name,float threshold);

    void extractDescriptor(string imgPath, int start,string R,string HR,int num,string location);
    void computeVocabWithoutExtractor(string desc, string R[]);

    int computeVocab(string imgsDir, int numImg);
    void buildInvIndex(string imgsDir,int numImg,int flag);
    void imageSearchUsingBoW(string I,int topn);
    void imageSearchUsingWxBSMatcher(string I,int topn,array<float,3756>& updateDistribution,int n,int m, bool initflag);
    
    void visualizeMatching();
    void computeImageRep(string I,vector<nodes> &RSIFTbinlist,vector<nodes> &HRSIFTbinlist,int &num, int flag,int n);
    int findCorrespondFeatures(vector<nodes> d1,vector<nodes>d2, vector<nodes> &out1,vector<nodes> &out2,multimap<int,int> matchlist);
    int findCorrespondFeatures(vector<nodes> RSIFTd1,vector<nodes> HRSIFTd1,vector<nodes>RSIFTd2, 
            vector<nodes> &out1,vector<nodes> &out2,multimap<int,int> RSIFTmatchlist,multimap<int,int> HRSIFTmatchlist);
    double calScore(TentativeCorrespListExt tentatives, double* H, int y);


    void testMatcher(string I,int y,int n, int m);    
    void drawMatcher(Mat query,Mat DB,Mat result,TentativeCorrespListExt matchList,string name);


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

    
    };

}