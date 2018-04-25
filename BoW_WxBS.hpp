#include <iostream>
#include "cv.hpp"
#include "iindex.hpp"
#include <host.h>
#include <kdtree.h>
#include <src/generic-driver.h>
#include "generic.h"
#include "sift.h"
#include "mathop.h"
#include <kmeans.h>
using namespace std;
using namespace cv;

namespace BoW{
    struct param{
        int numWords;
        int maxImgsForVocab;
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
    };

    class BagOfWords_WxBS{
    public:
    //private:
    param params;
    model models;
    invertedIndex index;
    //VlSiftFilt* sift;
    VlKMeans * kmeans;
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
    void saveVocab(string name);
    void loadVocab(string name);
    int computeVocab(string imgsDir, int numImg);
    void buildInvIndex(string imgsDir,int numImg,int flag);
    void imageSearch(string I,int topn);
    void visualizeMatching();
    nodes* computeImageRep(string I,int &num);
    
    };

}