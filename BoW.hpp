#include <iostream>
#include "cv.hpp"
#include "iindex.hpp"
#include "./thirdParty/vlfeat/vl/host.h"
#include <./thirdParty/vlfeat/vl/kdtree.h>
#include "./thirdParty/vlfeat/src/generic-driver.h"
#include "./thirdParty/vlfeat/vl/generic.h"
#include "./thirdParty/vlfeat/vl/sift.h"
#include "./thirdParty/vlfeat/vl/mathop.h"
#include <./thirdParty/vlfeat/vl/kmeans.h>
using namespace std;
using namespace cv;

namespace BoW{
    struct param{
        int numWords;
        int maxImgsForVocab;
    };
    struct model{
        int vocabSize;
        double  enrgy;
		const void *  centers;
        VlKDForest* kdtree;
        //vocab;
        //kdtree;
    };
    class BagOfWords{
    public:
    //private:
    param params;
    model models;
    invertedIndex index;
    VlSiftFilt* sift;
    VlKMeans * kmeans;
    //public:
    BagOfWords(){
        int width = 640;
        int height = 480;
        int noctaves = 5;
        //int noctaves = log2(min(width,height));
        int nlevels = 3;
        int o_min= 0;
        
        vl_size ntrees = 3;
        sift = vl_sift_new(width, height, noctaves, nlevels, o_min);
        kmeans = vl_kmeans_new (VL_TYPE_DOUBLE, VlDistanceL2) ;
        models.kdtree =  vl_kdforest_new(VL_TYPE_FLOAT,128,ntrees,VlDistanceL2);
    };
    ~BagOfWords(){
        vl_sift_delete(sift);
        vl_kmeans_delete(kmeans);
        vl_kdforest_delete(models.kdtree);
    };

    void getDefaultParam(int numWords, int maxImgsForVocab){
    params.numWords = numWords;
    params.maxImgsForVocab = maxImgsForVocab;
    };

    int computeVocab(string imgsDir);
    void buildInvIndex(string imgsDir);
    void imageSearch();
    void visualizeMatching();
    void computeImageRep();
    };

}