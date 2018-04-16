#include <iostream>
#include "cv.hpp"
#include "iindex.hpp"
#include "./thirdParty/vlfeat/vl/host.h"
#include <./thirdParty/vlfeat/vl/kdtree.h>
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
    //public:
    BagOfWords(){};
    ~BagOfWords(){};

    void getDefaultParam(int numWords, int maxImgsForVocab){
    params.numWords = numWords;
    params.maxImgsForVocab = maxImgsForVocab;
    };

    int computeVocab(string imgsDir);
    void buildInvIndex(string imgsDir);
    void imageSearch();
    void visualizeMatching();

    };

}