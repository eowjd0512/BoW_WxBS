#include <iostream>
#include "cv.hpp"
#include "./thirdParty/vlfeat/vl/host.h"
#include <./thirdParty/vlfeat/vl/kdtree.h>
using namespace std;
using namespace cv;

namespace BoW{
    class invertedIndex{
    public:
    //private:
    string dirname;
    vector<string> imgPaths;
    map<int,string> imgPath2id;
    int numImgs;
    //public:
    invertedIndex(){};
    ~invertedIndex(){};

    };

}