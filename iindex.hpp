#pragma once
#include <iostream>
#include "cv.hpp"
#include "./thirdParty/vlfeat/vl/host.h"
#include <./thirdParty/vlfeat/vl/kdtree.h>
#include <map>
using namespace std;
using namespace cv;

namespace BoW{
    class invertedIndex{
    public:
    //private:
    string dirname;
    //vector<string> imgPaths;
    map<int,string> imgPath2id;
    //vector<int> totalDescriptors;
    vector<map<int,int>> vw2imgsList;
    vector<map<int,vector<int>>> matchlist;
    int numImgs;
    //public:
    invertedIndex(){};
    ~invertedIndex(){};

    };

}