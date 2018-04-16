#include <iostream>
#include "BoW.hpp"
#include "cv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace BoW;
using namespace cv;
int main(){

    BagOfWords BW;
    BW.getDefaultParam(10000,10000);
    string imgPath="/home/jun/ImageDataSet/list.txt";
    BW.computeVocab(imgPath);
    BW.buildInvIndex(imgPath);
    Mat I = imread("/home/jun/testImage.jpg",0);
    resize(I,I,Size(64,480));

    int topn=10;

    BW.imageSearch(I,topn);

    return 0;

}
