#include <iostream>
//#include "BoW.hpp"
#include "BoW_WxBS.hpp"
#include "cv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>
using namespace std;
using namespace BoW;
using namespace cv;
int main(){

    //BagOfWords BW;
    BagOfWords_WxBS BW;
    int numImg=20;
    BW.getDefaultParam(10000,10000);
    string imgPath="/home/jun/ImageDataSet/trainImg/test.txt";
    BW.computeVocab(imgPath,numImg);
    BW.buildInvIndex(imgPath);
    string I = "/home/jun/ImageDataSet/testImg/testImage.jpg";
    //Mat I = imread("/home/jun/ImageDataSet/testImg/testImage.jpg",0);
    //resize(I,I,Size(640,480));
    
    
    int topn=20;
    if(numImg<topn)
        topn=numImg;
    clock_t begin = clock();
    BW.imageSearch(I,topn);
    clock_t end = clock();  
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<<"time: "<<elapsed_secs<<endl;
    //imshow("d",I);
    //waitKey(0);
    return 0;

}
