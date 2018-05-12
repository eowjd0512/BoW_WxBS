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

#define vocabload
#define indexload
#define _OPENMP
#define BoW

#ifdef test
int main(){

}
#endif

#ifdef BoW

void getParam(){

}

int main(){
    BagOfWords_WxBS BW;
    int numImg=120;

    string config = "/home/jun/BOW_WxBS/config_iter_mods_cviu_wxbs.ini";
    string iter = "/home/jun/BOW_WxBS/iters_mods_cviu_wxbs_2.ini";
    
    //BagOfWords BW;
    
    if(!BW.getWxBSParam(config,iter)){
        cerr<<"err in param"<<endl;
        exit(0);
    }
    BW.getDefaultParam(10000,10000);
    BW.setImgNum(numImg);

    string imgPath="/home/jun/ImageDataSet/trainImg/test.txt";
    clock_t begin = clock();


    #ifdef vocabsave
    BW.computeVocab(imgPath,numImg);
    BW.saveVocab("/home/jun/BOW_WxBS/vocabulary/voc_1iters_120_test.txt");
    BW.buildInvIndex(imgPath,numImg,0); //0 is save, 1 is load
    #endif

    #ifdef vocabload
    //voc_2iters_120image_WxBS
    //voc_4iters_50image_WxBS
    //voc_4iters_120image_WxBS
    //voc_1ters
    //voc_1iters_56image_WxBS
    BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/voc_1iters_120_test.txt");
    //BW.buildInvIndex(imgPath,numImg,1); //0 is save, 1 is load
    #endif
clock_t end = clock();  
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<<"get vocab time: "<<elapsed_secs<<endl;

    #ifdef indexsave
    //index_2iters_120image_WxBS
    //index_4iters_50image_WxBS
    //index_1ters
    //index_4iters_120image_WxBS
    BW.saveIndex("/home/jun/BOW_WxBS/index/index_1iters_120_test.txt");
    #endif 

    #ifdef indexload
    BW.loadIndex("/home/jun/BOW_WxBS/index/index_1iters_120_test.txt");
    #endif 

    //BW.buildInvIndex(imgPath,numImg,0); //0 is save, 1 is load
    string I = "/home/jun/ImageDataSet/testImg/testImage4.jpg";
    //string I = "/home/jun/ImageDataSet/trainImg/20180409_122718.jpg";
    //Mat I = imread("/home/jun/ImageDataSet/testImg/testImage.jpg",0);
    //resize(I,I,Size(640,480));
    
    
    int topn=20;
    if(numImg<topn)
        topn=numImg;
    begin = clock();

    //BW.imageSearchUsingBoW(I,topn);
    BW.imageSearchUsingWxBSMatcher(I,topn);

    end = clock();  
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<<"time: "<<elapsed_secs<<endl;
    //imshow("d",I);
    //waitKey(0);
    return 0;

}
#endif