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
#define RSIFT
#ifdef test
int main(){
    BagOfWords_WxBS BW;
    int numImg=3757/3;
    
    string config = "/home/jun/BOW_WxBS/config_iter_mods_cviu_wxbs.ini";
    string iter = "/home/jun/BOW_WxBS/iters_mods_cviu_wxbs_2.ini";
    
    //BagOfWords BW;
    
    if(!BW.getWxBSParam(config,iter)){
        cerr<<"err in param"<<endl;
        exit(0);
    }
    BW.getDefaultParam(10000,10000);
    BW.setImgNum(numImg);

    BW.descname = "HRSIFT";
    //HRSiftHessianVoc //HRSiftMSERVoc
    //keyPointIter3Hessian //keyPointIter1MSER 
    //indexIter3HRSift //indexIter1HRSift 
    BW.loadGeneralInfo("/home/jun/BOW_WxBS/index/generalInfo.txt");
    BW.iternum = 1;
    BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/HRSiftMSERVoc.txt");
    BW.loadKeyPoints("/home/jun/BOW_WxBS/index/keyPointIter1MSER.txt");
    BW.loadIndex("/home/jun/BOW_WxBS/index/indexIter1HRSift.txt");

    
    
    //BW.buildInvIndex(imgPath,numImg,0); //0 is save, 1 is load
    //1344 <> 1119()
    //1473 <>(426) 
    //3557 <>(1086)
    //2190<>1995(665)
    string queryImg = "/home/jun/ImageDataSet/VPRiCE-dataset/live/image-02190.png";
    
    int y=665;
    BW.testMatcher(queryImg,y,100,1);
    waitKey(0);
}
 #endif

#ifdef BoW

int main(){
    BagOfWords_WxBS BW;
    int numImg=3756/3;
    
    string config = "/home/jun/BOW_WxBS/config_iter_mods_cviu_wxbs.ini";
    string iter = "/home/jun/BOW_WxBS/iters_mods_cviu_wxbs_2.ini";
    
    //BagOfWords BW;
    
    if(!BW.getWxBSParam(config,iter)){
        cerr<<"err in param"<<endl;
        exit(0);
    }
    BW.getDefaultParam(10000,10000);
    BW.setImgNum(numImg);
    
    

    
    string imgPath="/home/jun/ImageDataSet/VPRiCE-dataset/list.txt";
    clock_t begin = clock();


    #ifdef vocabsave
    BW.iternum = 1;
    BW.descname = "RSIFT";
    string Rd[2] = {"/home/jun/BOW_WxBS/vocabulary/Rdesc1.txt","/home/jun/BOW_WxBS/vocabulary/Rdesc2.txt"};
    string HRd[2] = {"/home/jun/BOW_WxBS/vocabulary/HRdesc1.txt","/home/jun/BOW_WxBS/vocabulary/HRdesc2.txt"};
    
    //BW.extractDescriptor(imgPath,0,Rd[0],HRd[0]);
    //BW.extractDescriptor(imgPath,numImg/2,Rd[1],HRd[1]);

    BW.computeVocabWithoutExtractor("RSIFT",Rd);
    //BW.computeVocabWithoutExtractor("HRSIFT",HRd);

    //BW.computeVocab(imgPath,numImg);

    BW.descname = "RSIFT";
    BW.saveVocab("/home/jun/BOW_WxBS/vocabulary/RSiftMSERVocFullImages_test3.txt");
    
    //BW.descname = "HRSIFT";
    //BW.computeVocab(imgPath,numImg);
    //BW.saveVocab("/home/jun/BOW_WxBS/vocabulary/HRSiftMSERVocFullImages.txt");
    return 0;
    //BW.iternum = 3;
    //BW.descname = "RSIFT";
    //BW.computeVocab(imgPath,numImg);
//
    ////BW.descname = "RSIFT";
    //BW.saveVocab("/home/jun/BOW_WxBS/vocabulary/RSiftHessianVoc.txt");
    //
    //BW.descname = "HRSIFT";
    //BW.computeVocab(imgPath,numImg);
    //BW.saveVocab("/home/jun/BOW_WxBS/vocabulary/HRSiftHessianVoc.txt");
    //BW.buildInvIndex(imgPath,numImg,0); //0 is save, 1 is load
    #endif



    #ifdef vocabload
    //voc_2iters_120image_WxBS
    //voc_4iters_50image_WxBS
    //voc_4iters_120image_WxBS
    //voc_1ters
    //voc_1iters_56image_WxBS
    BW.iternum = 1;
    
    //BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/HRSiftMSERVoc.txt");
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
    BW.descname = "RSIFT";
    BW.iternum = 1;
    BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/RSiftMSERVocFullImages.txt");
    BW.buildInvIndex(imgPath,numImg,1);
    BW.saveIndex("/home/jun/BOW_WxBS/index/new/indexRSiftFullImages1.txt");
    
    BW.buildInvIndex(imgPath,numImg,2);
    BW.saveIndex("/home/jun/BOW_WxBS/index/new/indexRSiftFullImages2.txt");

    BW.buildInvIndex(imgPath,numImg,3);
    BW.saveIndex("/home/jun/BOW_WxBS/index/new/indexRSiftFullImages3.txt");
    return 0;
    BW.descname = "HRSIFT";

    BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/HRSiftMSERVocFullImages.txt");
    //BW.buildInvIndex(imgPath,numImg,1);
    //BW.saveIndex("/home/jun/BOW_WxBS/index/indexHRSiftFullImages1.txt");

    BW.buildInvIndex(imgPath,numImg,2);
    BW.saveIndex("/home/jun/BOW_WxBS/index/new/indexHRSiftFullImages2.txt");

    BW.buildInvIndex(imgPath,numImg,3);
    BW.saveIndex("/home/jun/BOW_WxBS/index/new/indexHRSiftFullImages3.txt");

    cout<<"done"<<endl;
    return 0;
    //BW.iternum = 3;
    //BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/HRSiftHessianVoc.txt");
    //BW.buildInvIndex(imgPath,numImg,1);
    //BW.saveIndex("/home/jun/BOW_WxBS/index/indexIter3HRSift.txt");
    
    
    //BW.descname = "RSIFT";
    //BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/RSiftMSERVoc.txt");
    //BW.iternum = 1;
    //BW.buildInvIndex(imgPath,numImg,1);
    //BW.saveIndex("/home/jun/BOW_WxBS/index/indexIter1RSift.txt");
    
    //BW.iternum = 3;
    //BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/RSiftHessianVoc.txt");
    //BW.buildInvIndex(imgPath,numImg,1);
    //BW.saveIndex("/home/jun/BOW_WxBS/index/indexIter3RSift.txt");
   

    //BW.saveGeneralInfo("/home/jun/BOW_WxBS/index/generalInfo3.txt",numImg*2);
    //BW.saveKeyPoints("/home/jun/BOW_WxBS/index/keyPointMSER3.txt");
    //BW.saveIndex("/home/jun/BOW_WxBS/index/indexIter1RSift3.txt");
    #endif 

    #ifdef indexload
     string generalinfo[3] = {"/home/jun/BOW_WxBS/index/generalInfo1.txt",
                            "/home/jun/BOW_WxBS/index/generalInfo2.txt",
                            "/home/jun/BOW_WxBS/index/generalInfo3.txt"};
    string keypoint[3] = {"/home/jun/BOW_WxBS/index/keyPointMSER1.txt",
                        "/home/jun/BOW_WxBS/index/keyPointMSER2.txt",
                        "/home/jun/BOW_WxBS/index/keyPointMSER3.txt"};
    string Rindex[3]={"/home/jun/BOW_WxBS/index/new/indexRSiftFullImages1.txt",
                    "/home/jun/BOW_WxBS/index/new/indexRSiftFullImages2.txt",
                    "/home/jun/BOW_WxBS/index/new/indexRSiftFullImages3.txt"};  
    string HRindex[3]={"/home/jun/BOW_WxBS/index/new/indexHRSiftFullImages1.txt",
                    "/home/jun/BOW_WxBS/index/new/indexHRSiftFullImages2.txt",
                    "/home/jun/BOW_WxBS/index/new/indexHRSiftFullImages3.txt"};  
    
    BW.iternum = 1;
    //BW.descname = "RSIFT";
    //HRSiftHessianVoc //HRSiftMSERVoc
    //keyPointIter3Hessian //keyPointIter1MSER 
    //indexIter3HRSift //indexIter1HRSift 
    
    ///* samples 3270/3 */
    //BW.loadGeneralInfo("/home/jun/BOW_WxBS/index/generalInfo1.txt");
    //BW.loadKeyPoints("/home/jun/BOW_WxBS/index/keyPointMSER1.txt");

    //BW.descname = "RSIFT";
    //BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/RSiftMSERVocFullImages.txt");
    //BW.loadIndex("/home/jun/BOW_WxBS/index/indexRSiftFullImages1.txt");
    
    //BW.descname = "HRSIFT";
    //BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/HRSiftMSERVocFullImages.txt");
    //BW.loadIndex("/home/jun/BOW_WxBS/index/new/indexHRSiftFullImages1.txt");

    BW.descname = "RSIFT";
    /*use ALL files 3756*/
    BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/RSiftMSERVocFullImages.txt");
    BW.loadGeneralInfo(generalinfo);
    BW.loadKeyPoints(keypoint);
    BW.loadIndex(Rindex);
    BW.descname = "HRSIFT";
    /*use ALL files 3756*/
    BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/HRSiftMSERVocFullImages.txt");
    BW.loadIndex(HRindex);
    #endif 
    BW.grouping("/home/jun/BOW_WxBS/index/groupInfo.txt",0.52);

    //BW.buildInvIndex(imgPath,numImg,0); //0 is save, 1 is load
    
    //string I = "/home/jun/ImageDataSet/trainImg/20180409_122718.jpg";
    //Mat I = imread("/home/jun/ImageDataSet/testImg/testImage.jpg",0);
    //resize(I,I,Size(640,480));
    
    BW.descname = "ALL";
    

    while(1){
    cout<<BW.descname<<endl;
    string name;
    cout<<"input the query image number: ";
    cin>>name;
    cout<<endl;
    
        int topn=100;
    if(numImg<topn)
        topn=numImg;
    begin = clock();

    string I = "/home/jun/ImageDataSet/VPRiCE-dataset/live/image-0"+name+".png";
    //BW.imageSearchUsingBoW(I,topn);
    cout<<I<<endl;
    BW.imageSearchUsingWxBSMatcher(I,topn,100,50);

    end = clock(); 

    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<<"time: "<<elapsed_secs<<endl;
    
    }
    //imshow("d",I);
    waitKey(0);
    return 0;

}
#endif