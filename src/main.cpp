#include <iostream>
#include <fstream>

//#include "BoW.hpp"
#include "BoW_WxBS.hpp"
#include "cv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>
#include <string>

using namespace std;
using namespace BoW;
using namespace cv;

#define vocabload
#define indexload
#define _OPENMP
#define BoW
#define RSIFT
#define WxBS_BSoW


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
    //int numImg=4022;
    string config = "/home/jun/BOW_WxBS/config_iter_mods_cviu_wxbs.ini";
    string iter = "/home/jun/BOW_WxBS/iters_mods_cviu_wxbs_2.ini";
    
    //BagOfWords BW;
    
    if(!BW.getWxBSParam(config,iter)){
        cerr<<"err in param"<<endl;
        exit(0);
    }
    BW.getDefaultParam(10000,10000);
    BW.setImgNum(numImg);
    
    

    
    string imgPath="/home/jun/ImageDataSet/VPRiCE-dataset/list_live.txt";
    string imgPath2="/home/jun/ImageDataSet/VPRiCE-dataset/list.txt";
    clock_t begin = clock();


    #ifdef vocabsave
    BW.iternum = 1;
    BW.descname = "HRSIFT";
    string Rd[4] = {"/home/jun/BOW_WxBS/vocabulary/Rdesc1.txt","/home/jun/BOW_WxBS/vocabulary/Rdesc2.txt"
                    ,"/home/jun/BOW_WxBS/vocabulary/Rdesc3.txt","/home/jun/BOW_WxBS/vocabulary/Rdesc4.txt"};
    string HRd[4] = {"/home/jun/BOW_WxBS/vocabulary/HRdesc1.txt","/home/jun/BOW_WxBS/vocabulary/HRdesc2.txt"
                    ,"/home/jun/BOW_WxBS/vocabulary/HRdesc3.txt","/home/jun/BOW_WxBS/vocabulary/HRdesc4.txt"};
    
    //BW.extractDescriptor(imgPath2,0,Rd[0],HRd[0],3756,"memory");
    //BW.extractDescriptor(imgPath2,3756/2,Rd[1],HRd[1],3756,"memory");
    //BW.extractDescriptor(imgPath,0,Rd[2],HRd[2],4022,"live");
    //BW.extractDescriptor(imgPath,4022/2,Rd[3],HRd[3],4022,"live");


    BW.computeVocabWithoutExtractor("RSIFT",Rd);

    //BW.descname = "RSIFT";
    //BW.saveVocab("/home/jun/BOW_WxBS/vocabulary/10K/RSiftMSERVocFullImages10K.txt");


    

    BW.computeVocabWithoutExtractor("HRSIFT",HRd);
    BW.saveVocab("/home/jun/BOW_WxBS/vocabulary/10K/HRSiftMSERVocFullImages10K.txt");
    //BW.computeVocab(imgPath,numImg);

    //BW.descname = "HRSIFT";
    //BW.computeVocab(imgPath,numImg);
    //BW.saveVocab("/home/jun/BOW_WxBS/vocabulary/new/HRSiftHessianVocFullImages1M.txt");
    //return 0;
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
    cout<<"RSIFT MSER"<<endl;
    BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/10K/RSiftMSERVocFullImages10K.txt");
    BW.buildInvIndex(imgPath2,numImg,1);
    BW.saveKeyPoints("/home/jun/BOW_WxBS/index/10K/10K_keyPointMSER1.txt");
    BW.saveIndex("/home/jun/BOW_WxBS/index/10K/10K_RSiftindexMSER1.txt");
    return 0;
    BW.buildInvIndex(imgPath2,numImg,2);
    BW.saveKeyPoints("/home/jun/BOW_WxBS/index/10K/10K_keyPointMSER2.txt");
    BW.saveIndex("/home/jun/BOW_WxBS/index/10K/10K_RSiftindexMSER2.txt");

    BW.buildInvIndex(imgPath2,numImg,3);
    BW.saveKeyPoints("/home/jun/BOW_WxBS/index/10K/10K_keyPointMSER3.txt");
    BW.saveIndex("/home/jun/BOW_WxBS/index/10K/10K_RSiftindexMSER3.txt");
    
    

    BW.descname = "HRSIFT";
    cout<<"HRSIFT MSER"<<endl;
    BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/10K/HRSiftMSERVocFullImages10K.txt");
    BW.buildInvIndex(imgPath2,numImg,1);
    BW.saveIndex("/home/jun/BOW_WxBS/index/10K/10K_HRSiftindexMSER1.txt");

    BW.buildInvIndex(imgPath2,numImg,2);
    BW.saveIndex("/home/jun/BOW_WxBS/index/10K/10K_HRSiftindexMSER2.txt");

    BW.buildInvIndex(imgPath2,numImg,3);
    BW.saveIndex("/home/jun/BOW_WxBS/index/10K/10K_HRSiftindexMSER3.txt");

    

    BW.descname = "RSIFT";
    BW.iternum = 2;
    cout<<"RSIFT Hessian"<<endl;
    BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/1M/RSiftHessianVocFullImages1M.txt");
    BW.buildInvIndex(imgPath2,numImg,1);
    BW.saveKeyPoints("/home/jun/BOW_WxBS/index/1M/keyPointHessian1.txt");
    BW.saveIndex("/home/jun/BOW_WxBS/index/1M/RSiftindexHessian1.txt");
    
    BW.buildInvIndex(imgPath2,numImg,2);
    BW.saveKeyPoints("/home/jun/BOW_WxBS/index/1M/keyPointHessian2.txt");
    BW.saveIndex("/home/jun/BOW_WxBS/index/1M/RSiftindexHessian2.txt");

    BW.buildInvIndex(imgPath2,numImg,3);
    BW.saveKeyPoints("/home/jun/BOW_WxBS/index/1M/keyPointHessian3.txt");
    BW.saveIndex("/home/jun/BOW_WxBS/index/1M/RSiftindexHessian3.txt");
    
    BW.descname = "HRSIFT";
    cout<<"RSIFT Hessian"<<endl;
    BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/1M/HRSiftHessianVocFullImages1M.txt");
    BW.buildInvIndex(imgPath,numImg,1);
    BW.saveIndex("/home/jun/BOW_WxBS/index/1M/HRSiftindexHessian1.txt");

    BW.buildInvIndex(imgPath2,numImg,2);
    BW.saveIndex("/home/jun/BOW_WxBS/index/1M/HRSiftindexHessian2.txt");

    BW.buildInvIndex(imgPath2,numImg,3);
    BW.saveIndex("/home/jun/BOW_WxBS/index/1M/HRSiftindexHessian3.txt");

    cout<<"done"<<endl;
    
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
    string keypoint[3] = {"/home/jun/BOW_WxBS/index/10K/10K_keyPointMSER1.txt",
                        "/home/jun/BOW_WxBS/index/10K/10K_keyPointMSER2.txt",
                        "/home/jun/BOW_WxBS/index/10K/10K_keyPointMSER3.txt"};
    string Rindex[3]={"/home/jun/BOW_WxBS/index/10K/10K_RSiftindexMSER1.txt",
                    "/home/jun/BOW_WxBS/index/10K/10K_RSiftindexMSER2.txt",
                    "/home/jun/BOW_WxBS/index/10K/10K_RSiftindexMSER3.txt"};  
    string HRindex[3]={"/home/jun/BOW_WxBS/index/10K/10K_HRSiftindexMSER1.txt",
                    "/home/jun/BOW_WxBS/index/10K/10K_HRSiftindexMSER2.txt",
                    "/home/jun/BOW_WxBS/index/10K/10K_HRSiftindexMSER3.txt"};  
    
    BW.iternum = 1;
    //BW.descname = "RSIFT";
    //HRSiftHessianVoc //HRSiftMSERVoc
    //keyPointIter3Hessian //keyPointIter1MSER 
    //indexIter3HRSift //indexIter1HRSift 
    
    ///* samples 3270/3 */
    //BW.loadGeneralInfo("/home/jun/BOW_WxBS/index/generalInfo1.txt");
    //BW.loadKeyPoints("/home/jun/BOW_WxBS/index/1M/1M_keyPointMSER1.txt");

    //BW.descname = "RSIFT";
    //BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/1M/RSiftMSERVocFullImages1M.txt");
    //BW.loadIndex("/home/jun/BOW_WxBS/index/1M/1M_RSiftindexMSER1.txt");
    
    //BW.descname = "HRSIFT";
    //BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/HRSiftMSERVocFullImages.txt");
    //BW.loadIndex("/home/jun/BOW_WxBS/index/new/indexHRSiftFullImages1.txt");
//*
    BW.descname = "RSIFT";
    //use ALL files 3756
    BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/10K/RSiftMSERVocFullImages10K.txt");
    BW.loadGeneralInfo(generalinfo);
    BW.loadKeyPoints(keypoint);
    BW.loadIndex(Rindex);
    BW.descname = "HRSIFT";
    //use ALL files 3756
    BW.loadVocab("/home/jun/BOW_WxBS/vocabulary/10K/HRSiftMSERVocFullImages10K.txt");
    BW.loadIndex(HRindex);
    //*/
    #endif 

    //BW.grouping("/home/jun/BOW_WxBS/index/groupInfo.txt",0.52);

    //BW.buildInvIndex(imgPath,numImg,0); //0 is save, 1 is load
    
    //string I = "/home/jun/ImageDataSet/trainImg/20180409_122718.jpg";
    //Mat I = imread("/home/jun/ImageDataSet/testImg/testImage.jpg",0);
    //resize(I,I,Size(640,480));
    int topn=100;
    BW.descname = "ALL";
    string queryList[56] = {"1401","1426","1470","1500","1525","1551","1590","1608",
    "1680","1720","1762","1910","1994","2092","2190","2487","2505","2517","2532","2543",
    "2557","2579","2585","2596","2623","2637","2671","3062","3075","3097","3157","3164",
    "3176","3194","3380","3416","3460","3475","3509","3532","3557","3562","3574","3637",
    "3682","3736","3768","3779","3785","3812","3833","3842","3860","3914","3951","3997"};
    
    string groundTruth[56] = {"1206","1237","1275","1305","1329","1356","1395","1412",
    "1485","1524","1567","1716","1799","1897","1995","2292","2310","2325","2334","2349",
    "2361","2382","2390","2400","2427","2442","2475","2820","2829","2856","2923","2930",
    "2952","2957","3060","3103","3147","3165","3200","3220","3259","3264","3274","3372",
    "3417","3471","3501","3515","3519","3546","3567","3576","3594","3648","3684","3731"};
    #ifdef WxBS_BSoW
    int sample[2] = {100,150};
    int threshold[3]={40,50,60};
    
    bool refinement= false;

    for(int s = 0; s<2;s++){
        for(int th=0;th<3;th++){
            string savef = "/home/jun/BOW_WxBS/result/suggestedSampling/our_suggested_sample"+to_string(sample[s])+"_threshold"+to_string(threshold[th])+".txt";
            string saveff = "/home/jun/BOW_WxBS/result/suggestedSampling/distribution"+to_string(sample[s])+"_threshold"+to_string(threshold[th])+".txt";
            
            
            for(int i=0; i < 56; i++){
                ofstream f;
                f.open(savef,ios::app);
                f<<endl<<endl;
                f<<"query image: "<<queryList[i]<<endl;
                f<<"grdth image: "<<groundTruth[i]<<endl;
                f.close();

                ofstream ff;
                ff.open(saveff,ios::app);
                ff<<endl<<endl;
                ff<<"query image: "<<queryList[i]<<endl;
                ff<<"grdth image: "<<groundTruth[i]<<endl;
                ff.close();

                array<float,3756> updateDistribution;
                updateDistribution.fill(1.0);
                string I = "/home/jun/ImageDataSet/VPRiCE-dataset/live/image-0"+queryList[i]+".png";
                double initQuantizationTime=0;
                double initSearchingTime=0;
                BW.imageSearchUsingWxBSMatcher(savef, saveff, I,topn,updateDistribution,sample[s],threshold[th],refinement,initQuantizationTime,initSearchingTime);
            }

            
            
        }
    }
    #endif
    #ifdef WxBS_BoW
    string savef = "/home/jun/BOW_WxBS/result/WxBS_BoW_HR_RSIFT_reranking_result.txt";
    BW.descname = "ALL";        
    
    for(int i=0; i < 56; i++){
        ofstream f;
        f.open(savef,ios::app);
        f<<endl<<endl;
        f<<"query image: "<<queryList[i]<<endl;
        f<<"grdth image: "<<groundTruth[i]<<endl;
        f.close();

        //array<float,3756> updateDistribution;
        //updateDistribution.fill(1.0);
        string I = "/home/jun/ImageDataSet/VPRiCE-dataset/live/image-0"+queryList[i]+".png";
        double initQuantizationTime=0;
        double initSearchingTime=0;
        BW.imageSearchUsingBoW(savef, I,topn,initQuantizationTime,initSearchingTime);
    }

    #endif

    //imshow("d",I);
    waitKey(0);
    return 0;

}
#endif