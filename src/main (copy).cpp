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
    string queryList[56] = {"1401","1426","1470","1500","1525","1551","1590","1608",
    "1680","1720","1762","1910","1994","2092","2190","2487","2505","2517","2532","2543",
    "2557","2579","2585","2596","2623","2637","2671","3062","3075","3097","3157","3164",
    "3176","3194","3380","3416","3460","3475","3509","3532","3557","3562","3574","3637",
    "3682","3736","3768","3779","3785","3812","3833","3842","3860","3914","3951","3997"};
    BW.descname = "ALL";
    //index.numImgs = 3756/3;
    int sample = 100;
    int threshold = 30;
    bool refinement= false;
    /*
    string savef = "/home/jun/BOW_WxBS/result/our_threshold30_sample100_noRefinement.txt";
    for(int i=0; i < 56; i++){
        imageSearchTest(filename, queryList[i],sample,threshold,refinement);
    }

    threshold = 40;
    savef = "/home/jun/BOW_WxBS/result/our_threshold40_sample100_noRefinement.txt";
    for(int i=0; i < 56; i++){
        imageSearchTest(filename, queryList[i],sample,threshold,refinement);
    }

    sample = 50;
    threshold = 50;
    savef = "/home/jun/BOW_WxBS/result/our_threshold50_sample50_noRefinement.txt";
    for(int i=0; i < 56; i++){
        imageSearchTest(filename, queryList[i],sample,threshold,refinement);
    }*/
    array<float,3756> updateDistribution;
    updateDistribution.fill(1.0);
        
    //*
    while(1){
        cout<<BW.descname<<endl;
        string name;
        cout<<"input the query image number('r' is rest): ";
        cin>>name;
        cout<<endl;
        if(name=="r"){
            for(int i=0;i<3756;i++)updateDistribution[i]=1.0;
            cout<<"input the query image number: ";
            cin>>name;
            cout<<endl;
            refinement=true;
        } 
        
            int topn=100;
        if(numImg<topn)
            topn=numImg;
        //begin = clock();

        string I = "/home/jun/ImageDataSet/VPRiCE-dataset/live/image-0"+name+".png";
        //BW.imageSearchUsingBoW(I,topn);
        cout<<I<<endl;
        ofstream f(I);
        BW.imageSearchUsingWxBSMatcher(I,topn,updateDistribution,200,50,refinement);
        refinement=false;//*/
        //end = clock(); 
        f.close();
        //elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        //cout<<"time: "<<elapsed_secs<<endl;
        
    }
    //imshow("d",I);
    waitKey(0);
    return 0;

}
#endif