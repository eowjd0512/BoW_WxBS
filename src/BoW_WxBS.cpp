//#pragma once
#include "BoW_WxBS.hpp"
#include "iindex.hpp"
#include <iostream>
#include <fstream>
#include "cv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "src/generic-driver.h"
#include "vl/generic.h"
#include "vl/sift.h"
#include "vl/mathop.h"
#include <vl/kmeans.h>
#include <vl/kdtree.h>
#include "detectors/structures.hpp"
#include "WxBSdet_desc.cpp"
#include "matching/matching.hpp"
#include "correspondencebank.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <ctime>       /* time */
#include <omp.h>
#include "kde.hpp"

#define nontest
using namespace std;
using namespace cv;
namespace BoW{
        struct greater
    {
        template<class T>
        bool operator()(T const &a, T const &b) const { return a > b; }
    };
    VL_INLINE void transpose_descriptor (vl_sift_pix* dst, vl_sift_pix* src)
    {
    int const BO = 8 ;  /* number of orientation bins */
    int const BP = 4 ;  /* number of spatial bins     */
    int i, j, t ;

    for (j = 0 ; j < BP ; ++j) {
        int jp = BP - 1 - j ;
        for (i = 0 ; i < BP ; ++i) {
        int o  = BO * i + BP*BO * j  ;
        int op = BO * i + BP*BO * jp ;
        dst [op] = src[o] ;
        for (t = 1 ; t < BO ; ++t)
            dst [BO - t + op] = src [t + o] ;
        }
    }
    }
    /*void LoadRegions(ImageRepresentation ImgRep,vector<nodes> d){
        //ImgRep.Name = 
        AffineRegionVector desc_regions;

        for(int i=0;i<d.size();i++){
            AffineRegion ar = d[i].region;
            desc_regions.push_back(ar);
        }
        ImgRep.AddRegions(desc_regions,"1","1");
    }*/
    int BagOfWords_WxBS::findCorrespondFeatures(vector<nodes> d1,vector<nodes>d2, vector<nodes> &out1,vector<nodes> &out2,multimap<int,int> matchlist){
        //find corresponding Feautres from d1 to d2
        int cnt=0;
        
        //query: i-th feature, j-th kdtree index ==  DB: multimap [j-th index] = i-th feature 
        for(int i=0; i<d1.size();i++){
            // d1[i].bin.index  is  j-th kdtree index
            int featureIndex = d1[i].bin[0].index;
            multimap<int, int>::iterator iter;
           
            //duplicate d1's some feature if that corrsponding feature of d2 is non-single 
            if(matchlist.count(featureIndex)>1){
                
                pair<map<int, int>::iterator, map<int, int>::iterator> iter_pair;
                
                iter_pair = matchlist.equal_range(featureIndex);
                //iter = matchlist.find(featureIndex);
                int dist=100000;
                int shortestIndex=0;
                for (iter = iter_pair.first; iter != iter_pair.second; ++iter){
                    int corrIndex = iter->second;

                    //double x1 = d1[i].region.reproj_kp.x;
                    //double y1 = d1[i].region.reproj_kp.y;
                    //double x2 = d2[corrIndex].region.reproj_kp.x;
                    //double y2 = d2[corrIndex].region.reproj_kp.y;
                    //double distance = sqrt(pow((x2-x1),2)+pow((y2-y1),2));
                    //
                    //if(distance<dist){
                    //    dist=distance;
                    //    shortestIndex=corrIndex;
                    //}
                    out1.push_back(d1[i]);
                    out2.push_back(d2[corrIndex]);
                    cnt++;
                }
                //out1.push_back(d1[i]);
                //out2.push_back(d2[shortestIndex]);
                //cnt++;
            }else if(matchlist.count(featureIndex)==1){
                
                iter = matchlist.find(featureIndex);
                int corrIndex = iter->second;
                out1.push_back(d1[i]);
                out2.push_back(d2[corrIndex]);
                cnt++;
            }
        }
        
        return cnt;
    }
    int BagOfWords_WxBS::findCorrespondFeatures(vector<nodes> RSIFTd1,vector<nodes> HRSIFTd1,vector<nodes>RSIFTd2, 
            vector<nodes> &out1,vector<nodes> &out2,multimap<int,int> RSIFTmatchlist,multimap<int,int> HRSIFTmatchlist){
        //find corresponding Feautres from HRSIFTd1 to RSIFTd2
        int cnt=0;
        
        //query: i-th feature, j-th kdtree index ==  DB: multimap [j-th index] = i-th feature 
        for(int i=0; i<RSIFTd1.size();i++){
            // RSIFTd1[i].bin.index  is  j-th kdtree index
            int featureIndex = RSIFTd1[i].bin[0].index;
            multimap<int, int>::iterator iter;
           
            //duplicate RSIFTd1's some feature if that corrsponding feature of RSIFTd2 is non-single 
            if(RSIFTmatchlist.count(featureIndex)>1){
                
                pair<map<int, int>::iterator, map<int, int>::iterator> iter_pair;
                
                iter_pair = RSIFTmatchlist.equal_range(featureIndex);
                //iter = RSIFTmatchlist.find(featureIndex);
                int dist=100000;
                int shortestIndex=0;
                for (iter = iter_pair.first; iter != iter_pair.second; ++iter){
                    int corrIndex = iter->second;

                    //double x1 = RSIFTd1[i].region.reproj_kp.x;
                    //double y1 = RSIFTd1[i].region.reproj_kp.y;
                    //double x2 = RSIFTd2[corrIndex].region.reproj_kp.x;
                    //double y2 = RSIFTd2[corrIndex].region.reproj_kp.y;
                    //double distance = sqrt(pow((x2-x1),2)+pow((y2-y1),2));
                    //
                    //if(distance<dist){
                    //    dist=distance;
                    //    shortestIndex=corrIndex;
                    //}
                    out1.push_back(RSIFTd1[i]);
                    out2.push_back(RSIFTd2[corrIndex]);
                    cnt++;
                }
                //out1.push_back(RSIFTd1[i]);
                //out2.push_back(RSIFTd2[shortestIndex]);
                //cnt++;
            }else if(RSIFTmatchlist.count(featureIndex)==1){
                
                iter = RSIFTmatchlist.find(featureIndex);
                int corrIndex = iter->second;
                out1.push_back(RSIFTd1[i]);
                out2.push_back(RSIFTd2[corrIndex]);
                cnt++;
            }
        }
        for(int i=0; i<HRSIFTd1.size();i++){
            // HRSIFTd1[i].bin.index  is  j-th kdtree index
            int featureIndex = HRSIFTd1[i].bin[0].index;
            multimap<int, int>::iterator iter;
           
            //duplicate HRSIFTd1's some feature if that corrsponding feature of HRSIFTd2 is non-single 
            if(HRSIFTmatchlist.count(featureIndex)>1){
                
                pair<map<int, int>::iterator, map<int, int>::iterator> iter_pair;
                
                iter_pair = HRSIFTmatchlist.equal_range(featureIndex);
                //iter = HRSIFTmatchlist.find(featureIndex);
                int dist=1000000;
                int shortestIndex=0;
                for (iter = iter_pair.first; iter != iter_pair.second; ++iter){
                    int corrIndex = iter->second;

                    //double x1 = HRSIFTd1[i].region.reproj_kp.x;
                    //double y1 = HRSIFTd1[i].region.reproj_kp.y;
                    //double x2 = HRSIFTd2[corrIndex].region.reproj_kp.x;
                    //double y2 = HRSIFTd2[corrIndex].region.reproj_kp.y;
                    //double distance = sqrt(pow((x2-x1),2)+pow((y2-y1),2));
                    //
                    //if(distance<dist){
                    //    dist=distance;
                    //    shortestIndex=corrIndex;
                    //}
                    out1.push_back(HRSIFTd1[i]);
                    out2.push_back(RSIFTd2[corrIndex]);
                    cnt++;
                }
                //out1.push_back(HRSIFTd1[i]);
                //out2.push_back(HRSIFTd2[shortestIndex]);
                //cnt++;
            }else if(HRSIFTmatchlist.count(featureIndex)==1){
                
                iter = HRSIFTmatchlist.find(featureIndex);
                int corrIndex = iter->second;
                out1.push_back(HRSIFTd1[i]);
                out2.push_back(RSIFTd2[corrIndex]);
                cnt++;
            }
        }
        
        return cnt;
    }
    double BagOfWords_WxBS::calScore(TentativeCorrespListExt tentatives, double* H,int y){
        double score=0;
        double score_=0;
        
        if(H[0]==-1||isnan(abs(H[0]))){
            return -1;
        }
        int n = tentatives.TCList.size();
        for(int i=0;i< n;i++){
            double x1 = tentatives.TCList[i].first.reproj_kp.x;
            double y1 = tentatives.TCList[i].first.reproj_kp.y;
            double x2 = tentatives.TCList[i].second.reproj_kp.x;
            double y2 = tentatives.TCList[i].second.reproj_kp.y;
            double x_ = (x1*H[0]+y1*H[1]+H[2])/(H[6]+H[7]+H[8]);
            double y_ = (x1*H[3]+y1*H[4]+H[5])/(H[6]+H[7]+H[8]);
            double distance = sqrt(pow((x2-x_),2)+pow((y2-y_),2));
            score += distance;
            score_+= x1*(x2*H[0]+y2*H[1]+H[2])+y1*(x2*H[3]+y2*H[4]+H[5])+(x2*H[6]+y2*H[7]+H[8]);
        }
        //cout<<"score: "<<score/double(n)<<endl;
        //cout<<"score_: "<<score_/double(n)<<endl;
        score = abs(score/double(n));
        if(score>1000) return -1;
        else if(isnan(score)){
            return -1;
        } 
        cout<<y<<"th image : detected hypotheses "<< index.imgPath2id[y]<<"   match num: "<<n<<", score: "<<score<<endl;
        score = score*100/n;
        
        return score;
    }

    void BagOfWords_WxBS::imageSearchUsingBoW(string I,int topn){
        // Returns the top matches to I from the inverted index 'iindex' computed
        // using bow_buildInvIndex
        // Uses TF-IDF based scoring to rank
        // config has following flags
        // config.geomRerank = m => do geometric reranking for top m results
        // config.topn = n => output top 'n' results (defaults to 10)
        // config.saveMatchesImageDir = 'dir/' => store the matches images in the
        // dir
        // @return : imgPaths is the list of image paths in ranked order
        // @return scores : is the corresponding scores - tf-idf in general, or
        // number of inliers (of top m images) if doing geometric reranking
        // @return all_matches : Only returned if the config.geomRerank = 1. A
        // cell array with {i} element = matches of I with the i^th image
        
        int query_num;
        vector<nodes> d,HRSIFT;
        computeImageRep(I, d,HRSIFT, query_num,0,0);
        
        cout<< "Tf-Idf..."<<endl;
        cout<<"query image descriptorn: "<<query_num<<endl;
        int Ni=0;
        int N=index.numImgs;
        int num = models.vocabSize;
        Mat tfIdf = Mat::zeros(Size(num,N),CV_32FC1);

        

        //count n_id
        for(int y=0;y<N;y++){
            //cout<<index.imgPath2id[y]<<endl;
            for(int x=0;x<200;x++){
                //cout <<tfIdf.at<double>(y,x)<<" ";
                if(0<=d[x].bin[0].index && d[x].bin[0].index<num){
                    if(index.vw2imgsList[y].count(d[x].bin[0].index)){
                        tfIdf.at<float>(y,d[x].bin[0].index)+=index.vw2imgsList[y][d[x].bin[0].index];
                        //tfIdf.at<double>(y,x)+=index.vw2imgsList[y][d[x].index];
                    }
                }
                //cout <<index.vw2imgsList[y][d[x].index]<<" ";
                //cout <<tfIdf.at<double>(y,x)<<" ";
            }
            //cout<<endl<<endl;
        }
        //for(int y=0;y<N;y++){
            //cout<<index.imgPath2id[y]<<endl;
            //for(int x=0;x<num;x++){
                //cout<<d[x].index<<" ";
                //cout <<tfIdf.at<double>(y,x)<<" ";
            //}
            //cout<<endl<<endl;
        //}

        //count n_d and calculate tf
        /* 
        for(int y=0;y<N;y++){
            for(int x=0;x<num;x++){
                //cout <<tfIdf.at<double>(y,x)<<" ";
                //if(tfIdf.at<double>(y,x)>0){
                    tfIdf.at<float>(y,x) /= float(index.vw2imgsList[y][-1]);
                    //cout <<tfIdf.at<double>(y,x)<<" ";
                //}
            }
        }
       //cout <<double(index.vw2imgsList[1][-1])<<" ";
      
        //calculate tf-idf by multiplying idf gotten
        for(int x=0;x<num;x++){
            if(0<=d[x].index && d[x].index<num){
                Ni=0;
                for(int y=0;y<N;y++){
                    if(index.vw2imgsList[y].count(d[x].index)){
                        Ni++;
                    }
                }
                //cout<<"x word: "<<x <<"th,  "<< d[x].index<< " NI: "<<Ni<<endl;
                for(int y=0;y<N;y++){
                    if(Ni>0)
                        if(tfIdf.at<float>(y,x)>0.){
                            tfIdf.at<float>(y,x)*=log10(float(N)/float(Ni));
                        }
                    else
                        tfIdf.at<float>(y,x)*=0;
                }
            }
        }

        for(int y=0;y<N;y++){
            cout<<index.imgPath2id[0]<<endl;
            for(int x=0;x<num;x++){
                cout <<tfIdf.at<double>(0,x)<<" ";
            }
            cout<<endl<<endl;
        }
        */
        //for efficiency to sort, exchange the indeces of img and sum, i.e. int,double -> double, int
        vector<pair<float,int>> score;
        //sum all of elemnts to score
        for(int y=0;y<N;y++){
            float sum=0.;
            for(int x=0;x<num;x++){
                //if(tfIdf.at<double>(y,x)>0.){
                    //cout<<tfIdf.at<double>(y,x)<<" ";
                    sum += tfIdf.at<float>(y,x);
                //}
            }
            //cout<<endl<<endl;
            //cout<<sum<<endl;
            score.push_back(make_pair(sum,y));
        }
        //sort
        sort(score.begin(), score.end(),greater());
        for(int i=0;i<topn;i++)
            cout<<"top "<<i<<" |score: "<<score[i].first<<" | at "<<index.imgPath2id[score[i].second]<<endl;

        //for(int mm=0;mm<query_num;mm++){
                //free(d[mm].bin);
            //}
            //free(d);
    }

    void BagOfWords_WxBS::imageSearchUsingWxBSMatcher(string I,int topn,array<float,3756>& updateDistribution,int n, int m, bool initflag){
        vector<pair<double,int>> score;
        int query_num;
        int N=index.numImgs;
        //ImageRepresentation ImgRep1,ImgRep2;
        Mat result;
        float array[3756]={0.};
        int VERB = Config1.OutputParam.verbose;
        vector<nodes> RSIFTbinlist,HRSIFTbinlist;
        computeImageRep(I, RSIFTbinlist,HRSIFTbinlist,query_num,0,n);
        float sumOfScoreInv=0.;
        
        //LoadRegions(ImgRep1,d1); //TODO
        if(descname=="RSIFT"){
            result = imread(I);
            for(int i=0;i<RSIFTbinlist.size();i++){
                Point2f pt=Point2f(RSIFTbinlist[i].region.reproj_kp.x,RSIFTbinlist[i].region.reproj_kp.y);
                circle(result,pt,2,Scalar(0,0,255),3);
            }

        int numPossbleRankingImgs=0;
        //1. find matching lists
        //query: i-th feature, j-th kdtree index ==  DB: multimap [j-th index] = i-th feature      
        for(int y=0;y<N;y++){
            //TODO have to know how Imgrep is constructed
            if (initflag == false){
                if(!(updateDistribution[y]>0))
                    continue;
            }
            vector<nodes> out1,out2;
            CorrespondenceBank Tentatives;
            map<string, TentativeCorrespListExt> tentatives, verified_coors;

            int corrnum = findCorrespondFeatures(RSIFTbinlist,RSIFTregionVector[y],out1,out2,index.RSIFTmatchlist[y]);
            //cout<<"corrnum: "<<corrnum<<endl;
            
            if(corrnum > m){
               
                //if(out1.size() != d1.size()){
                //    LoadRegions(ImgRep1,out1);
                //}
                //LoadRegions(ImgRep2,out2);


                //TODO convert to TentativeCorrespListExt
                TentativeCorrespListExt current_tents;
                //AffineRegionVector tempRegs1=imgrep1.GetAffineRegionVector("1","1");
                //AffineRegionVector tempRegs2=imgrep2.GetAffineRegionVector("1","1");
                //cout<<"out2.size(): "<<out2.size()<<endl;
                
                for(int i=0;i<out2.size();i++){
                    TentativeCorrespExt tmp_corr;
                    //if(out1.size() != d1.size()){
                    //    tmp_corr.first = d1[i];
                    //}else
                     tmp_corr.first = out1[i].region;
                    tmp_corr.second = out2[i].region;

                    //cout << out1[i].region.reproj_kp.x<<", "<<out1[i].region.reproj_kp.y<<endl;
                    //cout << out2[i].region.reproj_kp.x<<", "<<out2[i].region.reproj_kp.y<<endl;
                    current_tents.TCList.push_back(tmp_corr);
                }
                //Tentatives.AddCorrespondences(current_tents,"1","1");
                
                //tentatives["All"] = Tentatives.GetCorresponcesVector("1","1");
                tentatives["All"]=current_tents;
                
            //2. matching using WxBS Matcher : geometric verification
                DuplicateFiltering(tentatives["All"], Config1.FilterParam.duplicateDist,Config1.FilterParam.mode);
      
                log1.Tentatives1st = tentatives["All"].TCList.size();
                //ransac(lo-ransac like degensac) with LAF check
                //if (VERB) std::cerr << "LO-RANSAC(epipolar) verification is used..." << endl;
                //cout<<"log1.Tentatives1st: "<<log1.Tentatives1st<<endl;

                if(log1.Tentatives1st>m){
                    log1.TrueMatch1st =  LORANSACFiltering(tentatives["All"],
                                                        verified_coors["All"],
                                                        verified_coors["All"].H,
                                                        Config1.RANSACParam);
                    log1.InlierRatio1st = (double) log1.TrueMatch1st / (double) log1.Tentatives1st;

                //3. get score using L2 norm
                    // all of featurs convert using verified_coors["All"].H
                    //and scoring by distance
                    if(verified_coors["All"].H[0] != -1){
                        double score_ =0;
                        
                        //TODO need to remove duplicate points
                        score_= calScore(tentatives["All"],verified_coors["All"].H,y);
                        if(score_!=-1){
                            score.push_back(make_pair(score_,y));
                            numPossbleRankingImgs++;
                            array[y] =1/score_;
                            sumOfScoreInv+= 1/score_;
                        }
                    
                    }
                }
            }

        }
        if (numPossbleRankingImgs < topn) topn = numPossbleRankingImgs;
        
        //sort
        sort(score.begin(), score.end());
        cout<<"query: "<<I<<endl;
        for(int i=0;i<topn;i++)
            cout<<"top "<<i<<" |score: "<<score[i].first<<" | at "<<index.imgPath2id[score[i].second]<<endl;

        }
        else if(descname=="HRSIFT"){

            result = imread(I);
            for(int i=0;i<HRSIFTbinlist.size();i++){
                Point2f pt=Point2f(HRSIFTbinlist[i].region.reproj_kp.x,HRSIFTbinlist[i].region.reproj_kp.y);
                circle(result,pt,2,Scalar(0,0,255),3);
            }

        int numPossbleRankingImgs=0;
        //1. find matching lists
        //query: i-th feature, j-th kdtree index ==  DB: multimap [j-th index] = i-th feature      
        for(int y=0;y<N;y++){
            if (initflag == false){
                if(!(updateDistribution[y]>0))
                    continue;
            }

            //TODO have to know how Imgrep is constructed
            vector<nodes> out1,out2;
            CorrespondenceBank Tentatives;
            map<string, TentativeCorrespListExt> tentatives, verified_coors;
            
            int corrnum = findCorrespondFeatures(HRSIFTbinlist,RSIFTregionVector[y],out1,out2,index.HRSIFTmatchlist[y]);
            //cout<<"corrnum: "<<corrnum<<endl;
            
            if(corrnum > m){
               
                //if(out1.size() != d1.size()){
                //    LoadRegions(ImgRep1,out1);
                //}
                //LoadRegions(ImgRep2,out2);


                //TODO convert to TentativeCorrespListExt
                TentativeCorrespListExt current_tents;
                //AffineRegionVector tempRegs1=imgrep1.GetAffineRegionVector("1","1");
                //AffineRegionVector tempRegs2=imgrep2.GetAffineRegionVector("1","1");
                //cout<<"out2.size(): "<<out2.size()<<endl;
                
                for(int i=0;i<out2.size();i++){
                    TentativeCorrespExt tmp_corr;
                    //if(out1.size() != d1.size()){
                    //    tmp_corr.first = d1[i];
                    //}else
                     tmp_corr.first = out1[i].region;
                    tmp_corr.second = out2[i].region;

                    //cout << out1[i].region.reproj_kp.x<<", "<<out1[i].region.reproj_kp.y<<endl;
                    //cout << out2[i].region.reproj_kp.x<<", "<<out2[i].region.reproj_kp.y<<endl;
                    current_tents.TCList.push_back(tmp_corr);
                }
                //Tentatives.AddCorrespondences(current_tents,"1","1");
                
                //tentatives["All"] = Tentatives.GetCorresponcesVector("1","1");
                tentatives["All"]=current_tents;
                
            //2. matching using WxBS Matcher : geometric verification
                DuplicateFiltering(tentatives["All"], Config1.FilterParam.duplicateDist,Config1.FilterParam.mode);
      
                log1.Tentatives1st = tentatives["All"].TCList.size();
                //ransac(lo-ransac like degensac) with LAF check
                //if (VERB) std::cerr << "LO-RANSAC(epipolar) verification is used..." << endl;
                //cout<<"log1.Tentatives1st: "<<log1.Tentatives1st<<endl;

                if(log1.Tentatives1st>m){
                    log1.TrueMatch1st =  LORANSACFiltering(tentatives["All"],
                                                        verified_coors["All"],
                                                        verified_coors["All"].H,
                                                        Config1.RANSACParam);
                    log1.InlierRatio1st = (double) log1.TrueMatch1st / (double) log1.Tentatives1st;

                //3. get score using L2 norm
                    // all of featurs convert using verified_coors["All"].H
                    //and scoring by distance
                    if(verified_coors["All"].H[0] != -1){
                        double score_ =0;
                        
                        //TODO need to remove duplicate points
                        score_= calScore(tentatives["All"],verified_coors["All"].H,y);
                        if(score_!=-1){
                            score.push_back(make_pair(score_,y));
                            numPossbleRankingImgs++;
                            array[y] =1/score_;
                            sumOfScoreInv+= 1/score_;
                        }
                    
                    }
                }
            }

        }
        if (numPossbleRankingImgs < topn) topn = numPossbleRankingImgs;
        
        //sort
        sort(score.begin(), score.end());
        cout<<"query: "<<I<<endl;
        for(int i=0;i<topn;i++)
            cout<<"top "<<i<<" |score: "<<score[i].first<<" | at "<<index.imgPath2id[score[i].second]<<endl;

        }
        else if(descname=="ALL"){
            result = imread(I);
            for(int i=0;i<RSIFTbinlist.size();i++){
                Point2f pt=Point2f(RSIFTbinlist[i].region.reproj_kp.x,RSIFTbinlist[i].region.reproj_kp.y);
                circle(result,pt,2,Scalar(0,0,255),3);
            }

        int numPossbleRankingImgs=0;
        //1. find matching lists
        //query: i-th feature, j-th kdtree index ==  DB: multimap [j-th index] = i-th feature      
        for(int y=0;y<N;y++){
            if (initflag == false){
                if(!(updateDistribution[y]>0))
                    continue;
            }

            //TODO have to know how Imgrep is constructed
            vector<nodes> out1,out2;
            CorrespondenceBank Tentatives;
            map<string, TentativeCorrespListExt> tentatives, verified_coors;

            int corrnum = findCorrespondFeatures(RSIFTbinlist,HRSIFTbinlist, RSIFTregionVector[y],
                                            out1,out2,index.RSIFTmatchlist[y],index.HRSIFTmatchlist[y]);
            //cout<<"corrnum: "<<corrnum<<endl;
            
            if(corrnum > m){
               
                //if(out1.size() != d1.size()){
                //    LoadRegions(ImgRep1,out1);
                //}
                //LoadRegions(ImgRep2,out2);


                //TODO convert to TentativeCorrespListExt
                TentativeCorrespListExt current_tents;
                //AffineRegionVector tempRegs1=imgrep1.GetAffineRegionVector("1","1");
                //AffineRegionVector tempRegs2=imgrep2.GetAffineRegionVector("1","1");
                //cout<<"out2.size(): "<<out2.size()<<endl;
                
                for(int i=0;i<out2.size();i++){
                    TentativeCorrespExt tmp_corr;
                    //if(out1.size() != d1.size()){
                    //    tmp_corr.first = d1[i];
                    //}else
                     tmp_corr.first = out1[i].region;
                    tmp_corr.second = out2[i].region;

                    //cout << out1[i].region.reproj_kp.x<<", "<<out1[i].region.reproj_kp.y<<endl;
                    //cout << out2[i].region.reproj_kp.x<<", "<<out2[i].region.reproj_kp.y<<endl;
                    current_tents.TCList.push_back(tmp_corr);
                }
                //Tentatives.AddCorrespondences(current_tents,"1","1");
                
                //tentatives["All"] = Tentatives.GetCorresponcesVector("1","1");
                tentatives["All"]=current_tents;
                
            //2. matching using WxBS Matcher : geometric verification
                DuplicateFiltering(tentatives["All"], Config1.FilterParam.duplicateDist,Config1.FilterParam.mode);
      
                log1.Tentatives1st = tentatives["All"].TCList.size();
                //ransac(lo-ransac like degensac) with LAF check
                //if (VERB) std::cerr << "LO-RANSAC(epipolar) verification is used..." << endl;
                //cout<<"log1.Tentatives1st: "<<log1.Tentatives1st<<endl;

                if(log1.Tentatives1st>m){
                    log1.TrueMatch1st =  LORANSACFiltering(tentatives["All"],
                                                        verified_coors["All"],
                                                        verified_coors["All"].H,
                                                        Config1.RANSACParam);
                    log1.InlierRatio1st = (double) log1.TrueMatch1st / (double) log1.Tentatives1st;

                //3. get score using L2 norm
                    // all of featurs convert using verified_coors["All"].H
                    //and scoring by distance
                    if(verified_coors["All"].H[0] != -1){
                        double score_ =0;
                        
                        //TODO need to remove duplicate points
                        score_= calScore(tentatives["All"],verified_coors["All"].H,y);
                        if(score_!=-1){
                            score.push_back(make_pair(score_,y));
                            numPossbleRankingImgs++;
                            array[y] =1/score_;
                            sumOfScoreInv+= 1/score_;
                        }
                    
                    }
                }
            }

        }
        if (numPossbleRankingImgs < topn) topn = numPossbleRankingImgs;
        
        //sort
        sort(score.begin(), score.end());
        cout<<"query: "<<I<<endl;
        for(int i=0;i<topn;i++)
            cout<<"top "<<i<<" |score: "<<score[i].first<<" | at "<<index.imgPath2id[score[i].second]<<endl;

        }
        float kernel[5] = {0.06136,0.24477,0.38774,0.24477,0.06136};
        //array<int,3> kernel = {0.27901,0.44198,0.27901};
        float normalizedSumOfScoreInv=0.;
        float resultDistribution[3756]={0.};
        //imshow("query image",result);
        
        int sizeofKernel = sizeof(kernel)/sizeof(*kernel);
        for(int i=sizeofKernel/2;i<3756-sizeofKernel/2;i++){
            float p=array[i]/sumOfScoreInv;
            for(int j=0;j<sizeofKernel;j++){
                resultDistribution[i-sizeofKernel/2+j] = p*kernel[j] + resultDistribution[i-sizeofKernel/2+j];
                normalizedSumOfScoreInv+=p*kernel[j];
            }
        }
        
        for(int i=0;i<3756;i++){
            if(resultDistribution[i]!=0)
            resultDistribution[i]=resultDistribution[i]/normalizedSumOfScoreInv;
            //sumOfP+=resultDistribution[i];
            //TODO: cout<<resultDistribution[i]<<" ";
            
        }
        cout<<endl;
        float sumOfUpdateP=0.;
        for(int i=0;i<3756;i++){
            if (initflag == false){
                if(!(updateDistribution[i]>0))
                    continue;
            }
            //if(resultDistribution[i]==0)
                //resultDistribution[i]=updateDistribution[i];

            updateDistribution[i] *= resultDistribution[i];
            sumOfUpdateP+=updateDistribution[i];
        }
        cout<<sumOfUpdateP<<endl;
        for(int i=0;i<3756;i++){
            if (initflag == false){
                if(updateDistribution[i]== 0.)
                    continue;
            }
            updateDistribution[i] /= sumOfUpdateP;
            if(updateDistribution[i]!= 0.)
            //TODO: 
            cout<<"image "<<i<<": "<<updateDistribution[i]<<endl;
        }
        cout<<endl;
        Mat draw(2000,1878,CV_32FC3);
        line(draw,Point(0,1000),Point(1878,1000),Scalar(255,255,255),1);
        for(int i=0;i<1878;i++){
            circle(draw,Point(i,updateDistribution[i]*1000),8,Scalar(0,0,255),-1);
        }
        for(int i=1878;i<3756;i++){
            circle(draw,Point(i-1878,updateDistribution[i]*1000+1000),8,Scalar(0,0,255),-1);
        }
        auto roi = Rect(0,0,1878,1000);
        //flip(draw(roi), draw(roi), 1);
        auto roi2 = Rect(0,1000,1878,1000);
        //flip(draw(roi2), draw(roi2), 1);
        cv::resize(draw,draw, Size(1878/2,2000/2));
        imshow("graph",draw);
        waitKey(0);
    }


    void BagOfWords_WxBS::computeImageRep(string I,vector<nodes> &RSIFTbinlist,vector<nodes> &HRSIFTbinlist, int &num, int flag,int n){
        // Computes an image representation (of I) using the quantization 
        // parameters in the model
        // @param I : image after imread
        // @param model : model as generated by bow_computeVocab
        // @return : f (Same as from vl_sift) and bins = quantized descriptor values
        vector<AffineRegion> RSIFTregion;
        vector<AffineRegion> HRSIFTregion;
        //vl_sift_set_peak_thresh(sift,3);
        KDE* kde = new KDE();
        int i=0;
        WxBSdet_desc(I, RSIFTregion,HRSIFTregion);
        clock_t begin = clock();
        Mat img = imread(I);
        Mat img_rand = imread(I);
        Mat img_suggest_rand = imread(I);

        Mat position = Mat::zeros(480,640,CV_64FC1);
        Mat PDF = Mat::zeros(480,640,CV_64FC1);
        //draw point position to Mat
        if (flag == 0){

            kde->set_kernel_type(1);
            kde->set_bandwidth_opt_type(3);

            float N = RSIFTregion.size();
            for(int j=0; j< N;j++){
                int x = RSIFTregion[j].reproj_kp.x;
                int y = RSIFTregion[j].reproj_kp.y;

                kde->add_data((double)x,(double)y);

                //cout<<"position: "<<x<<", "<<y<<endl;
                position.at<double>(y,x) += 1;
                circle(img,Point(x,y),8,Scalar(0,0,255),-1);
            }

            int rand_=0;
            srand ((unsigned int)time(NULL));
            for(int k=0;k<n;k++){
                rand_ = rand() % RSIFTregion.size();
                int x = RSIFTregion[k].reproj_kp.x;
                int y = RSIFTregion[k].reproj_kp.y;
                circle(img_rand,Point(x,y),8,Scalar(0,0,255),-1);
            }

            int q=1;
            double sum=0.;
            double max=0.;
            for(int y=0;y<480;y++){
                for(int x=0;x<640;x++){
                    if(position.at<double>(y,x)>0){
                        //cout<<q<<" x: "<<x<<" , y: "<<y<<" , " <<kde->pdf(x,y)<<endl;
                        PDF.at<double>(y,x) = kde->pdf(x,y);
                        sum+=kde->pdf(x,y);
                        //q++;
                    }
                }
            }
            double sum2=0.;
            for(int y=0;y<480;y++){
                for(int x=0;x<640;x++){
                    if(position.at<double>(y,x)>0){
                        PDF.at<double>(y,x) /= sum;
                        //cout<<q<<" x: "<<x<<" , y: "<<y<<" , " <<PDF.at<double>(y,x)<<endl;
                        if(PDF.at<double>(y,x)>max)
                            max = PDF.at<double>(y,x);
                        sum2+=PDF.at<double>(y,x);
                        //q++;
                    }
                }
            }
            for(int y=0;y<480;y++){
                for(int x=0;x<640;x++){
                    if(position.at<double>(y,x)>0){
                        PDF.at<double>(y,x) /= max;
                        //cout<<" x: "<<x<<" , y: "<<y<<" , " <<PDF.at<double>(y,x)<<endl;
                    }
                }
            }
            //cout<<sum<<endl;
            //cout<<sum2<<endl;

            imwrite("/home/jun/BOW_WxBS/img.jpg",img);
            imwrite("/home/jun/BOW_WxBS/img_rand.jpg",img_rand);
        }


        if(descname=="RSIFT"){
            int Rsizevec = int(RSIFTregion.size());
            num = Rsizevec;
            
            cout<<Rsizevec<<endl;
            int maxth = omp_get_max_threads();
            omp_set_num_threads(8);
            
            if (flag == 1)
            
                #pragma omp parallel for ordered schedule(static)
                for(int i=0;i<Rsizevec;i++){
                    nodes a;
                    //printf("1, Number of threads = %d\n",omp_get_thread_num());
                    a.region = RSIFTregion[i];
                    //cout<<"11"<<endl;
                    //printf("2, Number of threads = %d\n",omp_get_thread_num());
                    int len= RSIFTregion[i].desc.vec.size();
                    //printf("3, Number of threads = %d\n",omp_get_thread_num());
                    float* desc = (float*)vl_malloc(len*sizeof(float));
                    //cout<<"len:"<<len<<endl;
                    //printf("4, Number of threads = %d\n",omp_get_thread_num());
                    for(int j=0;j<len;j++)
                        desc[j] = RSIFTregion[i].desc.vec[j];
                    //cout<<"111"<<endl;
                    //printf("5, Number of threads = %d\n",omp_get_thread_num());

                    #pragma omp critical (my)
                    {
                        vl_kdforest_query(models.RootSIFTkdtree,a.bin,1,desc);
                    }
                    //cout<<i<<" ";
                    //cout<<"1111"<<endl;
                    //binlist[i] = a;
                    //printf("6, Number of threads = %d\n",omp_get_thread_num());
                    //cout<<"11111"<<endl;
                    //binvec.push_back(a);
                    free(desc);
                    //printf("7, Number of threads = %d\n",omp_get_thread_num());
                #pragma omp ordered
                    {
                        RSIFTbinlist.push_back(a);
                    }
                    //cout<<"i: "<<i<<", bin: "<<a.bin<<endl;
                    //printf("%d, Number of threads = %d\n", i, omp_get_thread_num());
                    //cout<<"maxth num: "<<maxth<<endl;
                    
                }
                
            else if (flag == 0){ /*for WxBS matcher*/




            int rand_=0;
            srand ((unsigned int)time(NULL));
            /*random sampling*/

            int ITER=0;
            int idxRand=0;
            int uniRand=0;
//*
            while(ITER != n)
            {

                
                idxRand = rand() % RSIFTregion.size();//TODO: change to rand_r
                uniRand = rand() % n;
                
                nodes a;
                a.region = RSIFTregion[idxRand];
                int x = a.region.reproj_kp.x;
                int y = a.region.reproj_kp.y;

                if(uniRand < PDF.at<double>(y,x)*n){              
                    cout<<"accept"<<endl;
                    //cout << a.region.reproj_kp.x<<", "<<a.region.reproj_kp.y<<endl;
                    //cout<<"11"<<endl;
                    int len= RSIFTregion[idxRand].desc.vec.size();
                    float* desc = (float*)vl_malloc(len*sizeof(float));
                    //cout<<"len:"<<len<<endl;
                    for(int j=0;j<len;j++)
                        desc[j] = RSIFTregion[idxRand].desc.vec[j];
                    //cout<<"111"<<endl;
                    vl_kdforest_query(models.RootSIFTkdtree,a.bin,1,desc);
                    //cout<<i<<" ";
                    //cout<<"1111"<<endl;
                    //binlist[i] = a;
                    RSIFTbinlist.push_back(a);
                    //cout<<"11111"<<endl;
                    //binvec.push_back(a);
                    free(desc);
                    ITER++;
                }
                else{
                    cout<<"reject"<<endl;
                }
            }

            cout<<"ITER!!!: "<< ITER<<endl;
        }
//*/
/*
            for(int i=0;i<n;i++){

                //RSIFTregion.size()
                rand_ = rand() % RSIFTregion.size();//TODO: change to rand_r
                cout<<rand_ <<" ";
                //rand_ = i*4;
                //cout<<"rand: "<<rand_<<endl;
                nodes a;
                a.region = RSIFTregion[rand_];
                //cout << a.region.reproj_kp.x<<", "<<a.region.reproj_kp.y<<endl;
                //cout<<"11"<<endl;
                int len= RSIFTregion[rand_].desc.vec.size();
                float* desc = (float*)vl_malloc(len*sizeof(float));
                //cout<<"len:"<<len<<endl;
                for(int j=0;j<len;j++)
                    desc[j] = RSIFTregion[rand_].desc.vec[j];
                //cout<<"111"<<endl;
                vl_kdforest_query(models.RootSIFTkdtree,a.bin,1,desc);
                //cout<<i<<" ";
                //cout<<"1111"<<endl;
                //binlist[i] = a;
                RSIFTbinlist.push_back(a);
                //cout<<"11111"<<endl;
                //binvec.push_back(a);
                free(desc);
                }
            }
//*/
            //cout<<"2"<<endl;
        }else if(descname=="HRSIFT"){
            int Rsizevec = int(HRSIFTregion.size());
            num = Rsizevec;
            
            cout<<Rsizevec<<endl;
            
            if (flag == 1)
            #pragma omp parallel for ordered schedule(static)
                for(int i=0;i<Rsizevec;i++){
                    nodes a;
                    a.region = HRSIFTregion[i];
                    //cout<<"11"<<endl;
                    int len= HRSIFTregion[i].desc.vec.size();
                    float* desc = (float*)vl_malloc(len*sizeof(float));
                    //cout<<"len:"<<len<<endl;
                    for(int j=0;j<len;j++)
                        desc[j] = HRSIFTregion[i].desc.vec[j];
                    //cout<<"111"<<endl;
                    #pragma omp critical (my)
                    {
                    vl_kdforest_query(models.HalfRootSIFTkdtree,a.bin,1,desc);
                    }
                    
                    free(desc);
                    #pragma omp ordered
                    {
                    HRSIFTbinlist.push_back(a);
                    }
                }
            else if (flag == 0){ /*for WxBS matcher*/
                int rand_=0;
                srand ((unsigned int)time(NULL));
                /*random sampling*/


                int ITER=0;
                int idxRand=0;
                int uniRand=0;
                while(ITER != n)
                {

                    
                    idxRand = rand() % HRSIFTregion.size();//TODO: change to rand_r
                    uniRand = rand() % n;
                    cout<<"idxRand: "<<idxRand<<endl;
                    cout<<"uniRand: "<<uniRand<<endl;
                    
                    nodes a;
                    a.region = HRSIFTregion[idxRand];
                    int x = a.region.reproj_kp.x;
                    int y = a.region.reproj_kp.y;

                    if(uniRand < PDF.at<double>(y,x)*n){              

                        //cout << a.region.reproj_kp.x<<", "<<a.region.reproj_kp.y<<endl;
                        //cout<<"11"<<endl;
                        int len= HRSIFTregion[idxRand].desc.vec.size();
                        float* desc = (float*)vl_malloc(len*sizeof(float));
                        //cout<<"len:"<<len<<endl;
                        for(int j=0;j<len;j++)
                            desc[j] = HRSIFTregion[idxRand].desc.vec[j];
                        //cout<<"111"<<endl;
                        vl_kdforest_query(models.HalfRootSIFTkdtree,a.bin,1,desc);
                        //cout<<i<<" ";
                        //cout<<"1111"<<endl;
                        //binlist[i] = a;
                        HRSIFTbinlist.push_back(a);
                        //cout<<"11111"<<endl;
                        //binvec.push_back(a);
                        free(desc);
                        ITER++;
                    }
                }
/*
            for(int i=0;i<n;i++){

                
                rand_ = rand() % n;
                
                nodes a;
                a.region = HRSIFTregion[rand_];
                
                int len= HRSIFTregion[rand_].desc.vec.size();
                float* desc = (float*)vl_malloc(len*sizeof(float));
                //cout<<"len:"<<len<<endl;
                for(int j=0;j<len;j++)
                    desc[j] = HRSIFTregion[rand_].desc.vec[j];
                //cout<<"111"<<endl;
                vl_kdforest_query(models.HalfRootSIFTkdtree,a.bin,1,desc);
                
                HRSIFTbinlist.push_back(a);
                
                free(desc);
                }
*/

            }
        }else if(descname=="ALL"){
            int Rsizevec = int(HRSIFTregion.size());
            num = Rsizevec;
            
            cout<<Rsizevec<<endl;
            
            if (flag == 1)
            for(int i=0;i<Rsizevec;i++){
                nodes a;
                a.region = HRSIFTregion[i];
                //cout<<"11"<<endl;
                int len= HRSIFTregion[i].desc.vec.size();
                float* desc = (float*)vl_malloc(len*sizeof(float));
                //cout<<"len:"<<len<<endl;
                for(int j=0;j<len;j++)
                    desc[j] = HRSIFTregion[i].desc.vec[j];
                //cout<<"111"<<endl;
                vl_kdforest_query(models.HalfRootSIFTkdtree,a.bin,1,desc);
                
                HRSIFTbinlist.push_back(a);
                
                free(desc);
            }
            else if (flag == 0){ /*for WxBS matcher*/
                int rand_=0;
                srand ((unsigned int)time(NULL));
                /*random sampling*/
                map<int,int> curringlist;
                
                int ITER=0;
                int idxRand=0;
                int uniRand=0;
/*
                while(ITER != n/2)
                {

                    
                    idxRand = rand() % RSIFTregion.size();//TODO: change to rand_r
                    uniRand = rand() % n;
                    
                    nodes a;
                    a.region = RSIFTregion[idxRand];
                    int x = a.region.reproj_kp.x;
                    int y = a.region.reproj_kp.y;

                    if(uniRand < PDF.at<double>(y,x)*n){              

                        //cout << a.region.reproj_kp.x<<", "<<a.region.reproj_kp.y<<endl;
                        //cout<<"11"<<endl;
                        int len= RSIFTregion[idxRand].desc.vec.size();
                        float* desc = (float*)vl_malloc(len*sizeof(float));
                        //cout<<"len:"<<len<<endl;
                        for(int j=0;j<len;j++)
                            desc[j] = RSIFTregion[idxRand].desc.vec[j];
                        //cout<<"111"<<endl;
                        vl_kdforest_query(models.RootSIFTkdtree,a.bin,1,desc);
                        //cout<<i<<" ";
                        //cout<<"1111"<<endl;
                        //binlist[i] = a;
                        RSIFTbinlist.push_back(a);
                        //cout<<"11111"<<endl;
                        //binvec.push_back(a);
                        free(desc);

                        circle(img_suggest_rand,Point(x,y),8,Scalar(0,0,255),-1);

                        ITER++;
                    }
                }
                ITER =0;
                while(ITER != n/2)
                {
                    idxRand = rand() % HRSIFTregion.size();//TODO: change to rand_r
                    uniRand = rand() % n;
                    
                    nodes a;
                    a.region = HRSIFTregion[idxRand];
                    int x = a.region.reproj_kp.x;
                    int y = a.region.reproj_kp.y;

                    if(uniRand < PDF.at<double>(y,x)*n){              

                        //cout << a.region.reproj_kp.x<<", "<<a.region.reproj_kp.y<<endl;
                        //cout<<"11"<<endl;
                        int len= HRSIFTregion[idxRand].desc.vec.size();
                        float* desc = (float*)vl_malloc(len*sizeof(float));
                        //cout<<"len:"<<len<<endl;
                        for(int j=0;j<len;j++)
                            desc[j] = HRSIFTregion[idxRand].desc.vec[j];
                        //cout<<"111"<<endl;
                        vl_kdforest_query(models.HalfRootSIFTkdtree,a.bin,1,desc);
                        //cout<<i<<" ";
                        //cout<<"1111"<<endl;
                        //binlist[i] = a;
                        HRSIFTbinlist.push_back(a);
                        //cout<<"11111"<<endl;
                        //binvec.push_back(a);
                        free(desc);
                        circle(img_suggest_rand,Point(x,y),8,Scalar(0,0,255),-1);
                        ITER++;
                    }
                }
                imwrite("/home/jun/BOW_WxBS/img_suggest_rand.jpg",img_suggest_rand);
//*/
//*
                for(int i=0;i<n/2;i++){
                    rand_ = rand() % RSIFTregion.size();
                    
                    //rand_ = i*4;
                    nodes a;
                    a.region = RSIFTregion[rand_];
                    
                    int len= RSIFTregion[rand_].desc.vec.size();
                    float* desc = (float*)vl_malloc(len*sizeof(float));
                    //cout<<"len:"<<len<<endl;
                    for(int j=0;j<len;j++)
                        desc[j] = RSIFTregion[rand_].desc.vec[j];
                    //cout<<"111"<<endl;
                    vl_kdforest_query(models.RootSIFTkdtree,a.bin,1,desc);
                    
                    RSIFTbinlist.push_back(a);
                    
                    free(desc);
                }
                
                for(int i=0;i<n/2;i++){

                    
                    rand_ = rand() % HRSIFTregion.size();
                    
                    nodes a;
                    a.region = HRSIFTregion[rand_];
                    
                    int len= HRSIFTregion[rand_].desc.vec.size();
                    float* desc = (float*)vl_malloc(len*sizeof(float));
                    //cout<<"len:"<<len<<endl;
                    for(int j=0;j<len;j++)
                        desc[j] = HRSIFTregion[rand_].desc.vec[j];
                    //cout<<"111"<<endl;
                    vl_kdforest_query(models.HalfRootSIFTkdtree,a.bin,1,desc);
                    
                    HRSIFTbinlist.push_back(a);
                    
                    free(desc);
                }

//*/
            }
        }

        clock_t end = clock();  
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        cout<<"time: "<<elapsed_secs<<endl;
        delete kde;
    }
    void BagOfWords_WxBS::buildInvIndex(string imgsDir,int numImg,int flag){
        vector<multimap<int,int>> RSIFTmatchlist;
        vector<multimap<int,int>> HRSIFTmatchlist;
        index.RSIFTmatchlist.clear();
        index.HRSIFTmatchlist.clear();
        RSIFTregionVector.clear();
        HRSIFTregionVector.clear();
        index.imgPath2id.clear();
        //vector<map<int,int>> vw2imgsList;
        //vw2imgsList.reserve(models.vocabSize);
        //vw2imgsList.reserve(index.numImgs);
        //for i = 1 : model.vocabSize
        //    vw2imgsList{i} = containers.Map('KeyType', 'int64', 'ValueType', 'int64');
        //end
        index.dirname = imgsDir;
        string frpaths;
        fstream f;
        f.open(imgsDir);
        index.numImgs = numImg;
        
        if(descname=="RSIFT"||descname=="ALL"){

            for(int i=0;i<index.numImgs;i++){
                int num=0;
                //map<int,int> init;
                multimap<int,int> init_;
                //vw2imgsList.push_back(init);
                RSIFTmatchlist.push_back(init_);

                if(flag==1){
                    f>>frpaths;
                    string path = "/home/jun/ImageDataSet/VPRiCE-dataset/memory/"+frpaths;
                    f>>frpaths>>frpaths;//temp
                    //// Get imgs list
                    // Add these paths to a hash map as well
                    index.imgPath2id.insert(pair<int,string>(i,path));
                }else if(flag==2){
                    f>>frpaths>>frpaths;
                    string path = "/home/jun/ImageDataSet/VPRiCE-dataset/memory/"+frpaths;
                    f>>frpaths;//temp
                    //// Get imgs list
                    // Add these paths to a hash map as well
                    index.imgPath2id.insert(pair<int,string>(i,path));
                }else if(flag==3){
                    f>>frpaths>>frpaths>>frpaths;
                    string path = "/home/jun/ImageDataSet/VPRiCE-dataset/memory/"+frpaths;
                    //f;//temp
                    //// Get imgs list
                    // Add these paths to a hash map as well
                    index.imgPath2id.insert(pair<int,string>(i,path));
                }

            //for i = 1 : index.numImgs
                //try{
                    //Mat I = imread(index.imgPath2id[i],0);
                    //resize(I, I, Size(640,480));
                    string I = index.imgPath2id[i];
                    vector<nodes> d,HRSIFTd;
                    computeImageRep(I, d,HRSIFTd, num,1,0);


                    RSIFTregionVector.push_back(d);
                        //for(int x=10000;x<15000;x++){
                        //    cout<<d[x].bin[0].index<<" ";
                        //}
                        //cout<<endl;
                    
                    //[~, d] = bow_computeImageRep(I, model, 'PeakThresh', 3);//TODO conversion
                    //return descriptor vectors
    ///*
                    //index.totalDescriptors[i] = d.size();
                    for(int j=0; j<num;j++){
                    //for(int j=0; j<models.vocabSize;j++){
                        
                        RSIFTmatchlist[i].insert(pair<int, int>(d[j].bin[0].index, j));
                        
                    /*    
                        if (vw2imgsList[i].count(d[j].bin[0].index)){
                            vw2imgsList[i][d[j].bin[0].index] += 1;
                            
                        }else{
                            vw2imgsList[i][d[j].bin[0].index] = 1;
                            
                        }
                    
                    }
                    vw2imgsList[i].insert(pair<int,int>(-1,num));*/
                    }
                    

                printf("nFeat = %d. Indexed (%d / %d)\n", num, i+1, index.numImgs);

                for(int mm=0;mm<num;mm++){
                    free(d[mm].bin);
                }
                //free(d);

            }
            //index.vw2imgsList = vw2imgsList;
            index.RSIFTmatchlist = RSIFTmatchlist;
            
        }else if(descname=="HRSIFT"||descname=="ALL"){

            for(int i=0;i<index.numImgs;i++){
                int num=0;
                multimap<int,int> init_;
                HRSIFTmatchlist.push_back(init_);
                if(flag==1){
                    f>>frpaths;
                    string path = "/home/jun/ImageDataSet/VPRiCE-dataset/memory/"+frpaths;
                    f>>frpaths>>frpaths;//temp
                    index.imgPath2id.insert(pair<int,string>(i,path));
                }else if(flag==2){
                    f>>frpaths>>frpaths;
                    string path = "/home/jun/ImageDataSet/VPRiCE-dataset/memory/"+frpaths;
                    f>>frpaths;//temp
                    index.imgPath2id.insert(pair<int,string>(i,path));
                    cout<<path<<endl;
                }else if(flag==3){
                    f>>frpaths>>frpaths>>frpaths;
                    string path = "/home/jun/ImageDataSet/VPRiCE-dataset/memory/"+frpaths;
                    //f;//temp
                    index.imgPath2id.insert(pair<int,string>(i,path));
                    cout<<path<<endl;
                }

                    string I = index.imgPath2id[i];
                    vector<nodes> d,HRSIFTd;
                    computeImageRep(I, d,HRSIFTd, num,1,0);

                    HRSIFTregionVector.push_back(HRSIFTd);
             
                    for(int j=0; j<num;j++){
         
                        HRSIFTmatchlist[i].insert(pair<int, int>(HRSIFTd[j].bin[0].index, j));
                        
                    }
                    

                printf("nFeat = %d. Indexed (%d / %d)\n", num, i+1, index.numImgs);

                for(int mm=0;mm<num;mm++){
                    free(HRSIFTd[mm].bin);
                }
     
            }
            index.HRSIFTmatchlist = HRSIFTmatchlist;
            
        }



        f.close();
    }


    int BagOfWords_WxBS::computeVocab(string imgsDir, int numImg){

        index.dirname = imgsDir;
        int found_sifts = 0;
        cout<<"Reading SIFTs "<<endl;

        string frpaths;
        fstream f;
        f.open(imgsDir);
        int totalcnt=0;
        
        vector<float> RootSIFTdesc;
        vector<float> HalfRootSIFTdesc;
        int m=0;
        for (m = 0; m<numImg;m++){
            vector<AffineRegion> RootSIFTregion;
            vector<AffineRegion> HalfRootSIFTregion;

            f>>frpaths;
            string path = "/home/jun/ImageDataSet/VPRiCE-dataset/memory/"+frpaths;
            //f>>frpaths>>frpaths;//temp
            cout<< path<<" "<<m<<"/"<<numImg<<endl;
            //cout<<path<<endl;
            //index.imgPaths.push_back(path);
            //// Get imgs list
            // Add these paths to a hash map as well
            index.imgPath2id.insert(pair<int,string>(m,path));

            int totalnKey=0;
            int i=0;
            int descNum=0;
            WxBSdet_desc(path, RootSIFTregion,HalfRootSIFTregion);


            for(int i=0;i<RootSIFTregion.size();i++)
                for (int ddd = 0; ddd <RootSIFTregion[i].desc.vec.size(); ++ddd){
                    RootSIFTdesc.push_back(RootSIFTregion[i].desc.vec[ddd]);
                    //kpfile << ar.desc.vec[ddd] << " ";
                    }
            for(int i=0;i<HalfRootSIFTregion.size();i++)
                for (int ddd = 0; ddd <HalfRootSIFTregion[i].desc.vec.size(); ++ddd){
                    HalfRootSIFTdesc.push_back(HalfRootSIFTregion[i].desc.vec[ddd]);
                    //kpfile << ar.desc.vec[ddd] << " ";
                    }

            cout<<"sizeof R SIFT desc: "<<RootSIFTregion.size()<<endl;
            cout<<"sizeof Half R SIFT desc: "<<HalfRootSIFTregion.size()<<endl;
       
        }

        index.numImgs = numImg;
  
        cout<<"Done"<<endl;

        int Rsizevec = int(RootSIFTdesc.size())/128;
        int HRsizevec = int(HalfRootSIFTdesc.size())/64;
        cout<<"Rsize vec: "<<Rsizevec<<endl;
        cout<<"HRsize vec: "<<HRsizevec<<endl;
        float* Rdesc;
        Rdesc = (float*)vl_malloc(Rsizevec*128*sizeof(float));
        float* HRdesc;
        HRdesc = (float*)vl_malloc(Rsizevec*64*sizeof(float));

        for(int p=0;p<Rsizevec*128;p++){
           
            Rdesc[p] = RootSIFTdesc[p];
          
        }
        for(int p=0;p<HRsizevec*64;p++){
           
            HRdesc[p] = HalfRootSIFTdesc[p];
        }

        f.close();

        if(descname == "RSIFT" || descname =="ALL"){
        cout<< "Found RootSIFT " <<Rsizevec<<" descriptors. "<<endl;
        //cout<< "Found HlafRootSIFT " <<HRsizevec<<" descriptors. "<<endl;
        // K Means cluster the SIFTs, and create a model
        models.vocabSize = params.numWords;
        //vl_file_meta_close (&dsc) ;
        cout<<"clustering RootSIFTdesc..."<<endl;
        vl_size numData = Rsizevec;
        vl_size dimension = 128;
        vl_size numCenters = params.numWords;
        vl_size maxiter = 100;
        vl_size maxComp = 100;
        vl_size maxrep = 1;
        vl_size ntrees = 3;
        
        vl_kmeans_set_verbosity	(kmeans,1);
        // Use Lloyd algorithm
        vl_kmeans_set_algorithm(kmeans, VlKMeansANN) ;
        vl_kmeans_set_max_num_comparisons (kmeans, maxComp) ;
        vl_kmeans_set_num_repetitions (kmeans, maxrep) ;
        vl_kmeans_set_num_trees (kmeans, ntrees);
        vl_kmeans_set_max_num_iterations (kmeans, maxiter) ;
        // Initialize the cluster centers by randomly sampling the data
        vl_kmeans_init_centers_plus_plus (kmeans, Rdesc, dimension, numData, numCenters) ;
 
        vl_kmeans_cluster(kmeans,Rdesc,dimension,numData,numCenters);

        
        vl_kdforest_build(models.RootSIFTkdtree,numCenters,kmeans->centers);
        }else if(descname == "HRSIFT" || descname =="ALL"){
        cout<< "Found HRootSIFT " <<HRsizevec<<" descriptors. "<<endl;
        //cout<< "Found HlafRootSIFT " <<HRsizevec<<" descriptors. "<<endl;
        // K Means cluster the SIFTs, and create a model
        models.vocabSize = params.numWords;
        //vl_file_meta_close (&dsc) ;
        cout<<"clustering HRootSIFTdesc..."<<endl;
        vl_size numData = HRsizevec;
        vl_size dimension = 64;
        vl_size numCenters = params.numWords;
        vl_size maxiter = 100;
        vl_size maxComp = 100;
        vl_size maxrep = 1;
        vl_size ntrees = 3;
        
        vl_kmeans_set_verbosity	(kmeans,1);
        // Use Lloyd algorithm
        vl_kmeans_set_algorithm(kmeans, VlKMeansANN) ;
        vl_kmeans_set_max_num_comparisons (kmeans, maxComp) ;
        vl_kmeans_set_num_repetitions (kmeans, maxrep) ;
        vl_kmeans_set_num_trees (kmeans, ntrees);
        vl_kmeans_set_max_num_iterations (kmeans, maxiter) ;
        // Initialize the cluster centers by randomly sampling the data
        vl_kmeans_init_centers_plus_plus (kmeans, HRdesc, dimension, numData, numCenters) ;
 
        vl_kmeans_cluster(kmeans,HRdesc,dimension,numData,numCenters);

        
        vl_kdforest_build(models.HalfRootSIFTkdtree,numCenters,kmeans->centers);
        }

        vl_free(Rdesc);
        vl_free(HRdesc);

        RootSIFTdesc.clear();
        HalfRootSIFTdesc.clear();
        //vl_free(HRdesc);
        return 0;

            
    }
    void BagOfWords_WxBS::extractDescriptor(string imgPath, int start,string R,string HR,int num,string location){
        ofstream Rd(R);
        ofstream HRd(HR);

        int found_sifts = 0;
        cout<<"Reading SIFTs "<<endl;

        string frpaths,temp;
        fstream f;
        f.open(imgPath);
        int totalcnt=0;
        
        int m=0;
        for(int q=0;q<start;q++)f>>temp;
        for (m = start; m<start+num/2;m++){
            vector<AffineRegion> RootSIFTregion;
            vector<AffineRegion> HalfRootSIFTregion;
            f>>frpaths;
            string path;
            if(location=="memory")
                path = "/home/jun/ImageDataSet/VPRiCE-dataset/memory/"+frpaths;
            else if(location=="live")
                path = "/home/jun/ImageDataSet/VPRiCE-dataset/live/"+frpaths;
            
            //f>>frpaths>>frpaths;//temp
            cout<< path<<" "<<m<<"/"<<num<<endl;
           
            int totalnKey=0;
            int i=0;
            int descNum=0;
            WxBSdet_desc(path, RootSIFTregion,HalfRootSIFTregion);
            Rd<<RootSIFTregion.size()<<endl;
            HRd<<HalfRootSIFTregion.size()<<endl;

            for(int i=0;i<RootSIFTregion.size();i++)
                for (int ddd = 0; ddd <RootSIFTregion[i].desc.vec.size(); ++ddd){
                    Rd<< RootSIFTregion[i].desc.vec[ddd]<<" ";
                    
                    }
            for(int i=0;i<HalfRootSIFTregion.size();i++)
                for (int ddd = 0; ddd <HalfRootSIFTregion[i].desc.vec.size(); ++ddd){
                    HRd<< HalfRootSIFTregion[i].desc.vec[ddd]<<" ";
                    
                    }
            cout<<"sizeof R SIFT desc: "<<RootSIFTregion.size()<<endl;
            cout<<"sizeof Half R SIFT desc: "<<HalfRootSIFTregion.size()<<endl;
        }
        Rd.close();
        HRd.close();
    }
    void BagOfWords_WxBS::computeVocabWithoutExtractor(string desc, string R[]){
        if(desc=="RSIFT"){
            vector<float> RootSIFTdesc;
            fstream Rf;

            for(int i=0;i<2;i++){
                Rf.open(R[i]);
                for(int q=0;q<3756/2;q++){
                    int len=0;
                    Rf>>len;
                    for(int j=0;j<len;j++){
                        float val=0;
                        for(int k=0;k<128;k++){
                            Rf>>val;
                            RootSIFTdesc.push_back(val);
                        }
                    }
                }
            }
            for(int i=2;i<4;i++){
                Rf.open(R[i]);
                for(int q=0;q<4022/2;q++){
                    int len=0;
                    Rf>>len;
                    for(int j=0;j<len;j++){
                        float val=0;
                        for(int k=0;k<128;k++){
                            Rf>>val;
                            RootSIFTdesc.push_back(val);
                        }
                    }
                }
            }
            Rf.close();
            int Rsizevec = int(RootSIFTdesc.size())/128;
            float* Rdesc;
            Rdesc = (float*)vl_malloc(Rsizevec*128*sizeof(float));
            
            for(int p=0;p<Rsizevec*128;p++){
            
                Rdesc[p] = RootSIFTdesc[p];
            
            }
            
            models.vocabSize = params.numWords;
            //vl_file_meta_close (&dsc) ;
            cout<<"clustering RootSIFTdesc..."<<endl;
            vl_size numData = Rsizevec;
            cout<<"Rsev:"<<Rsizevec<<endl;
            vl_size dimension = 128;
            vl_size numCenters = params.numWords;
            vl_size maxiter = 1000;
            vl_size maxComp = 100;
            vl_size maxrep = 1;
            vl_size ntrees = 3;
            
            vl_kmeans_set_verbosity	(kmeans,1);
            // Use Lloyd algorithm
            vl_kmeans_set_algorithm(kmeans, VlKMeansANN) ;
            vl_kmeans_set_max_num_comparisons (kmeans, maxComp) ;
            vl_kmeans_set_num_repetitions (kmeans, maxrep) ;
            vl_kmeans_set_num_trees (kmeans, ntrees);
            vl_kmeans_set_max_num_iterations (kmeans, maxiter) ;
            vl_kmeans_set_min_energy_variation(kmeans,0.000001);
            // Initialize the cluster centers by randomly sampling the data
            vl_kmeans_init_centers_plus_plus (kmeans, Rdesc, dimension, numData, numCenters) ;
            //vl_kmeans_init_centers_with_rand_data(kmeans, Rdesc, dimension, numData, numCenters) ;
            vl_kmeans_cluster(kmeans,Rdesc,dimension,numData,numCenters);
            vl_kdforest_build(models.RootSIFTkdtree,numCenters,kmeans->centers);
            free(Rdesc);
        }else if(desc=="HRSIFT"){
            vector<float> RootSIFTdesc;
            fstream Rf;

            for(int i=0;i<2;i++){
                Rf.open(R[i]);
                for(int q=0;q<3756/2;q++){
                    int len=0;
                    Rf>>len;
                    for(int j=0;j<len;j++){
                        float val=0;
                        for(int k=0;k<64;k++){
                            Rf>>val;
                            RootSIFTdesc.push_back(val);
                        }
                    }
                }
            }
            for(int i=2;i<4;i++){
                Rf.open(R[i]);
                for(int q=0;q<4022/2;q++){
                    int len=0;
                    Rf>>len;
                    for(int j=0;j<len;j++){
                        float val=0;
                        for(int k=0;k<64;k++){
                            Rf>>val;
                            RootSIFTdesc.push_back(val);
                        }
                    }
                }
            }
            Rf.close();
            int Rsizevec = int(RootSIFTdesc.size())/64;
            float* Rdesc;
            Rdesc = (float*)vl_malloc(Rsizevec*64*sizeof(float));
            
            for(int p=0;p<Rsizevec*64;p++){
            
                Rdesc[p] = RootSIFTdesc[p];
            
            }
            
            models.vocabSize = params.numWords;
            //vl_file_meta_close (&dsc) ;
            cout<<"clustering HRootSIFTdesc..."<<endl;
            vl_size numData = Rsizevec;
            vl_size dimension = 64;
            vl_size numCenters = params.numWords;
            vl_size maxiter = 1000;
            vl_size maxComp = 100;
            vl_size maxrep = 1;
            vl_size ntrees = 3;
            
            vl_kmeans_set_verbosity	(kmeans,1);
            // Use Lloyd algorithm
            vl_kmeans_set_algorithm(kmeans, VlKMeansANN) ;
            vl_kmeans_set_max_num_comparisons (kmeans, maxComp) ;
            vl_kmeans_set_num_repetitions (kmeans, maxrep) ;
            vl_kmeans_set_num_trees (kmeans, ntrees);
            vl_kmeans_set_max_num_iterations (kmeans, maxiter) ;
            vl_kmeans_set_min_energy_variation(kmeans,0.000001);
            // Initialize the cluster centers by randomly sampling the data
            vl_kmeans_init_centers_plus_plus (kmeans, Rdesc, dimension, numData, numCenters) ;
            //vl_kmeans_init_centers_with_rand_data(kmeans, Rdesc, dimension, numData, numCenters) ;
            vl_kmeans_cluster(kmeans,Rdesc,dimension,numData,numCenters);
            vl_kdforest_build(models.HalfRootSIFTkdtree,numCenters,kmeans->centers);
            free(Rdesc);
        }
    }
    void BagOfWords_WxBS::saveVocab(string name){
        if(descname=="RSIFT"|| descname=="ALL"){
            cout<<"RSIFT saving vocabulary..."<<endl;
            ofstream f(name);

            float *vocab = (float*)malloc(models.vocabSize*128*sizeof(float));
            vocab = (float*)kmeans->centers;
            for(int i=0;i<models.vocabSize*128;i++){
                f<<vocab[i]<<" ";
            }
            f.close();
        }else if(descname=="HRSIFT"|| descname=="ALL"){
            cout<<"HRSIFT saving vocabulary..."<<endl;
        ofstream f(name);

        float *vocab = (float*)malloc(models.vocabSize*64*sizeof(float));
        vocab = (float*)kmeans->centers;
        for(int i=0;i<models.vocabSize*64;i++){
            f<<vocab[i]<<" ";
        }
        f.close();
        }
        //free(vocab);
    }
    void BagOfWords_WxBS::loadVocab(string name){
        if(descname=="RSIFT"||descname=="ALL"){
        cout<<" RSIFT loading vocabulary..."<<endl;
        fstream f;
        f.open(name);
         float *vocab = (float*)malloc(models.vocabSize*128*sizeof(float));
        for(int i=0;i<models.vocabSize*128;i++){
            f>>vocab[i];
        }
        models.vocabSize = params.numWords;
        vl_kdforest_build(models.RootSIFTkdtree,models.vocabSize,vocab);
        cout<<"done"<<endl;
        //free(vocab);
        f.close();

        }else if(descname=="HRSIFT"||descname=="ALL"){
        cout<<" HRSIFT loading vocabulary..."<<endl;
        fstream f;
        f.open(name);
         float *vocab = (float*)malloc(models.vocabSize*64*sizeof(float));
        for(int i=0;i<models.vocabSize*64;i++){
            f>>vocab[i];
        }
        models.vocabSize = params.numWords;
        vl_kdforest_build(models.HalfRootSIFTkdtree,models.vocabSize,vocab);
        cout<<"done"<<endl;
        //free(vocab);
        f.close();
        }
    }

    void BagOfWords_WxBS::saveIndex(string name){
        if(descname=="RSIFT"||descname=="ALL"){
            cout<<"RSIFT saving InvertedIndex..."<<endl;
            ofstream f(name);

            int i=0;
            //for(vec = index.vw2imgsList.begin(); vec != index.vw2imgsList.end(); i++,++vec){
            
            f<<index.RSIFTmatchlist.size()<<" ";
            //matchlist save
            for(i=0;i<index.RSIFTmatchlist.size();i++){
                f<<index.RSIFTmatchlist[i].size()<<" ";
                multimap<int, int>::iterator iter;  
                for(iter = index.RSIFTmatchlist[i].begin();iter!=index.RSIFTmatchlist[i].end();++iter){
                    f<<iter->first<<" "<<iter->second<<" ";
                }
            }

            f.close();
            
        }else if(descname=="HRSIFT"||descname=="ALL"){
            cout<<"HRSIFT saving InvertedIndex..."<<endl;
            ofstream f(name);

            int i=0;
            
            f<<index.HRSIFTmatchlist.size()<<" ";
            //matchlist save
            for(i=0;i<index.HRSIFTmatchlist.size();i++){
                f<<index.HRSIFTmatchlist[i].size()<<" ";
                multimap<int, int>::iterator iter;  
                for(iter = index.HRSIFTmatchlist[i].begin();iter!=index.HRSIFTmatchlist[i].end();++iter){
                    f<<iter->first<<" "<<iter->second<<" ";
                }
            }

            f.close();
        }
    }
    void BagOfWords_WxBS::loadIndex(string name){
        if(descname=="RSIFT"){
            cout<<"RSIFT loading InvertedIndex..."<<endl;
            fstream f;
            f.open(name);

            int len;
            f>>len;
            for(int i=0;i<len;i++){
                int len_;
                int first,second;
                f>>len_;
                
                multimap<int,int>a;
                for(int j=0;j<len_;j++){
                    f>>first>>second;
                    a.insert(pair<int, int>(first, second));
                } 
                index.RSIFTmatchlist.push_back(a);
            }

            f.close();
        }
        else if(descname=="HRSIFT"){
            cout<<"HRSIFT loading InvertedIndex..."<<endl;
            fstream f;
            f.open(name);
            
            int len;
            f>>len;
            for(int i=0;i<len;i++){
                int len_;
                int first,second;
                f>>len_;
                
                multimap<int,int>a;
                for(int j=0;j<len_;j++){
                    f>>first>>second;
                    a.insert(pair<int, int>(first, second));
                } 
                index.HRSIFTmatchlist.push_back(a);
            }

            f.close();
        }
        cout<<"done"<<endl;
    }
    void BagOfWords_WxBS::loadIndex(string name[]){
        if(descname=="RSIFT"){
            cout<<"RSIFT loading InvertedIndex..."<<endl;
            fstream f1;
            f1.open(name[0]);
            fstream f2;
            f2.open(name[1]);
            fstream f3;
            f3.open(name[2]);

            int len;
            f1>>len;
            f2>>len;
            f3>>len;
            for(int i=0;i<len;i++){
                int len_;
                int first,second;
                f1>>len_;
                
                multimap<int,int>a;
                map<int,int> b;
                for(int j=0;j<len_;j++){
                    f1>>first>>second;
                    a.insert(pair<int, int>(first, second));

                    if (b.count(first)){
                            b[first] += 1;     
                    }else{
                        b[first] = 1;  
                    }
                }
                b[-1]=len_;
                index.RSIFTmatchlist.push_back(a);
                index.vw2imgsList.push_back(b);

                f2>>len_;
                a.clear();
                b.clear();
                for(int j=0;j<len_;j++){
                    f2>>first>>second;
                    a.insert(pair<int, int>(first, second));

                    if (b.count(first)){
                            b[first] += 1;     
                    }else{
                        b[first] = 1;  
                    }
                }
                b[-1]=len_;
                index.RSIFTmatchlist.push_back(a);
                index.vw2imgsList.push_back(b);

                f3>>len_;
                a.clear();
                b.clear();
                for(int j=0;j<len_;j++){
                    f3>>first>>second;
                    a.insert(pair<int, int>(first, second));
                    
                    if (b.count(first)){
                            b[first] += 1;     
                    }else{
                        b[first] = 1;  
                    }
                }
                b[-1]=len_;
                index.RSIFTmatchlist.push_back(a);
                index.vw2imgsList.push_back(b);

            }

            f1.close();
            f2.close();
            f3.close();
        }
        else if(descname=="HRSIFT"){
            cout<<"HRSIFT loading InvertedIndex..."<<endl;
            fstream f1;
            f1.open(name[0]);
            fstream f2;
            f2.open(name[1]);
            fstream f3;
            f3.open(name[2]);

            int len;
            f1>>len;
            f2>>len;
            f3>>len;
            for(int i=0;i<len;i++){
                int len_;
                int first,second;
                f1>>len_;
                
                multimap<int,int>a;
                for(int j=0;j<len_;j++){
                    f1>>first>>second;
                    a.insert(pair<int, int>(first, second));
                } 
                index.HRSIFTmatchlist.push_back(a);

                f2>>len_;
                a.clear();
                for(int j=0;j<len_;j++){
                    f2>>first>>second;
                    a.insert(pair<int, int>(first, second));
                } 
                index.HRSIFTmatchlist.push_back(a);

                f3>>len_;
                a.clear();
                for(int j=0;j<len_;j++){
                    f3>>first>>second;
                    a.insert(pair<int, int>(first, second));
                } 
                index.HRSIFTmatchlist.push_back(a);

            }

            f1.close();
            f2.close();
            f3.close();
        }
        cout<<"done"<<endl;
    }
    void BagOfWords_WxBS::grouping(string name,float threshold){
        cout<<"Grouping..."<<endl;
        ofstream f(name);
        float group[100000]={0.};
        
        
        f<<0<<" ";
        int descnum=index.vw2imgsList[0][-1];
        int groupNum=1;

        map<int, int>::iterator iter;
        for(iter = index.vw2imgsList[0].begin(); iter != index.vw2imgsList[0].end(); iter++)
            group[iter->first]=iter->second;

        for(int i=1;i<index.vw2imgsList.size();i++){
            float query[100000]={0.}; //new image
            float tf[100000]={0.};

            for(iter = index.vw2imgsList[i].begin(); iter != index.vw2imgsList[i].end(); iter++)
                query[iter->first]=iter->second;
            //get n_id
            for(int x=0;x<100000;x++){
                if(query[x]!=0. && group[x]!=0.){
                    for(int j=0;j<query[x];j++)
                        tf[x]+=group[x];
                }
                //count n_d and calculate tf
                tf[x] /= descnum;
            }

            float sum=0.;
            for(int x=0;x<100000;x++){
                    sum += tf[x];
            }

            if(sum>=threshold){
                //if tf value is larger than threshold, new query merge to group(by mean, update descnum)
                for(int x=0;x<100000;x++){
                    group[x] *= groupNum;
                    group[x] += query[x];
                    group[x] /= groupNum+1;
                }
                descnum *=groupNum;
                descnum += index.vw2imgsList[i][-1];
                descnum /= groupNum+1;
                groupNum+=1;

            }else{
                //else, current group is saved and new query be new group
                f<<i-1<<endl;
                for(int x=0;x<100000;x++)
                    group[x] = query[x];
                descnum = index.vw2imgsList[i][-1];
                groupNum=1;
                f<<i<<" ";
            }

        }
        f<<3755<<endl;
        f.close();
    }
    void BagOfWords_WxBS::saveGeneralInfo(string name,int startnum){
        
        cout<<"GeneralInfo saving..."<<endl;
        ofstream f(name);

        f<< index.dirname<<endl;

        //total image size and map length save
        f<< index.numImgs<<endl;

        map<int, string>::iterator iter;
        for (iter = index.imgPath2id.begin(); iter != index.imgPath2id.end(); ++iter)
            f<<iter->first+startnum<<" "<<iter->second<<endl;
        

        f.close();
    
    }
    void BagOfWords_WxBS::loadGeneralInfo(string name){
        fstream f;
        f.open(name);
        models.vocabSize = params.numWords;
        f>>index.dirname;
        f>>index.numImgs;
        for(int i=0;i<index.numImgs;i++){
            int first;
            string second;

            f>>first>>second;
            index.imgPath2id.insert(pair<int,string>(first,second));
        }
        f.close();
    }
    void BagOfWords_WxBS::loadGeneralInfo(string name[]){
        fstream f1;
        f1.open(name[0]);
        fstream f2;
        f2.open(name[1]);
        fstream f3;
        f3.open(name[2]);

        models.vocabSize = params.numWords;
        f1>>index.dirname;
        f1>>index.numImgs;
        f2>>index.dirname;
        f2>>index.numImgs;
        f3>>index.dirname;
        f3>>index.numImgs;

        for(int i=0;i<index.numImgs;i++){
            int first;
            string second;

            f1>>first>>second;
            index.imgPath2id.insert(pair<int,string>(first*3,second));

            f2>>first>>second;
            first= first%index.numImgs;
            
            index.imgPath2id.insert(pair<int,string>(first*3+1,second));
            f3>>first>>second;
            first= first%index.numImgs;
            
            index.imgPath2id.insert(pair<int,string>(first*3+2,second));
        }
        f1.close();
        f2.close();
        f3.close();
        index.numImgs*=3;
    }

    void BagOfWords_WxBS::saveKeyPoints(string name){
        cout<<" saving keypoint..."<<endl;
        ofstream f(name);
        int i=0;
        for(i=0;i<index.numImgs;i++){    
                
                f<<RSIFTregionVector[i].size()<<endl;
                //keyregion save
                for(int j=0;j<RSIFTregionVector[i].size();j++){
                    //f<<*regionVector[i][j].bin[0].index<<" ";
                    f<<RSIFTregionVector[i][j].region.img_id<<" "<<RSIFTregionVector[i][j].region.img_reproj_id
                    <<" "<<RSIFTregionVector[i][j].region.id<<" "<<RSIFTregionVector[i][j].region.parent_id
                    //<<" "<<regionVector[i][j].region.type
                    //<<" "<<regionVector[i][j].region.det_kp.x
                    //<<" "<<regionVector[i][j].region.det_kp.y<<" "<<regionVector[i][j].region.det_kp.a11
                    //<<" "<<regionVector[i][j].region.det_kp.a12<<" "<<regionVector[i][j].region.det_kp.a21
                    //<<" "<<regionVector[i][j].region.det_kp.a22<<" "regionVector[i][j].region.det_kp.s
                    //<<" "<<regionVector[i][j].region.det_kp.response<<" "<<regionVector[i][j].region.det_kp.octave_number
                    //<<" "<<regionVector[i][j].region.det_kp.pyramid_scale<<" "<<regionVector[i][j].region.det_kp.sub_type
                    <<" "<<RSIFTregionVector[i][j].region.reproj_kp.x
                    <<" "<<RSIFTregionVector[i][j].region.reproj_kp.y<<" "<<RSIFTregionVector[i][j].region.reproj_kp.a11
                    <<" "<<RSIFTregionVector[i][j].region.reproj_kp.a12<<" "<<RSIFTregionVector[i][j].region.reproj_kp.a21
                    <<" "<<RSIFTregionVector[i][j].region.reproj_kp.a22<<" "<<RSIFTregionVector[i][j].region.reproj_kp.s
                    <<" "<<RSIFTregionVector[i][j].region.reproj_kp.response<<" "<<RSIFTregionVector[i][j].region.reproj_kp.octave_number
                    <<" "<<RSIFTregionVector[i][j].region.reproj_kp.pyramid_scale<<" "<<RSIFTregionVector[i][j].region.reproj_kp.sub_type;
                    f<<endl;

                    
                }
            }
        f.close();
    }
    void BagOfWords_WxBS::loadKeyPoints(string name){
        cout<<"RSIFT loading InvertedIndex..."<<endl;
        fstream f;
        f.open(name);
        
        vector<vector<nodes>> nodevv;
        for(int i=0;i<index.numImgs;i++){
            
            int length;
        
            f>>length;
            vector<nodes> nodev;
            for(int j=0;j<length;j++){
                nodes a;
                AffineRegion b;
                //f>>a.bin[0].index;
                f>>b.img_id>>b.img_reproj_id>>b.id;
                f>>b.parent_id;
                //f>>b.type;
                a.region = b;
                //>>a.region.det_kp.x>>a.region.det_kp.y
                //>>a.region.det_kp.a11>>a.region.det_kp.a12
                //>>a.region.det_kp.a21>>a.region.det_kp.a22
                //>>a.region.det_kp.s>>a.region.det_kp.response
                //>>a.region.det_kp.octave_number>>a.region.det_kp.pyramid_scale
                //>>a.region.det_kp.sub_type
                f>>a.region.reproj_kp.x>>a.region.reproj_kp.y;
                f>>a.region.reproj_kp.a11>>a.region.reproj_kp.a12;
                f>>a.region.reproj_kp.a21>>a.region.reproj_kp.a22;
                f>>a.region.reproj_kp.s>>a.region.reproj_kp.response;
                f>>a.region.reproj_kp.octave_number>>a.region.reproj_kp.pyramid_scale;
                f>>a.region.reproj_kp.sub_type;
                nodev.push_back(a);
            }
            nodevv.push_back(nodev);

            cout<<"done with "<< i+1<<"/"<<index.numImgs<<endl;
        }
        RSIFTregionVector=nodevv;
        f.close();
    }
    void BagOfWords_WxBS::loadKeyPoints(string name[]){
        cout<<"Full images loading InvertedIndex..."<<endl;
        fstream f1;
        f1.open(name[0]);
        fstream f2;
        f2.open(name[1]);
        fstream f3;
        f3.open(name[2]);
        
        vector<vector<nodes>> nodevv;
        for(int i=0;i<index.numImgs/3;i++){
            
            int length;
        
            f1>>length;
            //cout<<"f1 len: "<<length<<endl;
            vector<nodes> nodev;
            for(int j=0;j<length;j++){
                nodes a;
                AffineRegion b;
                //f>>a.bin[0].index;
                f1>>b.img_id>>b.img_reproj_id>>b.id;
                f1>>b.parent_id;
                //f>>b.type;
                a.region = b;
                //>>a.region.det_kp.x>>a.region.det_kp.y
                //>>a.region.det_kp.a11>>a.region.det_kp.a12
                //>>a.region.det_kp.a21>>a.region.det_kp.a22
                //>>a.region.det_kp.s>>a.region.det_kp.response
                //>>a.region.det_kp.octave_number>>a.region.det_kp.pyramid_scale
                //>>a.region.det_kp.sub_type
                f1>>a.region.reproj_kp.x>>a.region.reproj_kp.y;
                f1>>a.region.reproj_kp.a11>>a.region.reproj_kp.a12;
                f1>>a.region.reproj_kp.a21>>a.region.reproj_kp.a22;
                f1>>a.region.reproj_kp.s>>a.region.reproj_kp.response;
                f1>>a.region.reproj_kp.octave_number>>a.region.reproj_kp.pyramid_scale;
                f1>>a.region.reproj_kp.sub_type;
                nodev.push_back(a);
            }
            //cout<<"nodev len: "<<nodev.size()<<endl;
            nodevv.push_back(nodev);
            cout<<"done with "<< i*3+1<<"/"<<index.numImgs<<endl;

            nodev.clear();
            f2>>length;
            //cout<<"f2 len: "<<length<<endl;
            for(int j=0;j<length;j++){
                nodes a;
                AffineRegion b;
                //f>>a.bin[0].index;
                f2>>b.img_id>>b.img_reproj_id>>b.id;
                f2>>b.parent_id;
                a.region = b;
                f2>>a.region.reproj_kp.x>>a.region.reproj_kp.y;
                f2>>a.region.reproj_kp.a11>>a.region.reproj_kp.a12;
                f2>>a.region.reproj_kp.a21>>a.region.reproj_kp.a22;
                f2>>a.region.reproj_kp.s>>a.region.reproj_kp.response;
                f2>>a.region.reproj_kp.octave_number>>a.region.reproj_kp.pyramid_scale;
                f2>>a.region.reproj_kp.sub_type;
                nodev.push_back(a);
            }
            //cout<<"nodev len: "<<nodev.size()<<endl;
            nodevv.push_back(nodev);

            cout<<"done with "<< i*3+2<<"/"<<index.numImgs<<endl;

            nodev.clear();
            f3>>length;
            //cout<<"f3 len: "<<length<<endl;
            for(int j=0;j<length;j++){
                nodes a;
                AffineRegion b;
                //f>>a.bin[0].index;
                f3>>b.img_id>>b.img_reproj_id>>b.id;
                f3>>b.parent_id;
                a.region = b;
                f3>>a.region.reproj_kp.x>>a.region.reproj_kp.y;
                f3>>a.region.reproj_kp.a11>>a.region.reproj_kp.a12;
                f3>>a.region.reproj_kp.a21>>a.region.reproj_kp.a22;
                f3>>a.region.reproj_kp.s>>a.region.reproj_kp.response;
                f3>>a.region.reproj_kp.octave_number>>a.region.reproj_kp.pyramid_scale;
                f3>>a.region.reproj_kp.sub_type;
                nodev.push_back(a);
            }
            //cout<<"nodev len: "<<nodev.size()<<endl;
            nodevv.push_back(nodev);

            cout<<"done with "<< i*3+3<<"/"<<index.numImgs<<endl;
            //cout<<(long long unsigned int)sizeof(nodevv)<<endl;
            //cout<<"nodevvvvvv len: "<<nodevv.size()<<endl;
            
        }
        RSIFTregionVector=nodevv;
        f1.close();
        f2.close();
        f3.close();
    }
    void BagOfWords_WxBS::drawMatcher(Mat query,Mat DB,Mat result,TentativeCorrespListExt matchList,string name){
        result = Mat(480,640*2,query.type());
        //for(int y=0;y<result.cols;y++){
        //    for(int x=0;x<result.rows;x++){

        //    }
        //}
        cv::Mat roiImgResult_Left = result(cv::Rect(0,0,query.cols,query.rows));
        cv::Mat roiImgResult_Right = result(cv::Rect(query.cols,0,DB.cols,DB.rows));
        query.copyTo(roiImgResult_Left); //Img1 will be on the left of imgResult
        DB.copyTo(roiImgResult_Right); //Img2 will be on the right of imgResult
        RNG rng(12345);
        for(int i=0;i<matchList.TCList.size();i++){
                Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
                Point2f pt1=Point2f(matchList.TCList[i].first.reproj_kp.x,matchList.TCList[i].first.reproj_kp.y);
                Point2f pt2=Point2f(matchList.TCList[i].second.reproj_kp.x+640,matchList.TCList[i].second.reproj_kp.y);
                cv::line(result,pt1,pt2,color,1);
                circle(result,pt1,5,color,3);
                circle(result,pt2,5,color,3);

            }



        imshow(name,result);
    }
    void BagOfWords_WxBS::testMatcher(string I,int y,int n, int m){
        int query_num;
        int N=index.numImgs;
        //ImageRepresentation ImgRep1,ImgRep2;
        
        cout<<"?"<<endl;
        int VERB = Config1.OutputParam.verbose;
        vector<nodes> RSIFTbinlist,HRSIFTbinlist;
        computeImageRep(I, RSIFTbinlist,HRSIFTbinlist,query_num,0,n);
        cout<<"??"<<endl;
        //if(descname=="HRSIFT")RSIFTbinlist=HRSIFTbinlist;

        vector<nodes> out1,out2;
        CorrespondenceBank Tentatives;
        map<string, TentativeCorrespListExt> tentatives, verified_coors;
        int corrnum=0;
        if(descname=="RSIFT")
            corrnum = findCorrespondFeatures(RSIFTbinlist,RSIFTregionVector[y],out1,out2,index.RSIFTmatchlist[y]);
        else if(descname=="HRSIFT")
            corrnum = findCorrespondFeatures(HRSIFTbinlist,RSIFTregionVector[y],out1,out2,index.HRSIFTmatchlist[y]);
        cout<<"?/?"<<endl;
        
        //cout<<"corrnum: "<<corrnum<<endl;
        TentativeCorrespListExt current_tents;
        if(corrnum > m){
            
            //if(out1.size() != d1.size()){
            //    LoadRegions(ImgRep1,out1);
            //}
            //LoadRegions(ImgRep2,out2);


            //TODO convert to TentativeCorrespListExt
            
            //AffineRegionVector tempRegs1=imgrep1.GetAffineRegionVector("1","1");
            //AffineRegionVector tempRegs2=imgrep2.GetAffineRegionVector("1","1");
            //cout<<"out2.size(): "<<out2.size()<<endl;
            
            for(int i=0;i<out2.size();i++){
                TentativeCorrespExt tmp_corr;
                //if(out1.size() != d1.size()){
                //    tmp_corr.first = d1[i];
                //}else
                tmp_corr.first = out1[i].region;
                tmp_corr.second = out2[i].region;

                //cout << out1[i].region.reproj_kp.x<<", "<<out1[i].region.reproj_kp.y<<endl;
                //cout << out2[i].region.reproj_kp.x<<", "<<out2[i].region.reproj_kp.y<<endl;
                current_tents.TCList.push_back(tmp_corr);
            }
            //Tentatives.AddCorrespondences(current_tents,"1","1");
            
            //tentatives["All"] = Tentatives.GetCorresponcesVector("1","1");
            tentatives["All"]=current_tents;
            
        //2. matching using WxBS Matcher : geometric verification
            DuplicateFiltering(tentatives["All"], Config1.FilterParam.duplicateDist,Config1.FilterParam.mode);

            log1.Tentatives1st = tentatives["All"].TCList.size();
            //ransac(lo-ransac like degensac) with LAF check
            //if (VERB) std::cerr << "LO-RANSAC(epipolar) verification is used..." << endl;
            //cout<<"log1.Tentatives1st: "<<log1.Tentatives1st<<endl;

            if(log1.Tentatives1st>m){
                log1.TrueMatch1st =  LORANSACFiltering(tentatives["All"],
                                                    verified_coors["All"],
                                                    verified_coors["All"].H,
                                                    Config1.RANSACParam);
                log1.InlierRatio1st = (double) log1.TrueMatch1st / (double) log1.Tentatives1st;
            }
        }
        Mat query = imread(I);
        Mat DB = imread(index.imgPath2id[y]);
        //tentativelist 
        Mat tent_result;
        drawMatcher(query,DB,tent_result,tentatives["All"],"test_result");
        
        }
    
}