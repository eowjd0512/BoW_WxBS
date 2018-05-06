//pragma once
#include "BoW.hpp"
#include "iindex.hpp"
#include <iostream>
#include <fstream>
#include "cv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "./thirdParty/vlfeat/src/generic-driver.h"
#include "./thirdParty/vlfeat/vl/generic.h"
#include "./thirdParty/vlfeat/vl/sift.h"
#include "./thirdParty/vlfeat/vl/mathop.h"
#include <./thirdParty/vlfeat/vl/kmeans.h>
#include <./thirdParty/vlfeat/vl/kdtree.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
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

    void BagOfWords::imageSearch(Mat I,int topn){
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
        int num;
        VlKDForestNeighbor* d = computeImageRep(I, num);
        
        cout<< "Tf-Idf..."<<endl;
        cout<<"query image descriptorn: "<<num<<endl;
        int Ni=0;
        int N=index.numImgs;
        num = models.vocabSize;
        Mat tfIdf = Mat::zeros(Size(num,N),CV_64FC1);

        //count n_id
        for(int y=0;y<N;y++){
            //cout<<index.imgPath2id[y]<<endl;
            for(int x=0;x<num;x++){
                
                if(0<=d[x].index && d[x].index<num){
                    if(index.vw2imgsList[y].count(d[x].index)){
                        tfIdf.at<double>(y,d[x].index)+=index.vw2imgsList[y][d[x].index];
                        //tfIdf.at<double>(y,x)+=index.vw2imgsList[y][d[x].index];
                    }
                }
                //cout <<tfIdf.at<double>(y,x)<<" ";
            }
            //cout<<endl<<endl;
        }
        //count n_d and calculate tf
        for(int y=0;y<N;y++){
            for(int x=0;x<num;x++){
                if(tfIdf.at<double>(y,x)>0.){
                    tfIdf.at<double>(y,x)/=double(index.vw2imgsList[y][-1]);
                }
            }
        }
    /*    
        //calculate tf-idf by multiplying idf gotten
        for(int x=0;x<num;x++){
            Ni=0;
            for(int y=0;y<N;y++){
                if(index.vw2imgsList[y].count(d[x].index)){
                    Ni++;
                }
            }
            //cout<<"x word: "<<x <<"th,  "<< d[x].index<< " NI: "<<Ni<<endl;
            for(int y=0;y<N;y++){
                if(tfIdf.at<double>(y,x)>0.){
                    tfIdf.at<double>(y,x)*=log10(double(N)/double(Ni));
                }
            }
        }
*/
        //for(int y=0;y<N;y++){
            //cout<<index.imgPath2id[9]<<endl;
            //for(int x=0;x<num;x++){
            //    cout <<tfIdf.at<double>(9,x)<<" ";
            //}
            //cout<<endl<<endl;
        //}

        //for efficiency to sort, exchange the indeces of img and sum, i.e. int,double -> double, int
        vector<pair<double,int>> score;
        //sum all of elemnts to score
        for(int y=0;y<N;y++){
            double sum=0.;
            for(int x=0;x<num;x++){
                if(tfIdf.at<double>(y,x)>0.){
                    sum += tfIdf.at<double>(y,x);
                }
            }
            score.push_back(make_pair(sum,y));
        }
        //sort
        sort(score.begin(), score.end(),greater());
        for(int i=0;i<topn;i++)
            cout<<"top "<<i<<" |score: "<<score[i].first<<" | at "<<index.imgPath2id[score[i].second]<<endl;
    }

    VlKDForestNeighbor* BagOfWords::computeImageRep(Mat I,int &num){
        // Computes an image representation (of I) using the quantization 
        // parameters in the model
        // @param I : image after imread
        // @param model : model as generated by bow_computeVocab
        // @return : f (Same as from vl_sift) and bins = quantized descriptor values
        vector<vector<vl_sift_pix>> desc_vec;
        vl_sift_set_peak_thresh(sift,3);
        int width = I.cols;
        int height = I.rows;
        int val=0;
        int descNum=0;
        vl_sift_pix* fdata = (vl_sift_pix*)malloc(width*height*sizeof(vl_sift_pix));
        for (int y = 0 ; y <height ; y++) {
            for (int x = 0 ; x <width ; x++) {
            fdata [y*width+x] = (vl_sift_pix)I.at<vl_uint8>(y,x);
            }
        }
        int i=0;
        for (; ;){
            VlSiftKeypoint const *keys = 0 ;
            int nkeys;
            if(i==0){
                val = vl_sift_process_first_octave(sift, fdata);
                i++;
            }else{
                val = vl_sift_process_next_octave(sift);
            }
            //printf("sift: GSS octave %d computed\n",vl_sift_get_octave_index (sift));
            if(val){
                val = VL_ERR_OK ; break;
            }
            vl_sift_detect(sift);

            keys  = vl_sift_get_keypoints(sift);
            nkeys = vl_sift_get_nkeypoints(sift);
            //printf ("sift: detected %d (unoriented) keypoints\n", nkeys) ;
            for(int j=0; j<nkeys;j++){

                double                angles [4] ;
                int                   nangles ;
                VlSiftKeypoint        jk ;
                VlSiftKeypoint const *k ;

                k = keys + j ;
                nangles =vl_sift_calc_keypoint_orientations(sift, angles, k);
                
                for(int q=0; q<(unsigned) nangles;q++){//for each orientation:
                    //vl_sift_pix descr [128] ;
                    vl_sift_pix  buf [128] ;
                    vl_sift_pix rbuf [128] ;
                    vector<vl_sift_pix> dv;
                    vl_sift_calc_keypoint_descriptor(sift, buf, k, angles[q] );
                    transpose_descriptor (rbuf, buf) ;
                    
                    //if (dsc.active) {
                    int l ;
                    for (l = 0 ; l < 128 ; l++) {
                        
                        //double x = 512.0 * descr[l] ;
                        double x = 512.0 * rbuf[l] ;
                        x = (x < 255.0) ? x : 255.0 ;
                        //cout<<x<<" ";
                        //descr[l] = x;
                        dv.push_back((vl_sift_pix)x);
                    }
                    desc_vec.push_back(dv);
                    descNum++;
                }
            }
        }
        //index.totalDescriptors.push_back(descNum);
        int sizevec = num = descNum;
        //cout<<"size vec: "<<sizevec<<endl;
        double* desc;
        desc = (double*)vl_malloc(128*sizevec*sizeof(double));

            
        for(int p=0;p<sizevec;p++){
            for(int j=0;j<128;j++){
            
            desc[p*128+j] = desc_vec[p][j];
            
            }
            //cout<<endl;
        }
        VlKDForestSearcher* quary= vl_kdforest_new_searcher(models.kdtree); 	
        VlKDForestNeighbor * bin =(VlKDForestNeighbor *)vl_malloc(sizevec*sizeof(VlKDForestNeighbor));
        
        vl_kdforestsearcher_query(quary,bin,sizevec,desc);
        //vl_kdforest_query(models.kdtree,bin,models.vocabSize,desc);
        free(fdata);
        free(desc);
        return bin;
    }
    void BagOfWords::buildInvIndex(string imgsDir){
        // Build an inverted index for all image files in 'imgsDir' (recursively
        // searched) given the visual word quantization 'model'
        // model can be path to mat file too
        // Optional Parameters:
        // 'imgsListFpath', 'path/to/file.txt' :- File contains a newline separated
        // 'resDir', 'path/to/results'
        // list of image paths (relative to imgsDir) of the image files to 
        // build index upon. Typically used to set the train set.
        //save iindex.mat
        //string indexfpath = "/home/jun/ImageDataSet/iindex.txt";
        //string indexfpath = fullfile(imgsDir, 'iindex.mat');
        //if exist(indexfpath, 'file')
            //fprintf(2, 'The iindex file already exists. Remove it first\n');
            //return;
        //end
        //index = matfile(indexfpath, 'Writable', true);

        //index.imgPath2id = containers.Map(index.imgPaths,1 : index.numImgs);
        //fullpaths = cellfun2(@(x) fullfile(imgsDir, x), index.imgPaths);

        //// create inverted index
        
        //index.totalDescriptors = zeros(index.numImgs, 1);
        // will store the total # of words in each image
        // Create a cell array of vocabSize containers.Map (Assuming vocab ids are
        // 1..n). Each element stores <imgID : times that VW appears in that image>
        // Have to call it multiple times to initialize in a loop.. using 
        // `repmat` or `deal` simply makes multiple references to same object and 
        // that doesn't work
        vector<map<int,int>> vw2imgsList;
        //vw2imgsList.reserve(models.vocabSize);
        vw2imgsList.reserve(index.numImgs);
        //for i = 1 : model.vocabSize
        //    vw2imgsList{i} = containers.Map('KeyType', 'int64', 'ValueType', 'int64');
        //end
        
        for(int i=0;i<index.numImgs;i++){
            int num=0;
            map<int,int> init;
            vw2imgsList.push_back(init);
        //for i = 1 : index.numImgs
            try{
                Mat I = imread(index.imgPath2id[i],0);
                resize(I, I, Size(640,480));
                
                VlKDForestNeighbor* d = computeImageRep(I, num);
                //[~, d] = bow_computeImageRep(I, model, 'PeakThresh', 3);//TODO conversion
                //return descriptor vectors
///*
                //index.totalDescriptors[i] = d.size();
                for(int j=0; j<num;j++){
                //for j = 1 : numel(d)
                    //map<int,int> imgsList = vw2imgsList{d(j)};
                    //map<int,int> imgsList = vw2imgsList[d[j].index];
                    map<int,int> imgsList = vw2imgsList[i];

                    //if (imgsList.count(i)>0)
                    if (imgsList.count(d[j].index))
                        imgsList[d[j].index] = imgsList[d[j].index]+1;
                        //imgsList(i) = imgsList(i) + 1;
                    else
                        imgsList[d[j].index] = 1;
                        
                    //vw2imgsList[d[j].index] = imgsList;
                    vw2imgsList[i] = imgsList;
                    //vw2imgsList{d(j)} = imgsList;
                //end
                }
                vw2imgsList[i].insert(pair<int,int>(-1,num));
            }
            catch(int e){
                cerr<<"the error in making inverted index"<<endl;
                //disp(getReport(e));
                continue;
            }
            //end
            printf("nFeat = %d. Indexed (%d / %d)\n", num, i+1, index.numImgs);
            //if (i% 1000 == 0){
            //    index.vw2imgsList = vw2imgsList;
            //}
//*/
        }
        index.vw2imgsList = vw2imgsList;


        //if 1
        //    fprintf('Saving to %s after %d files\n', fullfile(imgsDir, 'iindex.mat'), i);
        //    save(fullfile(imgsDir, 'iindex.mat'), 'iindex', '-v7.3');
        //end
    }


    int BagOfWords::computeVocab(string imgsDir, int numImg){
        // Read all images recursively in imgsDir and learn a vocabulary by AKM
        // Optional param
        // 'imgsListFpath', 'path/to/file.txt' :- File contains a newline separated
        // list of image paths (relative to imgsDir) of the image files to 
        // build index upon. Typically used to set the train set.
        // 'avgSiftsPerImg', <count> :- (default: 1000). Used to pre-allocate the
        // storage array. Give an upper bound estimate. But take care that num_imgs
        // * avg_sift memory will be allocated.. so it may crash if the machine 
        // can't handle it.
        // params.numWords = size of voacbulary to learn
        // params.maxImgsForVocab = max number of images to use for computing it
    
        //p = inputParser;
        //addOptional(p, 'imgsListFpath', 0);
        //addOptional(p, 'avgSiftsPerImg', 400);
        //parse(p, varargin{:});
        //VlFileMeta out  = {1, "%.sift",  VL_PROT_ASCII, "", 0} ;
        //VlFileMeta frm  = {0, "%.frame", VL_PROT_ASCII, "", 0} ;
        //VlFileMeta dsc  = {1, "%.descr", VL_PROT_ASCII, "", 0} ;


        index.dirname = imgsDir;
        //index.totalDescriptors.reserve(index.numImgs);
        int avgSiftsPerImg = 1000;
        
        vector<vector<vl_sift_pix>> desc_vec;
        int est_n = avgSiftsPerImg*56; // expected number of sifts
        vl_sift_set_edge_thresh(sift,10);
        vl_sift_set_window_size(sift, 2);
        

        // Read images and create set of SIFTs
        //Mat toFloat; 
        //cvmat.convertTo(toFloat,CV_32F);
        //float *vlimage = (float*) tofloat.data;
        
        //descs = zeros(128, est_n, 'uint8'); // 128 x n dim matrix, for n SIFTs
        int found_sifts = 0;
        cout<<"Reading SIFTs "<<endl;
        /*
        for i = 1 : numel(fullpaths)
            // best to read one by one, in case of large number of images
            try
                I = single(rgb2gray(imread(fullpaths{i})));
                I = imresize(I,[640,480]);
            catch
                fprintf(2, 'Unable to read %s\n', fullpaths{i});
                continue;
            end
        */

        string frpaths;
        fstream f;
        f.open(imgsDir);
        int totalcnt=0;
        //[~, d] = vl_sift(I);
        
        int m=0;
        for (m = 0; m<numImg;m++){
            f>>frpaths;
            string path = "/home/jun/ImageDataSet/trainImg/"+frpaths;
            //cout<<path<<endl;
            //index.imgPaths.push_back(path);
            //// Get imgs list
            // Add these paths to a hash map as well
            index.imgPath2id.insert(pair<int,string>(m,path));
            Mat currImg = imread(path,0);
            
            resize(currImg, currImg, Size(640,480));
            
            //imshow("d",currImg);
            //waitKey(0);
            int width = currImg.cols;
            int height = currImg.rows;
            int val=0;
            //float* currImg_data = (float*)currImg.data;
            vl_sift_pix* fdata = (vl_sift_pix*)malloc(width*height*sizeof(vl_sift_pix));
            for (int y = 0 ; y <height ; y++) {
                for (int x = 0 ; x <width ; x++) {
                //cout<<y*width+x<<endl;
                fdata [y*width+x] = (vl_sift_pix)currImg.at<vl_uint8>(y,x);
            }//vl_sift_pix, uchar, 
                //cout<<q<<endl;
                //fdata [q] = currImg_data [q] ;
            }
            int totalnKey=0;
            int i=0;
            int descNum=0;
            for (; ;){
                VlSiftKeypoint const *keys = 0 ;
                int nkeys;
                if(i==0){
                    //cout<<"1"<<endl;
                    val = vl_sift_process_first_octave(sift, fdata);
                    //cout<<"2"<<endl;
                    i++;
                }else{
                    val = vl_sift_process_next_octave(sift);
                }
                //printf("sift: GSS octave %d computed\n",vl_sift_get_octave_index (sift));
                if(val){
                    val = VL_ERR_OK ; break;
                }
                vl_sift_detect(sift);

                keys  = vl_sift_get_keypoints  (sift) ;
                nkeys = vl_sift_get_nkeypoints (sift) ;
                totalnKey+=nkeys;
                //printf ("sift: detected %d (unoriented) keypoints\n", nkeys) ;
                for(int j=0; j<nkeys;j++){
                    
                    double                angles [4] ;
                    int                   nangles ;
                    VlSiftKeypoint        jk ;
                    VlSiftKeypoint const *k ;

                    k = keys + j ;
                    nangles =vl_sift_calc_keypoint_orientations(sift, angles, k);
                    
                    for(int q=0; q<(unsigned) nangles;q++){//for each orientation:
                        vl_sift_pix  buf [128] ;
 
                        vl_sift_pix rbuf [128] ;
                        
                        //vl_sift_pix descr [128] ;
                        vector<vl_sift_pix> dv;
                        vl_sift_calc_keypoint_descriptor(sift, buf, k, angles[q] );

                        transpose_descriptor (rbuf, buf) ;

                        //if (dsc.active) {
                        int l ;
                        for (l = 0 ; l < 128 ; l++) {
                            
                            //double x = 512.0 * descr[l] ;
                            double x = 512.0 * rbuf[l] ;
                            x = (x < 255.0) ? x : 255.0 ;
                            //cout<<x<<" ";
                            //descr[l] = x;
                            dv.push_back((vl_sift_pix)x);
                        //cout<<"??"<<endl;
                        //vl_file_meta_put_uint8 (&dsc, (vl_uint8) (x)) ;
                        //cout<<"???"<<endl;
                        }
                        desc_vec.push_back(dv);

                        //for(int j=0;j<128;j++){
                        //cout<<desc_vec[totalcnt][j] <<" ";
                        //}
                        //cout<<endl;
                        totalcnt++;
                        descNum++;
                        //cout<<endl;
                        //if (dsc.protocol == VL_PROT_ASCII) fprintf(dsc.file, "\n") ;
                        //}
                        
                    }
                }

            }
            //printf ("sift: detected %d (unoriented) keypoints\n", totalnKey) ;
            printf ("sift: detected %d (unoriented) descriptors (%d)\n", descNum,m+1) ;
            free(fdata);
            //free(currImg_data);
            
        }

        if (1) {
 
            printf("vl_sift: filter settings:\n") ;
 
            printf("vl_sift:   octaves      (O)      = %d\n",
 
                vl_sift_get_noctaves      (sift)) ;
 
            printf("vl_sift:   levels       (S)      = %d\n",
 
                vl_sift_get_nlevels       (sift)) ;
 
            printf("vl_sift:   first octave (o_min)  = %d\n",
 
                vl_sift_get_octave_first  (sift)) ;
 
            printf("vl_sift:   edge thresh           = %g\n",
 
                vl_sift_get_edge_thresh   (sift)) ;
 
            printf("vl_sift:   peak thresh           = %g\n",
 
                vl_sift_get_peak_thresh   (sift)) ;
 
            printf("vl_sift:   norm thresh           = %g\n",
 
                vl_sift_get_norm_thresh   (sift)) ;
 
            printf("vl_sift:   window size           = %g\n",
 
                vl_sift_get_window_size   (sift)) ;
 
            printf("vl_sift:   float descriptor      = %d\n",
 
                0) ;
    
        }
        index.numImgs = numImg;
            //textprogressbar(i * 100.0 / numel(fullpaths));
        
        cout<<"Done"<<endl;
        //cout<<totalcnt<<endl;
        int sizevec = int(desc_vec.size());
        cout<<"size vec: "<<sizevec<<endl;
        double* desc;
        desc = (double*)vl_malloc(128*sizevec*sizeof(double));

            
        for(int p=0;p<sizevec;p++){
            for(int j=0;j<128;j++){
            
            desc[p*128+j] = desc_vec[p][j];
            
            }
            //cout<<endl;
        }
        /*for(int j=0;j<128;j++){
            int p=1;
            cout<< desc_vec[p][j]<<" ";
            
        }
        cout<<endl<<endl;
        for(int j=0;j<128;j++){
            int p=2;
            cout<< desc[p*128+j]<<" ";
            
        }
        cout<<endl<<endl;
*/
        f.close();
        cout<< "Found " <<sizevec<<" descriptors. Clustering now..."<<endl;
        // K Means cluster the SIFTs, and create a model
        models.vocabSize = params.numWords;
        //vl_file_meta_close (&dsc) ;
        vl_size numData = sizevec;
        vl_size dimension = 128;
        vl_size numCenters = min(sizevec, params.numWords);
        vl_size maxiter = 100;
        vl_size maxComp = 100;
        vl_size maxrep = 1;
        vl_size ntrees = 3;
        //vl_size dataIdx, d;

    
        //double * centers ;
        // Use float data and the L2 distance for clustering
        
        vl_kmeans_set_verbosity	(kmeans,1);
        // Use Lloyd algorithm
        vl_kmeans_set_algorithm(kmeans, VlKMeansANN) ;
        vl_kmeans_set_max_num_comparisons (kmeans, maxComp) ;
        vl_kmeans_set_num_repetitions (kmeans, maxrep) ;
        vl_kmeans_set_num_trees (kmeans, ntrees);
        //vl_sift_pix const* c_desc = desc;
        vl_kmeans_set_max_num_iterations (kmeans, maxiter) ;
        // Initialize the cluster centers by randomly sampling the data
        vl_kmeans_init_centers_plus_plus (kmeans, desc, dimension, numData, numCenters) ;
        // Run at most 100 iterations of cluster refinement using Lloyd algorithm
        
        //vl_kmeans_refine_centers (kmeans, desc, numData) ;
        vl_kmeans_cluster(kmeans,desc,dimension,numData,numCenters);
        // Obtain the cluster centers
        //centers = (double*)vl_kmeans_get_centers(kmeans) ;
        
        
        //for(int p=0;p<sizevec;p++){
            //for(int j=0;j<128;j++){
            //cout<< centers[(numCenters-1)*128+j]<<" ";
            //desc[p*128+j] = desc_vec[p][j];
            
            //}
            //cout<<endl;
        //}

        //vl_uint32 * assignment = vl_malloc(sizeof(vl_uint32) * numData) ;
        //float * distance = vl_malloc(sizeof(float) * numData) ;
        //vl_kmeans_quantize_ANN(kmeans, assignment, distance, double(desc), numData) ;
        //models.assignments = assignment;
        //models.distances = distance;
        
        //vl_kmeans_cluster(kmeans,data,dimension,numData,numCenters);
        //VlKDForest* kdtree =  vl_kdforest_new(VL_TYPE_FLOAT,128,ntrees,VlDistanceL2);
        vl_kdforest_build(models.kdtree,numCenters,kmeans->centers);
        //models.kdtree= kdtree;
        models.vocabSize = numCenters;
        vl_free(desc);
        
        return 0;

        //descs = descs(:, 1 : found_sifts);
        
/*
        
        VlKDForest* kdtree =  vl_kdforest_new(VL_TYPE_FLOAT,128,numTrees,VlDistanceL2);
        vl_kdforest_build(kdtree,numData,data);
        models.kdtrees= kdtree;

        vl_kdforest_delete(kdtree);
        */
       
        //model.vocab = vl_kmeans(double(descs), ...
        //                        min(size(descs, 2), params.numWords), 'verbose', ...
        //                        'algorithm', 'ANN');
        //model.kdtree = vl_kdtreebuild(model.vocab);
            
    }

    
}