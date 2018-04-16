#pragma once
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
        string indexfpath = "/home/jun/ImageDataSet/iindex.txt";
        //string indexfpath = fullfile(imgsDir, 'iindex.mat');
        //if exist(indexfpath, 'file')
            //fprintf(2, 'The iindex file already exists. Remove it first\n');
            //return;
        //end
        //index = matfile(indexfpath, 'Writable', true);

        //// Get imgs list
        
        
        // Add these paths to a hash map as well
        //index.imgPath2id = containers.Map(index.imgPaths,1 : index.numImgs);
        fullpaths = cellfun2(@(x) fullfile(imgsDir, x), index.imgPaths);

        //// create inverted index
        index.totalDescriptors = zeros(index.numImgs, 1);
        // will store the total # of words in each image
        // Create a cell array of vocabSize containers.Map (Assuming vocab ids are
        // 1..n). Each element stores <imgID : times that VW appears in that image>
        // Have to call it multiple times to initialize in a loop.. using 
        // `repmat` or `deal` simply makes multiple references to same object and 
        // that doesn't work
        for i = 1 : model.vocabSize
            vw2imgsList{i} = containers.Map('KeyType', 'int64', 'ValueType', 'int64');
        end

        for i = 1 : index.numImgs
            try
                I = imread(fullpaths{i});
                I = imresize(I,[640,480]);
                [~, d] = bow_computeImageRep(I, model, 'PeakThresh', 3);
                index.totalDescriptors(i, 1) = numel(d);
                for j = 1 : numel(d)
                    imgsList = vw2imgsList{d(j)};
                    if imgsList.isKey(i)
                        imgsList(i) = imgsList(i) + 1;
                    else
                        imgsList(i) = 1;
                    end
                    vw2imgsList{d(j)} = imgsList;
                end
            catch e
                disp(getReport(e));
                continue;
            end
            fprintf(2, 'nFeat = %d. Indexed (%d / %d)\n', numel(d), i, index.numImgs);
            if mod(i, 1000) == 0
                index.vw2imgsList = vw2imgsList;
            end
        end
        index.vw2imgsList = vw2imgsList;


        if 1
            fprintf('Saving to %s after %d files\n', fullfile(imgsDir, 'iindex.mat'), i);
            save(fullfile(imgsDir, 'iindex.mat'), 'iindex', '-v7.3');
        end

    }


    int BagOfWords::computeVocab(string imgsDir){
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
    #ifdef test
    VlRand rand ;

  vl_size numData = 100000;
  vl_size dimension = 256;
  vl_size numCenters = 400;
  vl_size maxiter = 10;
  vl_size maxComp = 100;
  vl_size maxrep = 1;
  vl_size ntrees = 1;

  double * data;

  vl_size dataIdx, d;

  //VlKMeansAlgorithm algorithm = VlKMeansANN ;
  VlKMeansAlgorithm algorithm = VlKMeansLloyd ;
  //VlKMeansAlgorithm algorithm = VlKMeansElkan ;
  VlVectorComparisonType distance = VlDistanceL2 ;
  VlKMeans * kmeans = vl_kmeans_new (VL_TYPE_DOUBLE,distance) ;

  vl_rand_init (&rand) ;
  vl_rand_seed (&rand,  1000) ;

  data = (double*)vl_malloc(sizeof(double) * dimension * numData);

  for(dataIdx = 0; dataIdx < numData; dataIdx++) {
    for(d = 0; d < dimension; d++) {
      double randomNum = (double)vl_rand_real3(&rand)+1;
      data[dataIdx*dimension+d] = randomNum;
    }
  }

  vl_kmeans_set_verbosity	(kmeans,1);
  vl_kmeans_set_max_num_iterations (kmeans, maxiter) ;
  vl_kmeans_set_max_num_comparisons (kmeans, maxComp) ;
  vl_kmeans_set_num_repetitions (kmeans, maxrep) ;
  vl_kmeans_set_num_trees (kmeans, ntrees);
  vl_kmeans_set_algorithm (kmeans, algorithm);

  //struct timeval t1,t2;
  //gettimeofday(&t1, NULL);

  vl_kmeans_cluster(kmeans,data,dimension,numData,numCenters);

  //gettimeofday(&t2, NULL);

  //VL_PRINT("elapsed vlfeat: %f s\n",(double)(t2.tv_sec - t1.tv_sec) + ((double)(t2.tv_usec - t1.tv_usec))/1000000.);

  vl_kmeans_delete(kmeans);
  vl_free(data);
    return 0 ;
    #endif

    #ifdef nontest
        index.dirname = imgsDir;
        int avgSiftsPerImg = 1000;
        
        vector<vector<vl_sift_pix>> desc_vec;
        int est_n = avgSiftsPerImg*56; // expected number of sifts

        

        // Read images and create set of SIFTs
        //Mat toFloat; 
        //cvmat.convertTo(toFloat,CV_32F);
        //float *vlimage = (float*) tofloat.data;
        
        //descs = zeros(128, est_n, 'uint8'); // 128 x n dim matrix, for n SIFTs
        int found_sifts = 0;
        cout<<'Reading SIFTs '<<endl;
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
        for (int m = 0; m<10;m++){
            f>>frpaths;
            string path = "/home/jun/ImageDataSet/"+frpaths;
            cout<<path<<endl;
            index.imgPaths.push_back(path);
            index.imgPath2id.insert(pair<int,string>(m+1,path));
            Mat currImg = imread(path,0);
            resize(currImg, currImg, Size(640,480));
            //imshow("d",currImg);
            //waitKey(0);

            int width = currImg.cols;
            int height = currImg.rows;
            int noctaves = log2(min(width,height));
            //int noctaves = log2(min(width,height));
            int nlevels = 3;
            int o_min= 0;
            int val=0;
            VlSiftFilt* sift = vl_sift_new(width, height, noctaves, nlevels, o_min);
            //float* currImg_data = (float*)currImg.data;
            vl_sift_pix* fdata = (vl_sift_pix*)malloc(width*height*sizeof(vl_sift_pix));
            for (int y = 0 ; y <height ; y++) {
                for (int x = 0 ; x <width ; x++) {
                //cout<<y*width+x<<endl;
                fdata [y*width+x] = currImg.at<int>(y,x);
            }
                //cout<<q<<endl;
                //fdata [q] = currImg_data [q] ;
            }
            int i=0;
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
                printf ("sift: detected %d (unoriented) keypoints\n", nkeys) ;
                for(int j=0; j<nkeys;++j){
                    double                angles [4] ;
                    int                   nangles ;
                    VlSiftKeypoint        jk ;
                    VlSiftKeypoint const *k ;

                    k = keys + j ;
                    nangles =vl_sift_calc_keypoint_orientations(sift, angles, k);
                    
                    for(int q=0; q<(unsigned) nangles;++q){//for each orientation:
                        vl_sift_pix descr [128] ;
                        vector<vl_sift_pix> dv;
                        vl_sift_calc_keypoint_descriptor(sift, descr, k, angles[q] );
                        
                        
                        //if (dsc.active) {
                        int l ;
                        for (l = 0 ; l < 128 ; ++l) {
                            
                            double x = 512.0 * descr[l] ;
                            x = (x < 255.0) ? x : 255.0 ;
                            //cout<<x<<" ";
                            descr[l] = x;
                            dv.push_back(x);
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
                        //cout<<endl;
                        //if (dsc.protocol == VL_PROT_ASCII) fprintf(dsc.file, "\n") ;
                        //}
                        
                    }
                }
            }

            vl_sift_delete(sift);
            free(fdata);
            //free(currImg_data);
        }
            //textprogressbar(i * 100.0 / numel(fullpaths));
        
        cout<<"Done"<<endl;
        cout<<totalcnt<<endl;
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
        cout<<"??"<<endl;
        vl_size numData = sizevec;
        vl_size dimension = 128;
        vl_size numCenters = min(sizevec, params.numWords);
        cout<< "numCenter: "<<numCenters<<endl;
        vl_size maxiter = 100;
        vl_size maxComp = 100;
        vl_size maxrep = 1;
        vl_size ntrees = 10;
        vl_size dataIdx, d;
        cout<<"??"<<endl;
        double energy ;
        double * centers ;
        // Use float data and the L2 distance for clustering
        VlKMeans * kmeans = vl_kmeans_new (VL_TYPE_DOUBLE, VlDistanceL2) ;
        vl_kmeans_set_verbosity	(kmeans,1);
        // Use Lloyd algorithm
        vl_kmeans_set_algorithm(kmeans, VlKMeansANN) ;
        vl_kmeans_set_max_num_comparisons (kmeans, maxComp) ;
        vl_kmeans_set_num_repetitions (kmeans, maxrep) ;
        vl_kmeans_set_num_trees (kmeans, ntrees);
        //vl_sift_pix const* c_desc = desc;
        
        // Initialize the cluster centers by randomly sampling the data
        vl_kmeans_init_centers_with_rand_data (kmeans, desc, dimension, numData, numCenters) ;
        // Run at most 100 iterations of cluster refinement using Lloyd algorithm
        vl_kmeans_set_max_num_iterations (kmeans, maxiter) ;
        vl_kmeans_refine_centers (kmeans, desc, numData) ;
        //vl_kmeans_cluster(kmeans,desc,dimension,numData,numCenters);
        // Obtain the energy of the solution
        energy = vl_kmeans_get_energy(kmeans) ;
        models.enrgy = energy;
        cout<<"energy: "<<energy<<endl;
        // Obtain the cluster centers
        centers = (double*)vl_kmeans_get_centers(kmeans) ;
        models.centers = centers;
        cout<<"??"<<endl;
        
        //for(int p=0;p<sizevec;p++){
            for(int j=0;j<128;j++){
            cout<< centers[(sizevec-1)*128+j]<<" ";
            //desc[p*128+j] = desc_vec[p][j];
            
            }
            cout<<endl;
        //}

        //vl_uint32 * assignment = vl_malloc(sizeof(vl_uint32) * numData) ;
        //float * distance = vl_malloc(sizeof(float) * numData) ;
        //vl_kmeans_quantize_ANN(kmeans, assignment, distance, double(desc), numData) ;
        //models.assignments = assignment;
        //models.distances = distance;
        

        

        //vl_kmeans_cluster(kmeans,data,dimension,numData,numCenters);
        VlKDForest* kdtree =  vl_kdforest_new(VL_TYPE_FLOAT,128,ntrees,VlDistanceL2);
        vl_kdforest_build(kdtree,numData,desc);
        models.kdtree= kdtree;

        vl_kdforest_delete(kdtree);
        vl_kmeans_delete(kmeans);
        vl_free(desc);
        return 0;

        //descs = descs(:, 1 : found_sifts);
        #endif
        
        
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