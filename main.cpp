#include <iostream>
#include "BoW.hpp"

using namespace std;
using namespace BoW;

int main(){

    BagOfWords BW;
    BW.getDefaultParam(100000,100000);
    BW.computeVocab("/home/jun/ImageDataSet/list.txt");

    return 0;

}
