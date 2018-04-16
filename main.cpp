#include <iostream>
#include "BoW.hpp"

using namespace std;
using namespace BoW;

int main(){

    BagOfWords BW;
    BW.getDefaultParam(10000,10000);
    BW.computeVocab("/home/jun/ImageDataSet/list.txt");

    return 0;

}
