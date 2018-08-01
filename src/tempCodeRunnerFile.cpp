ofstream f;
        f.open(savef,ios::app);
        f<<endl<<endl;
        f<<"query image: "<<queryList[i]<<endl;
        f<<"grdth image: "<<groundTruth[i]<<endl;
        f.close();