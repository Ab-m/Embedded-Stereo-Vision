#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <sstream>
class imagePairReader
{
private:
    /* data */
    int k;
    int num_imgs;
    char* imgs_directory;
    char* imgs_filename;
    char* extension;
    std::string alg;
    std::vector<std::string> imageList;

public:
    imagePairReader(const std::string& filename,std::string in_name);
    ~imagePairReader();
    void randomStart();
    void readNext(cv::Mat *imgL, cv::Mat *imgR);
    void writeImg(cv::Mat disp);
};
