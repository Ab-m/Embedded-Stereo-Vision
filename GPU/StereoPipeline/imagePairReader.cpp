#include "imagePairReader.hpp"

imagePairReader::imagePairReader(const std::string& filename,std::string in_name)
{
    k=0;
    num_imgs=0;
    imageList.resize(0);
    alg.assign(in_name);
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    cv::FileNode n = fs.getFirstTopLevelNode();
    cv::FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
       { //std::cout << ((std::string)*it) << '\n';
        imageList.push_back((std::string)*it);
        num_imgs++;
       }


}

imagePairReader::~imagePairReader()
{
}

void imagePairReader::randomStart()
{
    k = (rand() % (num_imgs/2))*2;
}
void imagePairReader::readNext(cv::Mat *imgL, cv::Mat *imgR)
{
    if(k>=(num_imgs)) k=0;
    *imgL = cv::imread(imageList[k],-1);
    k++;
    *imgR = cv::imread(imageList[k],-1);
    k++;
}
void imagePairReader::writeImg(cv::Mat disp)
{

    std::ostringstream outFile;
    outFile << "at/" << alg << k/2 << ".png";
    std::string fn = outFile.str();
    cv::imwrite(fn, disp);
}
