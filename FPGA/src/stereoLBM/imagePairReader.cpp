#include "imagePairReader.h"
#include "xf_headers.h"

imagePairReader::imagePairReader(const std::string &filename, cv::Mat &left, cv::Mat &right)
{
    char *cstr = new char[100];
    k = 0;
    num_imgs = 0;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    cv::FileNode n = fs.getFirstTopLevelNode();
    cv::FileNodeIterator it = n.begin(), it_end = n.end();

    for (; it != it_end; ++it)
    {
        strcpy(imageList[num_imgs++], ((std::string)*it).c_str());
    }
    delete[] cstr;
    left = cv::imread(imageList[0], 0);
    right = cv::imread(imageList[1], 0);
}

imagePairReader::~imagePairReader()
{
}

void imagePairReader::readNext(xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &leftMat, xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &rightMat)
{
    if (k >= (num_imgs - 2))
        k = 0;
    printf("image pair read number %s \n", imageList[k]);
    leftMat = xf::imread<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(imageList[k], 0);
    k++;
    printf("image pair read number %s \n", imageList[k]);
    rightMat = xf::imread<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(imageList[k], 0);
    k++;
}
