#include "xf_headers.h"
#include "stereopipeline_hw.h"

class imagePairReader
{
private:
    /* data */
    int k;
    int num_imgs;
    char imageList[200][100];

public:
    imagePairReader(const std::string &filename);
    ~imagePairReader();
    void readNext(xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &leftMat, xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &rightMat);
};
