#include "stereopipeline_hw.h"
#include "imgproc/xf_sgbm.hpp"

void stereoHW_SGBM(xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &leftMat, xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &rightMat, xf::Mat<OUT_T, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &dispMat,
				   xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapxLMat, xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapyLMat, xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapxRMat,
				   xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapyRMat, xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &leftRemappedMat, xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &rightRemappedMat, unsigned char p1, unsigned char p2)
{

#pragma SDS async(1)
#pragma SDS resource(1)
	xf::remap<XF_REMAP_BUFSIZE, XF_INTERPOLATION_BILINEAR, XF_8UC1, XF_32FC1, XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(leftMat, leftRemappedMat, mapxLMat, mapyLMat);

#pragma SDS async(2)
#pragma SDS resource(2)
	xf::remap<XF_REMAP_BUFSIZE, XF_INTERPOLATION_BILINEAR, XF_8UC1, XF_32FC1, XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(rightMat, rightRemappedMat, mapxRMat, mapyRMat);

#pragma SDS wait(1)
#pragma SDS wait(2)
#pragma SDS wait(3)
#pragma SDS async(3)
	xf::SemiGlobalBM<XF_BORDER_CONSTANT, WINDOW_SIZE, NO_OF_DISPARITIES, PARALLEL_UNITS, NUM_DIR, IN_T, OUT_T, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(leftRemappedMat, rightRemappedMat, dispMat, p1, p2);
}
