#ifndef _STEREO_HW_
#define _STEREO_HW_

#include "imgproc/xf_sgbm.hpp"

#include "hls_stream.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "imgproc/xf_stereo_pipeline.hpp"
#include "imgproc/xf_remap.hpp"

#define IN_T XF_8UC1
#define OUT_T XF_8UC1
/* SGBM */
/* set the height and width */
#define XF_HEIGHT 1080
#define XF_WIDTH 1920
/* set penalties for SGM */
#define SMALL_PENALTY 20
#define LARGE_PENALTY 40
/* Census transform window size */
#define WINDOW_SIZE 7
/* NO_OF_DISPARITIES must be greater than '0' and less than the image width */
#define NO_OF_DISPARITIES 160
/* NO_OF_DISPARITIES must not be lesser than PARALLEL_UNITS and NO_OF_DISPARITIES/PARALLEL_UNITS must be a non-fractional number */
#define PARALLEL_UNITS 40
/* Number of directions */
#define NUM_DIR 3

#define XF_REMAP_BUFSIZE 128
#define XF_CAMERA_MATRIX_SIZE 9
#define XF_DIST_COEFF_SIZE 5


#define IN_TYPE ap_uint<8>
#define OUT_TYPE ap_uint<16>

void stereoHW_ComputeRemap(xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapxLMat, xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapyLMat, xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapxRMat,
						   xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapyRMat, ap_fixed<32, 12> *cameraMA_l_fix, ap_fixed<32, 12> *cameraMA_r_fix, ap_fixed<32, 12> *distC_l_fix, ap_fixed<32, 12> *distC_r_fix,
						   ap_fixed<32, 12> *irA_l_fix, ap_fixed<32, 12> *irA_r_fix, int _cm_size, int _dc_size);

void stereoHW_SGBM(xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &leftMat, xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &rightMat, xf::Mat<OUT_T, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &dispMat,
				   xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapxLMat, xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapyLMat, xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapxRMat,
				   xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapyRMat, xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &leftRemappedMat, xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &rightRemappedMat, unsigned char p1, unsigned char p2);

#endif // _STEREO_HW__
