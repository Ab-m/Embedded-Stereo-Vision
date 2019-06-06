#ifndef _STEREO_HW_
#define _STEREO_HW_

#include "hls_stream.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "imgproc/xf_stereo_pipeline.hpp"
#include "imgproc/xf_remap.hpp"
#include "imgproc/xf_stereoBM.hpp"

/* NO_OF_DISPARITIES must be greater than '0' and less than the image width */
#define NO_OF_DISPARITIES 160

/* NO_OF_DISPARITIES must not be lesser than PARALLEL_UNITS and NO_OF_DISPARITIES/PARALLEL_UNITS must be a non-fractional number */
#define PARALLEL_UNITS 80

/* SAD window size must be an odd number and it must be less than minimum of image height and width and less than the tested size '21' */
#define SAD_WINDOW_SIZE 15

// Configure this based on the number of rows needed for Remap function
#define XF_REMAP_BUFSIZE 128

/* config width and height */
#define XF_HEIGHT 1080
#define XF_WIDTH 1920

#define XF_CAMERA_MATRIX_SIZE 9
#define XF_DIST_COEFF_SIZE 5

#define IN_TYPE ap_uint<8>
#define OUT_TYPE ap_uint<16>

void stereoHW_ComputeRemap(xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapxLMat, xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapyLMat, xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapxRMat,
						   xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapyRMat, ap_fixed<32, 12> *cameraMA_l_fix, ap_fixed<32, 12> *cameraMA_r_fix, ap_fixed<32, 12> *distC_l_fix, ap_fixed<32, 12> *distC_r_fix,
						   ap_fixed<32, 12> *irA_l_fix, ap_fixed<32, 12> *irA_r_fix, int _cm_size, int _dc_size);

void stereoHW_RectifyLocalBlockMatch(xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &leftMat, xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &rightMat, xf::Mat<XF_16UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &dispMat,
									 xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapxLMat, xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapyLMat, xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapxRMat,
									 xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapyRMat, xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &leftRemappedMat, xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &rightRemappedMat,
									 xf::xFSBMState<SAD_WINDOW_SIZE, NO_OF_DISPARITIES, PARALLEL_UNITS> &bm_state);

#endif // _STEREO_HW__
