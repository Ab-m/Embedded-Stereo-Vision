#include "stereopipeline_hw.h"

void stereoHW_ComputeRemap(
	xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapxLMat, xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapyLMat, xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapxRMat,
	xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> &mapyRMat, ap_fixed<32, 12> *cameraMA_l_fix, ap_fixed<32, 12> *cameraMA_r_fix, ap_fixed<32, 12> *distC_l_fix, ap_fixed<32, 12> *distC_r_fix,
	ap_fixed<32, 12> *irA_l_fix, ap_fixed<32, 12> *irA_r_fix, int _cm_size, int _dc_size)
{
#pragma HLS INTERFACE m_axi depth = 9 port = cameraMA_l_fix offset = direct bundle = cameraMA
#pragma HLS INTERFACE m_axi depth = 9 port = distC_l_fix offset = direct bundle = distC
#pragma HLS INTERFACE m_axi depth = 9 port = irA_l_fix offset = direct bundle = irA
#pragma HLS INTERFACE m_axi depth = 9 port = cameraMA_r_fix offset = direct bundle = cameraMA
#pragma HLS INTERFACE m_axi depth = 9 port = distC_r_fix offset = direct bundle = distC
#pragma HLS INTERFACE m_axi depth = 9 port = irA_r_fix offset = direct bundle = irA

	xf::InitUndistortRectifyMapInverse<XF_CAMERA_MATRIX_SIZE, XF_DIST_COEFF_SIZE, XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(cameraMA_l_fix, distC_l_fix, irA_l_fix, mapxLMat, mapyLMat, _cm_size, _dc_size);

	xf::InitUndistortRectifyMapInverse<XF_CAMERA_MATRIX_SIZE, XF_DIST_COEFF_SIZE, XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(cameraMA_r_fix, distC_r_fix, irA_r_fix, mapxRMat, mapyRMat, _cm_size, _dc_size);
}
