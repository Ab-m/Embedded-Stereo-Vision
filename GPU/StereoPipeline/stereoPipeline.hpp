#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/cudastereo.hpp"
#include <iostream>
#include <stdio.h>

#include <libsgm.h>

class stereoPipeline
{
private:
    std::string img1_filename, img2_filename;
    /* Parameters from calibration used in rectification process */
    cv::Mat M1, D1, M2, D2, R, T, R1, P1, R2, P2, Q, CPU_disp, temp_imgL, temp_imgR;

    /* Matrices used in remaping the image into a rectified image */
    cv::Mat map11, map12, map21, map22;
    cv::Rect roi1, roi2;

    int BlockSize, MaxDisparity;
	cv::cuda::Stream leftStream, rightStream;

    cv::cuda::GpuMat GPU_imgL, GPU_imgR, imgL_r, imgR_r, GPU_map11 , GPU_map12, GPU_map21, GPU_map22, GPU_disp;
    /* Stereo Matching */

    cv::Ptr<cv::StereoBM> CPU_bm;
    cv::Ptr<cv::StereoSGBM> CPU_sgbm;

    cv::Ptr<cv::cuda::StereoBM> GPU_bm;
    sgm::LibSGMWrapper GPU_sgbm{128};
    cv::Ptr<cv::cuda::StereoBeliefPropagation> GPU_bp;
    cv::Ptr<cv::cuda::StereoConstantSpaceBP> GPU_csbp;


public:
    stereoPipeline();
    ~stereoPipeline();
    
    /*  Functions that run on CPU */
    void CPU_getCameraParameters(std::string = "intrinsics.yml", std::string = "extrinsics.yml");
    void CPU_computeRemapMatrix(cv::Mat *imgL , cv::Mat *imgR);
    void CPU_rectifyStereoImages(cv::Mat *img1 , cv::Mat *img2);

    void CPU_initLBM(int = 15, int = 160);
    void CPU_initSGBM(int = 7, int = 160);

    cv::Mat CPU_LBM(cv::Mat *imgL , cv::Mat *imgR);
    // Semi global block matching
    cv::Mat CPU_SGBM(cv::Mat *imgL , cv::Mat *imgR);
    // H. Hirschmueller
    cv::Mat CPU_SGBM_3WAY(cv::Mat *imgL , cv::Mat *imgR);
    cv::Mat CPU_SGBM_HH(cv::Mat *imgL , cv::Mat *imgR);


    /*  Functions that run on GPU */
    void GPU_rectifyStereoImages(cv::Mat *imgL , cv::Mat *imgR);

    void GPU_initLBM(int = 15, int = 160);
    void GPU_initSGBM(int = 15, int = 128);
    void GPU_initBP(int = 160, int = 10, int = 10);
    void GPU_initCSBP(int = 160, int = 10, int = 10);

    cv::Mat GPU_LBM(cv::Mat *imgL , cv::Mat *imgR);
	cv::Mat GPU_SGBM(cv::Mat *imgL , cv::Mat *imgR);
    cv::Mat GPU_BP(cv::Mat *imgL , cv::Mat *imgR);
    cv::Mat GPU_CSBP(cv::Mat *imgL , cv::Mat *imgR);

};
