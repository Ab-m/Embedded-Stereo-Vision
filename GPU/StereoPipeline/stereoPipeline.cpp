#include "stereoPipeline.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudawarping.hpp>
#include <benchmark/benchmark.h>
#include <libsgm.h>

stereoPipeline::stereoPipeline()
{
}

stereoPipeline::~stereoPipeline()
{
}

void stereoPipeline::CPU_getCameraParameters(std::string intrinsicsFile, std::string extrinsicsFile)
{
    cv::FileStorage ymlFile(intrinsicsFile, cv::FileStorage::READ);
    if (!ymlFile.isOpened())
    {
        printf("Failed to open intrinsics file \n");
    }
    ymlFile["M1"] >> M1;
    ymlFile["D1"] >> D1;
    ymlFile["M2"] >> M2;
    ymlFile["D2"] >> D2;
    ymlFile.open(extrinsicsFile, cv::FileStorage::READ);
    if (!ymlFile.isOpened())
    {
        printf("Failed to open extrinsics file \n");
    }
    ymlFile["R"] >> R;
    ymlFile["T"] >> T;
}
void stereoPipeline::CPU_computeRemapMatrix(cv::Mat imgL, cv::Mat imgR)
{
    cv::Mat temp_map11, temp_map12, temp_map21, temp_map22;
    stereoRectify(M1, D1, M2, D2, imgL.size(), R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, imgL.size(), &roi1, &roi2);
    initUndistortRectifyMap(M1, D1, R1, P1, imgL.size(), CV_8U, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, imgL.size(), CV_8U, map21, map22);

    initUndistortRectifyMap(M1, D1, R1, P1, imgL.size(), CV_32FC1, temp_map11, temp_map12);
    initUndistortRectifyMap(M2, D2, R2, P2, imgL.size(), CV_32FC1, temp_map21, temp_map22);

    GPU_map11.upload(temp_map11);
    GPU_map12.upload(temp_map12);
    GPU_map21.upload(temp_map21);
    GPU_map22.upload(temp_map22);
}
void stereoPipeline::CPU_rectifyStereoImages(cv::Mat &imgL, cv::Mat &imgR)
{

    cv::Mat tempL, tempR;

    remap(imgL, tempL, map11, map12, cv::INTER_LINEAR);
    remap(imgR, tempR, map21, map22, cv::INTER_LINEAR);

    (tempL).convertTo(imgL, CV_8U, 0.00390625);
    (tempR).convertTo(imgR, CV_8U, 0.00390625);
}

void stereoPipeline::CPU_initLBM(int WindowSize, int NumOfDisparity)
{
    BlockSize = WindowSize;
    MaxDisparity = NumOfDisparity;
    CPU_bm = cv::StereoBM::create(MaxDisparity, WindowSize);
    CPU_bm->setBlockSize(BlockSize);
    CPU_bm->setMinDisparity(0);
    CPU_bm->setNumDisparities(MaxDisparity);
    CPU_bm->setPreFilterSize(9);
    CPU_bm->setPreFilterCap(31);
    // CPU_bm->setTextureThreshold(1);
    // CPU_bm->setUniquenessRatio(1);
    //CPU_bm->setSpeckleWindowSize(50);
    //CPU_bm->setSpeckleRange(32);
    CPU_bm->setDisp12MaxDiff(1);
    CPU_bm->setROI1(roi1);
    CPU_bm->setROI2(roi2);
}
void stereoPipeline::CPU_initSGBM(int WindowSize, int NumOfDisparity)
{
    BlockSize = WindowSize;
    MaxDisparity = NumOfDisparity;
    CPU_sgbm = cv::StereoSGBM::create(0, MaxDisparity, WindowSize);
    CPU_sgbm->setBlockSize(BlockSize);
    CPU_sgbm->setMinDisparity(0);
    CPU_sgbm->setNumDisparities(MaxDisparity);
    CPU_sgbm->setPreFilterSize(9);
    CPU_sgbm->setPreFilterCap(31);
    //CPU_sgbm->setUniquenessRatio(3);
    //CPU_sgbm->setSpeckleWindowSize(100);
    //CPU_sgbm->setSpeckleRange(64);
    // CPU_sgbm->setDisp12MaxDiff(4);

    CPU_sgbm->setP1(200);
    CPU_sgbm->setP2(600);
    CPU_sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
}
cv::Mat stereoPipeline::CPU_LBM(cv::Mat imgL, cv::Mat imgR)
{
    cv::Mat Disparity, Disparity8bit;
    cv::cvtColor(imgL, temp_imgL, CV_BGR2GRAY);
    cv::cvtColor(imgR, temp_imgR, CV_BGR2GRAY);
    CPU_bm->setROI1(roi1);
    CPU_bm->setROI2(roi2);
    this->CPU_rectifyStereoImages(temp_imgL, temp_imgR);
    CPU_bm->compute(temp_imgL, temp_imgR, Disparity);
    Disparity.convertTo(Disparity, CV_8U, 255 / (MaxDisparity * 16.));
    return Disparity;
}

cv::Mat stereoPipeline::CPU_SGBM(cv::Mat imgL, cv::Mat imgR)
{
    cv::Mat Disparity;
    cv::cvtColor(*imgL, temp_imgL, CV_BGR2GRAY);
    cv::cvtColor(*imgR, temp_imgR, CV_BGR2GRAY);
    this->CPU_rectifyStereoImages(temp_imgL, temp_imgR);
    CPU_sgbm->compute(imgL, imgR, CPU_disp);
    CPU_disp.convertTo(CPU_disp, CV_8U, 255 / (MaxDisparity * 16.));
    return CPU_disp;
}
cv::Mat stereoPipeline::CPU_SGBM_HH(cv::Mat imgL, cv::Mat imgR)
{
    cv::Mat Disparity;
    cv::cvtColor(*imgL, temp_imgL, CV_BGR2GRAY);
    cv::cvtColor(*imgR, temp_imgR, CV_BGR2GRAY);
    this->CPU_rectifyStereoImages(temp_imgL, temp_imgR);

    CPU_sgbm->setMode(2); //HH mode

    CPU_sgbm->compute(imgL, imgR, Disparity);
    Disparity.convertTo(Disparity, CV_8U, 255 / (MaxDisparity * 16.));
    return Disparity;
}
cv::Mat stereoPipeline::CPU_SGBM_3WAY(cv::Mat *imgL, cv::Mat *imgR)
{
    cv::Mat Disparity;
    cv::cvtColor(*imgL, temp_imgL, CV_BGR2GRAY);
    cv::cvtColor(*imgR, temp_imgR, CV_BGR2GRAY);
    this->CPU_rectifyStereoImages(temp_imgL, temp_imgR);
    CPU_sgbm->setMode(4); //SGBM3WAY mode

    CPU_sgbm->compute(temp_imgL, temp_imgR, Disparity);
    Disparity.convertTo(Disparity, CV_8U, 255 / (MaxDisparity * 16.));
    return Disparity;
}

void stereoPipeline::GPU_initLBM(int WindowSize, int NumOfDisparity)
{
    BlockSize = WindowSize;
    MaxDisparity = NumOfDisparity;

    GPU_bm = cv::cuda::createStereoBM(NumOfDisparity, WindowSize);

    BlockSize = WindowSize;
    MaxDisparity = NumOfDisparity;
    // GPU_bm->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL );
    //GPU_bm->setUsePrefilter(true);
    //GPU_bm->setPreFilterSize(31);
    // GPU_bm->setPreFilterCap(11);
    GPU_bm->setBlockSize(BlockSize);
    GPU_bm->setMinDisparity(0);
    GPU_bm->setNumDisparities(MaxDisparity);
    //GPU_bm->setTextureThreshold(1);
    //  GPU_bm->setUniquenessRatio(3);
    //  GPU_bm->setSpeckleWindowSize(100);
    // GPU_bm->setSpeckleRange(32);
    // GPU_bm->setDisp12MaxDiff(1);
    //GPU_bm->setROI1(roi1);
    //GPU_bm->setROI2(roi2);
}
void stereoPipeline::GPU_initSGBM(int WindowSize, int NumOfDisparity)
{
    BlockSize = WindowSize;
    MaxDisparity = NumOfDisparity;
}
void stereoPipeline::GPU_initBP(int NumOfDisparity, int NumOfIterators, int NumOfLevels)
{
    GPU_bp = cv::cuda::createStereoBeliefPropagation(NumOfDisparity);
    GPU_bp->setNumIters(NumOfIterators);
    GPU_bp->setNumLevels(NumOfLevels);
}
void stereoPipeline::GPU_initCSBP(int NumOfDisparity, int NumOfIterators, int NumOfLevels)
{
    GPU_csbp = cv::cuda::createStereoConstantSpaceBP(MaxDisparity);
    GPU_bp->setNumIters(NumOfIterators);
    GPU_bp->setNumLevels(NumOfLevels);
}

cv::Mat stereoPipeline::GPU_SGBM(cv::Mat *imgL, cv::Mat *imgR)
{
    cv::cvtColor(*imgL, temp_imgL, CV_BGR2GRAY);
    cv::cvtColor(*imgR, temp_imgR, CV_BGR2GRAY);
    GPU_imgL.upload(temp_imgL, leftStream);
    GPU_imgR.upload(temp_imgR, rightStream);

    cv::cuda::remap(GPU_imgL, imgL_r, GPU_map11, GPU_map12, cv::INTER_LINEAR, 0, 0);
    cv::cuda::remap(GPU_imgR, imgR_r, GPU_map21, GPU_map22, cv::INTER_LINEAR, 0, 0);

    GPU_sgbm.execute(imgL_r, imgR_r, GPU_disp);

    cv::cuda::GpuMat Norm_disp;
    GPU_disp.convertTo(Norm_disp, CV_8UC1, 256. / MaxDisparity);
    Norm_disp.download(CPU_disp);
    return CPU_disp;
}

cv::Mat stereoPipeline::GPU_LBM(cv::Mat *imgL, cv::Mat *imgR)
{

    cv::cvtColor(*imgL, temp_imgL, CV_BGR2GRAY);
    cv::cvtColor(*imgR, temp_imgR, CV_BGR2GRAY);
    GPU_imgL.upload(temp_imgL, leftStream);
    GPU_imgR.upload(temp_imgR, rightStream);

    cv::cuda::remap(GPU_imgL, imgL_r, GPU_map11, GPU_map12, cv::INTER_LINEAR, 0, 0, leftStream);
    cv::cuda::remap(GPU_imgR, imgR_r, GPU_map21, GPU_map22, cv::INTER_LINEAR, 0, 0, rightStream);

    cv::cuda::equalizeHist(imgL_r, imgL_r, leftStream);
    cv::cuda::equalizeHist(imgR_r, imgR_r, rightStream);

    GPU_bm->compute(GPU_imgL, GPU_imgR, GPU_disp, rightStream);

    cv::cuda::GpuMat Norm_disp;

    GPU_disp.convertTo(Norm_disp, CV_8UC1, 256. / MaxDisparity);
    Norm_disp.download(CPU_disp);
    return CPU_disp;
}

cv::Mat stereoPipeline::GPU_BP(cv::Mat *imgL, cv::Mat *imgR)
{
    (*imgL).convertTo(*imgL, CV_8U, 0.00390625);
    (*imgR).convertTo(*imgR, CV_8U, 0.00390625);

    GPU_imgL.upload(*imgL);
    GPU_imgR.upload(*imgR);

    cv::cuda::remap(GPU_imgL, GPU_imgL, map11, map12, cv::INTER_LINEAR);
    cv::cuda::remap(GPU_imgR, GPU_imgR, map21, map22, cv::INTER_LINEAR);

    cv::Mat CPU_disp((*imgL).size(), CV_8U);
    cv::cuda::GpuMat GPU_disp((*imgL).size(), CV_8U);

    GPU_bp->compute(GPU_imgL, GPU_imgR, GPU_disp);

    GPU_disp.download(CPU_disp);
    return CPU_disp;
}
cv::Mat stereoPipeline::GPU_CSBP(cv::Mat *imgL, cv::Mat *imgR)
{

    GPU_imgL.upload(*imgL);
    GPU_imgR.upload(*imgR);

    cv::Mat CPU_disp((*imgL).size(), CV_8U);
    cv::cuda::GpuMat GPU_disp((*imgL).size(), CV_8U);

    cv::cuda::remap(GPU_imgL, GPU_imgL, map11, map12, cv::INTER_LINEAR);
    cv::cuda::remap(GPU_imgR, GPU_imgR, map21, map22, cv::INTER_LINEAR);

    GPU_csbp->compute(GPU_imgL, GPU_imgR, GPU_disp);

    GPU_disp.download(CPU_disp);
    return CPU_disp;
}
