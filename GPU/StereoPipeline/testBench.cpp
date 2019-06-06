#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/core/utility.hpp"
#include <iostream>
#include <chrono>
#include <benchmark/benchmark.h>
#include "imagePairReader.hpp"
#include "stereoPipeline.hpp"
#include "TX2PowerMonitor.hpp"

//Manual time is used to get correct timing for GPU performance
#define BENCHMARK_TX2(x) BENCHMARK(x)->UseManualTime()->Unit(benchmark::kMillisecond)->MinTime(30)

static void BM_GPU_LBM(benchmark::State &state)
{

        TX2PowerMonitor pmon{state};
        stereoPipeline stereo;
        imagePairReader images("image_list.xml", "gl");
        cv::Mat imgL, imgR;
        cv::Mat colorDisp;
        images.randomStart();
        images.readNext(&imgL, &imgR);

        stereo.CPU_getCameraParameters();
        stereo.CPU_computeRemapMatrix(&imgL, &imgR);
        stereo.GPU_initLBM(15, 160);

        //Ramp up pipeline;
        cv::Mat DisparityMap = stereo.GPU_LBM(&imgL, &imgR);
        images.readNext(&imgL, &imgR);
        DisparityMap = stereo.GPU_LBM(&imgL, &imgR);
        for (auto _ : state)
        {

                images.readNext(&imgL, &imgR);

                auto start = std::chrono::high_resolution_clock::now();
                DisparityMap = stereo.GPU_LBM(&imgL, &imgR);
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);
                state.SetIterationTime(elapsed_seconds.count());
                pmon.measurePower();
                images.writeImg(DisparityMap);
        }
        pmon.reportAverage(state);
}
static void BM_GPU_SGBM(benchmark::State &state)
{
        TX2PowerMonitor pmon{state};
        stereoPipeline stereo;
        imagePairReader images("image_list.xml", "gs");
        cv::Mat imgL, imgR;
        cv::Mat colorDisp;
        images.randomStart();
        images.readNext(&imgL, &imgR);

        stereo.CPU_getCameraParameters();
        stereo.CPU_computeRemapMatrix(&imgL, &imgR);
        stereo.GPU_initSGBM(15, 160);

        //Ramp up pipeline;
        cv::Mat DisparityMap = stereo.GPU_SGBM(&imgL, &imgR);
        images.readNext(&imgL, &imgR);
        DisparityMap = stereo.GPU_SGBM(&imgL, &imgR);
        for (auto _ : state)
        {
                images.readNext(&imgL, &imgR);
                auto start = std::chrono::high_resolution_clock::now();
                DisparityMap = stereo.GPU_SGBM(&imgL, &imgR);
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);
                state.SetIterationTime(elapsed_seconds.count());
                pmon.measurePower();
                images.writeImg(DisparityMap);
        }
        pmon.reportAverage(state);
}

static void BM_CPU_LBM(benchmark::State &state)
{
        TX2PowerMonitor pmon{state};
        stereoPipeline stereo;
        imagePairReader images("image_list.xml", "cl");
        cv::Mat imgL, imgR;
        cv::Mat colorDisp;
        images.randomStart();
        images.readNext(&imgL, &imgR);

        stereo.CPU_getCameraParameters();
        stereo.CPU_computeRemapMatrix(&imgL, &imgR);
        stereo.CPU_initLBM(15, 160);

        //Ramp up pipeline;
        cv::Mat DisparityMap = stereo.CPU_LBM(&imgL, &imgR);
        images.readNext(&imgL, &imgR);
        DisparityMap = stereo.CPU_LBM(&imgL, &imgR);
        for (auto _ : state)
        {
                images.readNext(&imgL, &imgR);
                auto start = std::chrono::high_resolution_clock::now();
                DisparityMap = stereo.CPU_LBM(&imgL, &imgR);
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);
                state.SetIterationTime(elapsed_seconds.count());
                pmon.measurePower();
                images.writeImg(DisparityMap);
        }

        pmon.reportAverage(state);
}
static void BM_CPU_SGBM3(benchmark::State &state)
{
        TX2PowerMonitor pmon{state};
        stereoPipeline stereo;
        imagePairReader images("image_list.xml", "cs");
        cv::Mat imgL, imgR;
        cv::Mat colorDisp;
        images.randomStart();
        images.readNext(&imgL, &imgR);

        stereo.CPU_getCameraParameters();
        stereo.CPU_computeRemapMatrix(&imgL, &imgR);
        stereo.CPU_initSGBM(15, 160);

        //Ramp up pipeline;
        cv::Mat DisparityMap = stereo.CPU_SGBM_3WAY(&imgL, &imgR);
        images.readNext(&imgL, &imgR);
        DisparityMap = stereo.CPU_SGBM_3WAY(&imgL, &imgR);
        for (auto _ : state)
        {
                images.readNext(&imgL, &imgR);
                auto start = std::chrono::high_resolution_clock::now();
                //stereo.CPU_rectifyStereoImages(&imgL,&imgR);
                DisparityMap = stereo.CPU_SGBM_3WAY(&imgL, &imgR);
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start);
                state.SetIterationTime(elapsed_seconds.count());
                pmon.measurePower();

                pmon.measurePower();
                images.writeImg(DisparityMap);
        }
        pmon.reportAverage(state);
}

BENCHMARK_TX2(BM_GPU_LBM);
BENCHMARK_TX2(BM_GPU_SGBM);
BENCHMARK_TX2(BM_CPU_LBM);
BENCHMARK_TX2(BM_CPU_SGBM3);

BENCHMARK_MAIN();
