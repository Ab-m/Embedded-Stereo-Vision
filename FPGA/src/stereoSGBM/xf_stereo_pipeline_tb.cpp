cv::setUseOptimized(false);

if (argc != 3)
{
	fprintf(stderr, "Three arguments are required!\nUsage: ./<executable> <YML intrinsics file> <YML extrinsics file> <XML image list>\n");
	return -1;
}

printf("Reading camera parameters\n");
cv::Mat M1, D1, M2, D2, R, T, R1, R2, Q, irAL, irAR, P1temp, P2temp;
cv::FileStorage ymlFile(argv[1], 0);
if (!ymlFile.isOpened())
{
	printf(stderr, "Failed to open intrinsics file \n");
	return -1;
}
ymlFile["M1"] >> M1;
ymlFile["D1"] >> D1;
ymlFile["M2"] >> M2;
ymlFile["D2"] >> D2;
ymlFile.open(argv[2], 0);
if (!ymlFile.isOpened())
{
	printf(stderr, "Failed to open extrinsics file \n");
	return -1;
}
ymlFile["R"] >> R;
ymlFile["T"] >> T;
ymlFile["R1"] >> R1;
ymlFile["R2"] >> R2;
ymlFile["P1"] >> P1temp;
ymlFile["P2"] >> P2temp;
cv::Mat P1(3, 3, CV_64FC1);
cv::Mat P2(3, 3, CV_64FC1);

cv::Mat left_img, right_img;
imagePairReader images(argv[3], left_img, right_img);

int rows = left_img.rows;
int cols = left_img.cols;

xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> leftMat(rows, cols);
xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> rightMat(rows, cols);

static xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> mapxLMat(rows, cols);
static xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> mapyLMat(rows, cols);
static xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> mapxRMat(rows, cols);
static xf::Mat<XF_32FC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> mapyRMat(rows, cols);

static xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> leftRemappedMat(rows, cols);
static xf::Mat<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> rightRemappedMat(rows, cols);

static xf::Mat<XF_16UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1> dispMat(rows, cols);

// camera parameters for rectification
#if __SDSCC__
ap_fixed<32, 12> *cameraMA_l_fix = (ap_fixed<32, 12> *)sds_alloc_non_cacheable(XF_CAMERA_MATRIX_SIZE * sizeof(ap_fixed<32, 12>));
ap_fixed<32, 12> *cameraMA_r_fix = (ap_fixed<32, 12> *)sds_alloc_non_cacheable(XF_CAMERA_MATRIX_SIZE * sizeof(ap_fixed<32, 12>));
ap_fixed<32, 12> *irA_l_fix = (ap_fixed<32, 12> *)sds_alloc_non_cacheable(XF_CAMERA_MATRIX_SIZE * sizeof(ap_fixed<32, 12>));
ap_fixed<32, 12> *irA_r_fix = (ap_fixed<32, 12> *)sds_alloc_non_cacheable(XF_CAMERA_MATRIX_SIZE * sizeof(ap_fixed<32, 12>));
ap_fixed<32, 12> *distC_l_fix = (ap_fixed<32, 12> *)sds_alloc_non_cacheable(XF_DIST_COEFF_SIZE * sizeof(ap_fixed<32, 12>));
ap_fixed<32, 12> *distC_r_fix = (ap_fixed<32, 12> *)sds_alloc_non_cacheable(XF_DIST_COEFF_SIZE * sizeof(ap_fixed<32, 12>));
#else
ap_fixed<32, 12> *cameraMA_l_fix = (ap_fixed<32, 12> *)malloc(XF_CAMERA_MATRIX_SIZE * sizeof(ap_fixed<32, 12>));
ap_fixed<32, 12> *cameraMA_r_fix = (ap_fixed<32, 12> *)malloc(XF_CAMERA_MATRIX_SIZE * sizeof(ap_fixed<32, 12>));
ap_fixed<32, 12> *irA_l_fix = (ap_fixed<32, 12> *)malloc(XF_CAMERA_MATRIX_SIZE * sizeof(ap_fixed<32, 12>));
ap_fixed<32, 12> *irA_r_fix = (ap_fixed<32, 12> *)malloc(XF_CAMERA_MATRIX_SIZE * sizeof(ap_fixed<32, 12>));
ap_fixed<32, 12> *distC_l_fix = (ap_fixed<32, 12> *)malloc(XF_DIST_COEFF_SIZE * sizeof(ap_fixed<32, 12>));
ap_fixed<32, 12> *distC_r_fix = (ap_fixed<32, 12> *)malloc(XF_DIST_COEFF_SIZE * sizeof(ap_fixed<32, 12>));
#endif

//	leftMat.copyTo(left_img.data);
//	rightMat.copyTo(right_img.data);
leftMat = xf::imread<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(argv[1], 0);
rightMat = xf::imread<XF_8UC1, XF_HEIGHT, XF_WIDTH, XF_NPPC1>(argv[2], 0);

xf::xFSBMState<SAD_WINDOW_SIZE, NO_OF_DISPARITIES, PARALLEL_UNITS> bm_state;
bm_state.preFilterCap = 31;
bm_state.uniquenessRatio = 15;
bm_state.textureThreshold = 20;
bm_state.minDisparity = 0;

/* oseya add */

for (int i = 0; i < 3; i++)
{
	for (int j = 0; j < 3; j++)
	{
		P1.at<double>(i, j) = P1temp.at<double>(i, j);
		P2.at<double>(i, j) = P2temp.at<double>(i, j);
	}
}
irAL = (P1 * R1).inv();
irAR = (P2 * R2).inv();
/* print stuff */

printf("P1 = \n");
for (int r = 0; r < P1.rows; r++)
{
	for (int c = 0; c < P1.cols; c++)
	{
		printf("%*.4f ", 10, P1.at<double>(r, c));
	}
	printf("\n");
}
printf("R1 = \n");
for (int r = 0; r < R1.rows; r++)
{
	for (int c = 0; c < R1.cols; c++)
	{
		printf("%*.4f ", 10, R1.at<double>(r, c));
	}
	printf("\n");
}

printf("irAL = \n");
for (int r = 0; r < irAL.rows; r++)
{
	for (int c = 0; c < irAL.cols; c++)
	{
		printf("%*.4f ", 10, irAL.at<double>(r, c));
	}
	printf("\n");
}

printf("irAR = \n");
for (int r = 0; r < irAR.rows; r++)
{
	for (int c = 0; c < irAR.cols; c++)
	{
		printf("%*.4f ", 10, irAR.at<double>(r, c));
	}
	printf("\n");
}

for (int i = 0; i < 3; i++)
{
	for (int j = 0; j < 3; j++)
	{
		int a = (3 * i) + j;
		cameraMA_l_fix[a] = (ap_fixed<32, 12>)(M1.at<double>(i, j));
		cameraMA_r_fix[a] = (ap_fixed<32, 12>)(M2.at<double>(i, j));
		irA_l_fix[a] = (ap_fixed<32, 12>)(irAL.at<double>(i, j));
		irA_r_fix[a] = (ap_fixed<32, 12>)(irAR.at<double>(i, j));
	}
}
printf("D1 = \n");
for (int i = 0; i < 4; i++)
{
	distC_l_fix[i] = (ap_fixed<32, 12>)(D1.at<double>(0, i));
	distC_r_fix[i] = (ap_fixed<32, 12>)(D2.at<double>(0, i));
	printf("%*.4f ", 10, D1.at<double>(0, i));
}
distC_l_fix[4] = (ap_fixed<32, 12>)(D1.at<double>(0, 7));
distC_r_fix[4] = (ap_fixed<32, 12>)(D2.at<double>(0, 7));
printf("%*.4f \n", 10, D1.at<double>(0, 7));

//accelerated remap matrix calculation
stereoHW_ComputeRemap(mapxLMat, mapyLMat, mapxRMat, mapyRMat, cameraMA_l_fix, cameraMA_r_fix, distC_l_fix, distC_r_fix, irA_l_fix, irA_r_fix, 9, 5);
char file_name[100];

cv::VideoWriter video("StereoVideo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 20, cv::Size(cols, rows));
cv::Mat out_disp_16(rows, cols, CV_16UC1);
cv::Mat out_disp_img(rows, cols, CV_8UC1);
printf("Starting hardware acceleration \n");

perf_counter pipeline_ctr;
perf_counter loop_ctr;

loop_ctr.start();

for (int i = 0; i < 100; i++)
{
	pipeline_ctr.start();
	images.readNext(leftMat, rightMat);

	stereoHW_SGBM(leftMat, rightMat, dispMat, SMALL_PENALTY, LARGE_PENALTY);

	pipeline_ctr.stop();
	uint64_t pipeline_ms = pipeline_ctr.avg_cpu_cycles() / 1500000;
	printf("The pipeline took %ju ms \n ", pipeline_ms);
	if (i > 0)
	{
		sprintf(file_name, "Results/Disparity%d.png", i);
		out_disp_16.data = dispMat.copyFrom();
		out_disp_16.convertTo(out_disp_img, CV_8U, (256.0 / NO_OF_DISPARITIES) / (16.));
		video.write(out_disp_img);
	}
	uint64_t overall_ms = overall_ctr.avg_cpu_cycles() / 1500000;
	printf("The overall loop cycle took %ju ms \n ", overall_ms);
	pipeline_ctr.reset();
}
loop_ctr.stop();
uint64_t loop_ms = loop_ctr.avg_cpu_cycles() / 1500000;

printf("100 cycles took: %ju ms \n ", loop_ms);

out_disp_16.data = dispMat.copyFrom();
out_disp_16.convertTo(out_disp_img, CV_8U, (256.0 / NO_OF_DISPARITIES) / (16.));

imwrite("sgbm_out.png", out_disp_img);

video.release();
printf("run complete !\n\n");

return 0;
}
