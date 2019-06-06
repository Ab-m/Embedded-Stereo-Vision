/* Optimization type */
#define RO 			0    // Resource Optimized (8-pixel implementation)
#define NO 			1	 // Normal Operation (1-pixel implementation)

//max down scale factor 2 for all 1-pixel modes, and for upscale in x direction 
#define MAXDOWNSCALE 2

/* Interpolation type*/
#define INTERPOLATION	2
// 0 - Nearest Neighbor Interpolation
// 1 - Bilinear Interpolation
// 2 - AREA Interpolation

/* Input image Dimensions */
#define WIDTH 			3840	// Maximum Input image width
#define HEIGHT 			2160   	// Maximum Input image height

/* Output image Dimensions */
#define NEWWIDTH 		1920  // Maximum output image width
#define NEWHEIGHT 		1080  // Maximum output image height
