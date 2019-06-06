/***************************************************************************
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors 
may be used to endorse or promote products derived from this software 
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
HOWEVER CXFSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

***************************************************************************/

#ifndef _XF_CANNY_CONFIG_H__
#define _XF_CANNY_CONFIG_H__

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "imgproc/xf_canny.hpp"
#include "imgproc/xf_edge_tracing.hpp"
#include "xf_config_params.h"

#define WIDTH 3840
#define HEIGHT 2160

#if NO
#define INTYPE XF_NPPC1
#define OUTTYPE XF_NPPC4
#elif RO
#define INTYPE XF_NPPC8
#define OUTTYPE XF_NPPC8
#endif

#if L1NORM
#define NORM_TYPE XF_L1NORM
#elif L2NORM
#define NORM_TYPE XF_L2NORM
#endif

void canny_accel(xf::Mat<XF_8UC1, HEIGHT, WIDTH, INTYPE> &_src,xf::Mat<XF_2UC1, HEIGHT, WIDTH, OUTTYPE> &_dst,xf::Mat<XF_2UC1, HEIGHT, WIDTH, XF_NPPC32> &_dst1,xf::Mat<XF_8UC1, HEIGHT, WIDTH, XF_NPPC8> &_dst2,unsigned char low_threshold,unsigned char high_threshold);

#endif
