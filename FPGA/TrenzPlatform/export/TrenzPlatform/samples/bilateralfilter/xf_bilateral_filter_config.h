/***************************************************************************
Copyright (c) 2016, Xilinx, Inc.
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
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

***************************************************************************/

#ifndef _XF_BILATERAL_FILTER_CONFIG_H_
#define _XF_BILATERAL_FILTER_CONFIG_H_
#include "hls_stream.h"
#include "ap_int.h"

#include "common/xf_common.h"
#include "xf_config_params.h"
#include "imgproc/xf_bilateral_filter.hpp"

typedef unsigned short int  uint16_t;

#define ERROR_THRESHOLD 0 // acceptable error threshold range 0 to 255

#if RO
#define IN_TYPE ap_uint<64>
#define OUT_TYPE IN_TYPE
#define NPC1 XF_NPPC8
#endif
#if NO
#define IN_TYPE ap_uint<8>
#define OUT_TYPE IN_TYPE
#define NPC1 XF_NPPC1
#endif

#if FILTER_SIZE_3
#define FILTER_WIDTH 3
#elif FILTER_SIZE_5
#define FILTER_WIDTH 5
#elif FILTER_SIZE_7
#define FILTER_WIDTH 7
#endif




#define TYPE XF_8UC1


void bilateral_filter_accel(xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &_src, xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &_dst, float sigma_color, float sigma_space);
#endif //_AU_BILATERAL_FILTER_CONFIG_H_
