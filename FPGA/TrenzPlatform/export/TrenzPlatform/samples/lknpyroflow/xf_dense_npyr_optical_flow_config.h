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
 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ***************************************************************************/
#ifndef __XF_DENSE_NONPYR_OPTICAL_FLOW_CONFIG__
#define __XF_DENSE_NONPYR_OPTICAL_FLOW_CONFIG__

#include <iostream>
#include "common/xf_common.h"
#include "common/xf_types.h"
#include "xf_config_params.h"
#include "imgproc/xf_dense_npyr_optical_flow.hpp"

#if WORD_SZ == 2
	#define NPPC XF_NPPC2
#else
	#define NPPC XF_NPPC1
#endif
void dense_non_pyr_of_accel(xf::Mat<XF_8UC1, MAX_HEIGHT, MAX_WIDTH, NPPC> &buf0, xf::Mat<XF_8UC1, MAX_HEIGHT, MAX_WIDTH, NPPC> &buf1, xf::Mat<XF_32FC1,MAX_HEIGHT, MAX_WIDTH, NPPC> &flowx, xf::Mat<XF_32FC1,MAX_HEIGHT, MAX_WIDTH, NPPC> &flowy);

#endif

