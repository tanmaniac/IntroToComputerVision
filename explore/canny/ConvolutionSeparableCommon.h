/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma once

////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
void setConvolutionKernel(float* h_Kernel);

void convolutionRowsGPU(float* d_Dst, float* d_Src, int imageW, int imageH);

void convolutionColumnsGPU(float* d_Dst, float* d_Src, int imageW, int imageH);
