

#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_HOUGH_OP_GPU_H
#define TENSORFLOW_CORE_KERNELS_HOUGH_OP_GPU_H

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"


namespace tensorflow {


bool DiscreteHough(
	const int batches, const int channels,
	const int height_bottom, const int height_top,
	const int width_bottom, const int width_top,
	float* top, const float* bottom,
	const int64* map, const float threshold,
	const Eigen::GpuDevice& d);

bool DiscreteHough(
	const int batches, const int channels,
	const int height_bottom, const int height_top,
	const int width_bottom, const int width_top,
	Eigen::half* top, const Eigen::half* bottom,
	const int64* map, const float threshold,
	const Eigen::GpuDevice& d);


bool DiscreteHoughGrad(
	const int batches, const int channels,
	const int height_bottom, const int height_top,
	const int width_bottom, const int width_top,
	const float* top_grad, const float* top_out,
	const float* bottom_in, float* bottom_grad,
	const int64* map, const float threshold,
	const Eigen::GpuDevice& d);
	
bool DiscreteHoughGrad(
	const int batches, const int channels,
	const int height_bottom, const int height_top,
	const int width_bottom, const int width_top,
	const Eigen::half* top_grad, const Eigen::half* top_out,
	const Eigen::half* bottom_in, Eigen::half* bottom_grad,
	const int64* map, const float threshold,
	const Eigen::GpuDevice& d);


}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_HOUGH_OP_GPU_H

