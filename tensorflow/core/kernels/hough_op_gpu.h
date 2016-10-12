

#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_HOUGH_OP_GPU_H
#define TENSORFLOW_CORE_KERNELS_HOUGH_OP_GPU_H

namespace tensorflow {


bool DiscreteHough(
	const float* input, float* output, const int64* map, 
	const float threshold, const Eigen::GpuDevice& d);

bool DiscreteHough(
	const Eigen::half* input, Eigen::half* output, const int64* map, 
	const float threshold, const Eigen::GpuDevice& d);


bool DiscreteHoughGrad(
	const float* input, const float* grad_in, float* grad_out,
	const int64* map, const float threshold, const Eigen::GpuDevice& d);
	
bool DiscreteHoughGrad(
	const Eigen::half* input, const Eigen::half* grad_in, Eigen::half* grad_out,
	const int64* map, const float threshold, const Eigen::GpuDevice& d);


}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_HOUGH_OP_GPU_H

