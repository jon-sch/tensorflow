

// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd

namespace tensorflow {

namespace {

template <typename dtype>
__global__ void HoughForwardNHWC(const int nthreads,
							     const dtype* input, dtype* output,
							     const dtype* map, const float threshold) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
	  
  }
}

template <typename dtype>
__global__ void HoughBackwardNHWC(const int nthreads,
							      const dtype* input, dtype* output,
							      const dtype* map, const float threshold) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
	  
  }
}

//? #undef CUDA_1D_KERNEL_LOOP
}  // namespace



bool DiscreteHough(
	const float* input, float* output, const int64* map, 
	const float threshold, const Eigen::GpuDevice& d) {
  const int kThreadsPerBlock = 1024;
  const int output_size; // work pixel by pixel, iterate through batch and channel in kernel
  
  HoughForwardNHWC<<<(/* blocks? */, /* threads per block? */, /* ? */, d.stream()>>>(
  /* ... */
  );
  return d.ok();
}

// duplicate with Eigen::half


bool DiscreteHoughGrad(
	const float* input, const float* grad_in, float* grad_out,
	const int64* map, const float threshold, const Eigen::GpuDevice& d) {
  const int kThreadsPerBlock = 1024;
  const int output_size; // work pixel by pixel, iterate through batch and channel in kernel
  
  HoughForwardNHWC<<<(/* blocks? */, /* threads per block? */, /* ? */, d.stream()>>>(
  /* ... */
  );
  return d.ok();
}

// duplicate with Eigen::half

}  // namespace tensorflow
