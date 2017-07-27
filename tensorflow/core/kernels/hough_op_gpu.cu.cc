


#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/util/cuda_kernel_helper.h"


// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
// https://en.wikipedia.org/wiki/Row-major_order#Address_calculation_in_general
// https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/core/kernels/resize_bilinear_op_gpu.cu.cc


namespace tensorflow {

namespace {

template <typename T>
__global__ void HoughForwardNHWC(const int nthreads, const int channels,
								 const int height_bottom, const int height_top,
								 const int width_bottom, const int width_top,
							     T* top, const T* bottom,
							     const int64* map, const float threshold) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
	if ((float) bottom[index] > threshold){
	  int batch_bottom_size = height_bottom * width_bottom * channels;
	  int row_bottom_size   = width_bottom * channels;
	  int col_bottom_size   = channels;
	  
	  int batch_top_size = height_top * width_top * channels;
	  int row_top_size   = width_top * channels;
	  int col_top_size   = channels;
	  
	  int n = index;
	  
	  // unflatten input index
	  int batch = n / batch_bottom_size;
	  n = n % batch_bottom_size;
	  int row = n / row_bottom_size;
	  n = n % row_bottom_size;
	  int col = n / col_bottom_size;
	  n = n % col_bottom_size;
	  int channel = n;
	  
	  // loop variables
	  int row_top = 0;
	  int col_top = 0;
	  int index_top = 0;
	  int bins = height_top;
	  
	  for(int i = 0; i < bins; i++){
		row_top = i;
		col_top = map[i + bins * col + bins * width_bottom * row];
		
		// flatten output index
		index_top = channel + \
			col_top_size * col_top + \
			row_top_size * row_top + \
			batch_top_size * batch;
		
		// increment atomically
		CudaAtomicAdd(top + index_top, T(1));
	  }
    }
  }
}

template <typename T>
__global__ void HoughBackwardNHWC(const int nthreads, const int channels,
								  const int height_bottom, const int height_top,
								  const int width_bottom, const int width_top,
							      const T* top_grad, const T* top_out,
							      const T* bottom_in, T* bottom_grad,
							      const int64* map, const float threshold) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
	if ((float) bottom_in[index] > threshold){
	  int batch_bottom_size = height_bottom * width_bottom * channels;
	  int row_bottom_size   = width_bottom * channels;
	  int col_bottom_size   = channels;
	  
	  int batch_top_size = height_top * width_top * channels;
	  int row_top_size   = width_top * channels;
	  int col_top_size   = channels;
	  
	  int n = index;
	  
	  // unflatten input index
	  int batch = n / batch_bottom_size;
	  n = n % batch_bottom_size;
	  int row = n / row_bottom_size;
	  n = n % row_bottom_size;
	  int col = n / col_bottom_size;
	  n = n % col_bottom_size;
	  int channel = n;
	  
	  // loop variables
	  T accum = T(0);
	  int row_top = 0;
	  int col_top = 0;
	  int index_top = 0;
	  int bins = height_top;
	  
	  for(int i = 0; i < bins; i++){
		row_top = i;
		col_top = map[i + bins * col + bins * width_bottom * row];
		
		// flatten output index
		index_top = channel + \
			col_top_size * col_top + \
			row_top_size * row_top + \
			batch_top_size * batch;
		
		
		// collect fractions of all affected upper cells
		accum += top_grad[index_top] / top_out[index_top];
	  }
	  
	  bottom_grad[index] = accum;
    }
  }
}
}  // namespace




bool DiscreteHough(
	const int batches, const int channels,
	const int height_bottom, const int height_top,
	const int width_bottom, const int width_top,
	float* top, const float* bottom,
	const int64* map, const float threshold,
	const Eigen::GpuDevice& d) {
	
  const int top_size = batches * height_top * width_top * channels;
  const int bottom_size = batches * height_bottom * width_bottom * channels;
  CudaLaunchConfig configTop    = GetCudaLaunchConfig(top_size, d);
  CudaLaunchConfig configBottom = GetCudaLaunchConfig(bottom_size, d);
  
  // zero everything
  SetZero<<<configTop.block_count, configTop.thread_per_block, 0, d.stream()>>>(
	configTop.virtual_thread_count, top
  );
  
  // do computation
  HoughForwardNHWC<<<configBottom.block_count, configBottom.thread_per_block, 0, d.stream()>>>(
	configBottom.virtual_thread_count, channels,
	height_bottom, height_top,
	width_bottom, width_top,
	top, bottom,
	map, threshold
  );
  
  return d.ok();
}

bool DiscreteHough(
	const int batches, const int channels,
	const int height_bottom, const int height_top,
	const int width_bottom, const int width_top,
	Eigen::half* top, const Eigen::half* bottom,
	const int64* map, const float threshold,
	const Eigen::GpuDevice& d) {
	
  const int top_size = batches * height_top * width_top * channels;
  const int bottom_size = batches * height_bottom * width_bottom * channels;
  CudaLaunchConfig configTop    = GetCudaLaunchConfig(top_size, d);
  CudaLaunchConfig configBottom = GetCudaLaunchConfig(bottom_size, d);
  
  // zero everything
  SetZero<<<configTop.block_count, configTop.thread_per_block, 0, d.stream()>>>(
	configTop.virtual_thread_count, top
  );
  
  // do computation
  HoughForwardNHWC<<<configBottom.block_count, configBottom.thread_per_block, 0, d.stream()>>>(
	configBottom.virtual_thread_count, channels,
	height_bottom, height_top,
	width_bottom, width_top,
	top, bottom,
	map, threshold
  );
  
  return d.ok();
}




bool DiscreteHoughGrad(
	const int batches, const int channels,
	const int height_bottom, const int height_top,
	const int width_bottom, const int width_top,
	const float* top_grad, const float* top_out,
	const float* bottom_in, float* bottom_grad,
	const int64* map, const float threshold,
	const Eigen::GpuDevice& d) {
		
  const int bottom_size = batches * height_bottom * width_bottom * channels;
  CudaLaunchConfig configBottom = GetCudaLaunchConfig(bottom_size, d);
  
  // zero everything
  SetZero<<<configBottom.block_count, configBottom.thread_per_block, 0, d.stream()>>>(
	configBottom.virtual_thread_count, bottom_grad
  );
  
  // do computation
  HoughBackwardNHWC<<<configBottom.block_count, configBottom.thread_per_block, 0, d.stream()>>>(
    configBottom.virtual_thread_count, channels,
	height_bottom, height_top,
	width_bottom, width_top,
	top_grad, top_out,
	bottom_in, bottom_grad,
	map, threshold
  );
  
  return d.ok();
}

bool DiscreteHoughGrad(
	const int batches, const int channels,
	const int height_bottom, const int height_top,
	const int width_bottom, const int width_top,
	const Eigen::half* top_grad, const Eigen::half* top_out,
	const Eigen::half* bottom_in, Eigen::half* bottom_grad,
	const int64* map, const float threshold,
	const Eigen::GpuDevice& d) {
		
  const int bottom_size = batches * height_bottom * width_bottom * channels;
  CudaLaunchConfig configBottom = GetCudaLaunchConfig(bottom_size, d);
  
  // zero everything
  SetZero<<<configBottom.block_count, configBottom.thread_per_block, 0, d.stream()>>>(
	configBottom.virtual_thread_count, bottom_grad
  );
  
  // do computation
  HoughBackwardNHWC<<<configBottom.block_count, configBottom.thread_per_block, 0, d.stream()>>>(
    configBottom.virtual_thread_count, channels,
	height_bottom, height_top,
	width_bottom, width_top,
	top_grad, top_out,
	bottom_in, bottom_grad,
	map, threshold
  );
  
  return d.ok();
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
