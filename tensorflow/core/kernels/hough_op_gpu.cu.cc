

#include "tensorflow/core/util/cuda_kernel_helper.h"


// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
// https://en.wikipedia.org/wiki/Row-major_order#Address_calculation_in_general

namespace tensorflow {

namespace {

template <typename dtype>
__global__ void HoughForwardNHWC(const int nthreads,
							     const dtype* top, dtype* bottom,
							     const int64* map, const float threshold) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
	// missing:
	// height_bottom, height_top
	// width_bottom, width_top
	// channels
	
	if (bottom[index] > threshold){
	  int batch_bottom_size = height_bottom * width_bottom * channels;
	  int row_bottom_size   = width_bottom * channels;
	  int col_bottom_size   = channels;
	  
	  int batch_top_size = height_top * width_top * channels;
	  int row_top_size   = width_top * channels;
	  int col_top_size   = channels;
	  
	  int n = index;
	  
	  // unflatten input index
	  int batch = n / batch_size;
	  n = n % batch_size;
	  int row = n / row_size;
	  n = n % row_size;
	  int col = n / col_size;
	  n = n % col_size;
	  int channel = n;
	  
	  // loop variables
	  int row_top = 0;
	  int col_top = 0;
	  int index_top = 0;
	  
	  for(int i = 0; i < bins; i++){
		row_top = i;
		col_top = map[i + bins * col + bins * width * row];
		
		// flatten output index
		index_top = channel + \
					col_top_size * col_top + \
					row_top_size * row_top + \
					batch_top_size * batch;
		
		// increment atomically
		atomicAdd(top + index_top, 1.0);
	  }
    }
  }
}

template <typename dtype>
__global__ void HoughBackwardNHWC(const int nthreads,
							      const dtype* top_grad, const dtype* top_out, dtype* bottom,
							      const int64* map, const float threshold) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // missing:
	// height_bottom, height_top
	// width_bottom, width_top
	// channels
	
	if (bottom[index] > threshold){
	  int batch_bottom_size = height_bottom * width_bottom * channels;
	  int row_bottom_size   = width_bottom * channels;
	  int col_bottom_size   = channels;
	  
	  int batch_top_size = height_top * width_top * channels;
	  int row_top_size   = width_top * channels;
	  int col_top_size   = channels;
	  
	  int n = index;
	  
	  // unflatten input index
	  int batch = n / batch_size;
	  n = n % batch_size;
	  int row = n / row_size;
	  n = n % row_size;
	  int col = n / col_size;
	  n = n % col_size;
	  int channel = n;
	  
	  // loop variables
	  dtype accum = 0.0;
	  int row_top = 0;
	  int col_top = 0;
	  int index_top = 0;
	  
	  for(int i = 0; i < bins; i++){
		row_top = i;
		col_top = map[i + bins * col + bins * width * row];
		
		// flatten output index
		index_top = channel + \
					col_top_size * col_top + \
					row_top_size * row_top + \
					batch_top_size * batch;
		
		
	    // collect fractions of all affected upper cells
		accum += top_grad[index_top] / top_out[index_top];
	  }
	  
	  bottom[index] = accum;
    }
  }
}

//? #undef CUDA_1D_KERNEL_LOOP
}  // namespace



bool DiscreteHough(
	const float* input, float* output, const int64* map, 
	const float threshold, const Eigen::GpuDevice& d) {
	
  const int input_size = batch * channels * height * width;
		
		
  const int kThreadsPerBlock = 1024;
  const int output_size; // work pixel by pixel, iterate through batch and channel in kernel
  
  // zero everything
  SetZero<<<(bottom_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
            kThreadsPerBlock, 0, d.stream()>>>(bottom_size, bottom_diff);
  
  // do computation
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
  
  // zero everything
  SetZero<<<(bottom_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
            kThreadsPerBlock, 0, d.stream()>>>(bottom_size, bottom_diff);
  
  HoughForwardNHWC<<<(/* blocks? */, /* threads per block? */, /* ? */, d.stream()>>>(
  /* ... */
  );
  return d.ok();
}

// duplicate with Eigen::half

}  // namespace tensorflow
