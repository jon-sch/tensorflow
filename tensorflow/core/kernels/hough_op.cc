


#if GOOGLE_CUDA
#include "tensorflow/core/kernels/hough_op_gpu.h"
#endif  // GOOGLE_CUDA


namespace tensorflow {

// no CPU implementation so far

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;


// Hough Transform -----------------------------------------------------

template <typename Device, typename T>
struct LaunchHoughTransform;

template <typename Device, typename T>
class HoughTransformOp : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit HoughTransformOp(OpKernelConstruction* context)
      : OpKernel(context) {
	//...
	// get map as attribute?
	// get batch size
	// get row size
	// get col size
	// get channel size
	// check NHWC dataformat
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
	
	// allocate output tensor
	
	// set output tensor to 0
	// there is an existing function somewhere implemented for that
	
    LaunchHoughTransform<Device, T>::launch( /* ... */);
  }

 private:
  // forward tensor map
  // output shape
};

template <typename T>
struct LaunchHoughTransform<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context,
					 const Tensor& input, Tensor* output,
					 const Tensor& map, const float threshold){
    bool status = DiscreteHough(
	  input.flat<T>().data(), output.flat<T>().data(),
	  map.flat<T>().data(), threshold, context->eigen_gpu_device());
	
	// missing array information, depth, width, height, channels
	
	if (!status) {
	  context->SetStatus(
	    errors::Internal("Failed launching DiscreteHough"));
	}
  }
};


REGISTER_KERNEL_BUILDER(
    Name("HoughTransform")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T")
        .TypeConstraint<int64>("Tmap"),
    HoughTransformOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("HoughTransform")
        .Device(DEVICE_GPU)
        .TypeConstraint<Eigen::half>("T")
        .TypeConstraint<int64>("Tmap"),
    HoughTransformOp<Eigen::GpuDevice, Eigen::half>);





// Hough Transform Gradient --------------------------------------------

template <typename Device, typename T>
struct LaunchHoughTransformGrad;

template <typename Device, typename T>
class HoughTransformGradOp : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit HoughTransformGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
	//...
  }
  
  void Compute(OpKernelContext* context) override {
	  
	  
    LaunchHoughTransformGrad<Device, T>::launch( /* ... */);
  }
 
 private:
  // 
  //
};

template <typename T>
struct LaunchHoughTransformGrad<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context,
					 const Tensor& input, const Tensor& grad_in, Tensor* grad_out,
					 const Tensor& map, const float threshold){
    bool status = DiscreteHoughGrad(
      input.flat<T>().data(), grad_in.flat<T>.data(), grad_out.flat<T>.data(),
      map.flat<T>().data(), threshold, context->eigen_gpu_device());
	
	// missing array information, depth, width, height, channels
	
	if (!status) {
	  context->SetStatus(
	    errors::Internal("Failed launching DiscreteHoughGrad"));
	}
  }
};


REGISTER_KERNEL_BUILDER(
    Name("HoughTransformGrad")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T")
        .TypeConstraint<int64>("Tmap"),
    HoughTransformOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("HoughTransformGrad")
        .Device(DEVICE_GPU)
        .TypeConstraint<Eigen::half>("T")
        .TypeConstraint<int64>("Tmap"),
    HoughTransformOp<Eigen::GpuDevice, Eigen::half>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
