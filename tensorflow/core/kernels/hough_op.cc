

#include "tensorflow/core/kernels/hough_op.h"

#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"


#if GOOGLE_CUDA
#include "tensorflow/core/kernels/hough_op_gpu.h"
#endif  // GOOGLE_CUDA


namespace tensorflow {

// no CPU implementation so far

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;


// Hough Transform -----------------------------------------------------

template <typename T>
struct LaunchHoughTransform;

template <typename T>
class HoughTransformOp : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit HoughTransformOp(OpKernelConstruction* context)
      : OpKernel(context) {
	OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
	
	OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
	OP_REQUIRES(context, data_format == FORMAT_NHWC,
                errors::InvalidArgument("Hough Transform only supports NHWC format"));
	
	OP_REQUIRES_OK(context, context->GetAttr("out_shape", &out_img_shape);
    OP_REQUIRES(context, outShape.size() == 2,
                errors::InvalidArgument("Output image shape has to have 2 dimensions (height, width)"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& map   = context->input(1);
    
    // setup output shape
    TensorShape out_shape;
    out_shape.SlowCopyFrom(&input.shape());
    out_shape.set_dim(1, out_img_shape[0]);
    out_shape.set_dim(2, out_img_shape[1]);
    
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
	
	// Checks	
	CHECK(map.dim_size() == 3) << "Incorrect format of the Hough Map";
	// check input.shape[1] == map.shape[0]
	// check input.shape[2] == map.shape[1]
	// check out_img_shape[0] == map.shape[3]
	
    LaunchHoughTransform<Device, T>::launch(
		context, input, output, map, threshold
	);
  }

 private:
  float threshold;
  TensorFormat data_format;
  std::vector<int> out_img_shape;
};

template <typename T>
struct LaunchHoughTransform<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context,
					 const Tensor& input, Tensor* output,
					 const Tensor& map, const float threshold){
    TensorShape in_shape = input.shape();
    TensorShape out_shape = output.shape();
	
	bool status = DiscreteHough(
		in_shape.dim_size(0), in_shape.dim_size(3),
		in_shape.dim_size(1), out_shape.dim_size(1),
		in_shape.dim_size(2), out_shape.dim_size(2),
		output.flat<T>().data(), input.flat<T>.data(),
		map.flat<int64>().data(), threshold,
		context->eigen_gpu_device()
	);
	
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

template <typename T>
struct LaunchHoughTransformGrad;

template <typename T>
class HoughTransformGradOp : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit HoughTransformGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
	OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
	
	OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
	OP_REQUIRES(context, data_format == FORMAT_NHWC,
                errors::InvalidArgument("Hough Transform only supports NHWC format"));
	
	OP_REQUIRES_OK(context, context->GetAttr("out_shape", &out_img_shape);
    OP_REQUIRES(context, outShape.size() == 2,
                errors::InvalidArgument("Output image shape has to have 2 dimensions (height, width)"));
  }
  
  void Compute(OpKernelContext* context) override {
    const Tensor& input   = context->input(0);
    const Tensor& map     = context->input(1);
	const Tensor& grad_in = context->input(2);
	const Tensor& output  = context->input(3);
    
    Tensor* grad_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &grad_out));
	
    LaunchHoughTransformGrad<Device, T>::launch(
		context, grad_in, output, input, grad_out, map, threshold
	);
  }
 
 private:
  float threshold;
  TensorFormat data_format;
  std::vector<int> out_img_shape;
};

template <typename T>
struct LaunchHoughTransformGrad<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context,
					 const Tensor& grad_in, const Tensor& output,
					 const Tensor& input, Tensor* grad_out,
					 const Tensor& map, const float threshold){
    TensorShape in_shape = input.shape();
    TensorShape out_shape = output.shape();
	
	bool status = DiscreteHoughGrad(
		in_shape.dim_size(0), in_shape.dim_size(3),
		in_shape.dim_size(1), out_shape.dim_size(1),
		in_shape.dim_size(2), out_shape.dim_size(2),
		grad_in.flat<T>().data(), output.flat<T>.data(),
		input.flat<T>().data(), grad_out.flat<T>.data(),
		map.flat<int64>().data(), threshold,
		context->eigen_gpu_device()
	);
	
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
