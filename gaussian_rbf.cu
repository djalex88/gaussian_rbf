#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MAX_NODES WARP_SIZE
#define NUM_THREADS_FW 256

// using CUDA's atomicAdd() is slightly faster and is memory efficient
// however, atomicAdd() is non-deterministic!

__global__ void
	gaussian_rbf_cuda_forward(
		int num_images,
		int num_channels,
		int image_size,
		float *data_in,
		float *data_out,
		int num_nodes,
		float *weights,
		float *mu,
		float sigma
		)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	int c = blockIdx.y;

	__shared__ float w_s[MAX_NODES];
	__shared__ float m_s[MAX_NODES];

	if(threadIdx.x < num_nodes){
		w_s[threadIdx.x] = weights[c*num_nodes + threadIdx.x];
		m_s[threadIdx.x] = mu[threadIdx.x];
	}
	__syncthreads();

	if(j >= image_size) return;

	float r_2sqr_sigma = 0.5/(sigma*sigma);

	for(int imageIdx=0; imageIdx<num_images; ++imageIdx){
		int dataIdx = (imageIdx*num_channels*image_size) + (c*image_size) + j;
		float s = 0;
		float x = data_in[dataIdx];
		for(int k=0; k<num_nodes; ++k){
			float t = x - m_s[k];
			s+= expf(-t*t*r_2sqr_sigma) * w_s[k];
		}
		data_out[dataIdx] = s;
	}
}


__global__ void
	gaussian_rbf_cuda_backward(
		int num_images,
		int num_channels,
		int image_size,
		float *data_in,
		float *grad_wrt_data_out,
		float *grad_wrt_data_in,
		float *grad_wrt_weights,
		int num_nodes,
		float *weights,
		float *mu,
		float sigma
		)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	int c = blockIdx.y;

	__shared__ float w_s[MAX_NODES];
	__shared__ float m_s[MAX_NODES];

	// buffer for grad wrt weights
	__shared__ float grad_w_s[MAX_NODES][WARP_SIZE];
	// initialize to zero
	for(int k=0; k<num_nodes; ++k){
		grad_w_s[k][threadIdx.x] = 0;
	}

	float r_sigma = 1/sigma;

	if(threadIdx.x < num_nodes){
		w_s[threadIdx.x] = weights[c*num_nodes + threadIdx.x] * r_sigma; // (*)
		m_s[threadIdx.x] = mu[threadIdx.x];
	}
	__syncwarp();

	if(j < image_size)
	{
		for(int imageIdx=0; imageIdx<num_images; ++imageIdx){
			int dataIdx = (imageIdx*num_channels*image_size) + (c*image_size) + j;
			float s = 0;
			float x = data_in[dataIdx];
			float grad = grad_wrt_data_out[dataIdx];
			for(int k=0; k<num_nodes; ++k){
				float a = (x - m_s[k]) * r_sigma;
				float b = expf(-0.5f*a*a);
				s-= a * b * w_s[k]; // see (*)
				grad_w_s[k][threadIdx.x]+= b * grad;
			}
			grad_wrt_data_in[dataIdx] = s * grad;
		}
	}
	__syncwarp();

	// sum up
#ifdef RBF_USE_ATOMIC_ADD
	if(threadIdx.x < num_nodes){
		float s = 0;
		for(int i=0; i<WARP_SIZE; ++i){
			s+= grad_w_s[threadIdx.x][i];
		}
		atomicAdd(grad_wrt_weights + c*num_nodes + threadIdx.x, s);
	}
#else
	for(int k=0; k<num_nodes; ++k)
	{
#pragma unroll
		for(int offset=WARP_SIZE>>1; offset>1; offset=offset>>1){
			__syncwarp();
			if(threadIdx.x < offset){
				grad_w_s[k][threadIdx.x]+= grad_w_s[k][threadIdx.x + offset];
			}
		}
		__syncwarp();
		if(threadIdx.x == 0){
			int gradWrtWeightsIdx = (c*num_nodes*gridDim.x) + (k*gridDim.x) + blockIdx.x;
			grad_wrt_weights[gradWrtWeightsIdx] = grad_w_s[k][0] + grad_w_s[k][1];
		}
	}
#endif
}


#include <torch/extension.h>
#include <vector>
#include <iostream>

#define STR(x) #x
#define VAL(x) STR(x)
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x) TORCH_CHECK(x.dtype() == torch::kF32, #x " must be a float32 tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_TYPE(x)


torch::Tensor
	gaussian_rbf_forward(
		torch::Tensor x,
		torch::Tensor weights,
		torch::Tensor mu,
		     float sigma ) {

	CHECK_INPUT(x);
	CHECK_INPUT(weights);
	CHECK_INPUT(mu);

	int64_t device = x.get_device();

	TORCH_CHECK(weights.get_device() == device && mu.get_device() == device, "input tensors must reside on the same device");

	cudaSetDevice((int) device);

	int num_images = x.size(0);
	int num_channels = x.size(1);
	int image_size = x.size(2) * x.size(3);
	int num_nodes = mu.size(0);

	TORCH_CHECK(weights.size(0) == num_channels, "x and weights have different number of channels");
	TORCH_CHECK(weights.size(1) == num_nodes, "weights and mu have different number of nodes");
	TORCH_CHECK(num_nodes <= MAX_NODES, "number of nodes > " VAL(MAX_NODES));

	int num_blocks = image_size / NUM_THREADS_FW + (image_size % NUM_THREADS_FW ? 1 : 0);

	torch::Tensor y = torch::empty_like(x);

	// run forward kernel
	dim3 blocks(num_blocks, num_channels);
	dim3 threads(NUM_THREADS_FW, 1);
	gaussian_rbf_cuda_forward <<< blocks , threads >>> (
			num_images,
			num_channels,
			image_size,
			x.data_ptr<float>(),
			y.data_ptr<float>(),
			num_nodes, weights.data_ptr<float>(), mu.data_ptr<float>(), sigma
		);

	return y;
}


std::vector<torch::Tensor>
	gaussian_rbf_backward(
		torch::Tensor x,
		torch::Tensor weights,
		torch::Tensor mu,
		     float sigma,
		torch::Tensor grad_wrt_output ) {

	CHECK_INPUT(x);
	CHECK_INPUT(weights);
	CHECK_INPUT(mu);
	CHECK_INPUT(grad_wrt_output);

	int64_t device = x.get_device();

	TORCH_CHECK(weights.get_device() == device && mu.get_device() == device && \
		grad_wrt_output.get_device() == device, "input tensors must reside on the same device");

	cudaSetDevice((int) device);

	static int deviceWarpSizes[] = {0, 0, 0, 0, 0, 0, 0, 0};
	if(!deviceWarpSizes[(int) device]){
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, (int) device);
		TORCH_CHECK(prop.warpSize == WARP_SIZE, "prop.warpSize != " VAL(WARP_SIZE));
		deviceWarpSizes[(int) device] = WARP_SIZE;
	}

	int num_images = x.size(0);
	int num_channels = x.size(1);
	int image_size = x.size(2) * x.size(3);
	int num_nodes = mu.size(0);

	TORCH_CHECK(weights.size(0) == num_channels, "x and weights have different number of channels");
	TORCH_CHECK(weights.size(1) == num_nodes, "weights and mu have different number of nodes");
	TORCH_CHECK(grad_wrt_output.sizes() == x.sizes(), "gradient wrt output has different shape");
	TORCH_CHECK(num_nodes <= MAX_NODES, "number of nodes > " VAL(MAX_NODES));

	int num_blocks = image_size / WARP_SIZE + (image_size % WARP_SIZE ? 1 : 0);

	torch::Tensor grad_x = torch::empty_like(x);
#ifdef RBF_USE_ATOMIC_ADD
	torch::Tensor grad_weights = torch::zeros_like(weights);
#else
	torch::Tensor grad_weights = torch::empty({num_channels, num_nodes, num_blocks}, weights.options());
#endif

	// run backward kernel
	dim3 blocks(num_blocks, num_channels);
	dim3 threads(WARP_SIZE, 1);
	gaussian_rbf_cuda_backward <<< blocks , threads >>> (
			num_images,
			num_channels,
			image_size,
			x.data_ptr<float>(),
			grad_wrt_output.data_ptr<float>(),
			grad_x.data_ptr<float>(),
			grad_weights.data_ptr<float>(),
			num_nodes, weights.data_ptr<float>(), mu.data_ptr<float>(), sigma
		);

#ifdef RBF_USE_ATOMIC_ADD
	return { grad_x, grad_weights };
#else
	return { grad_x, grad_weights.sum(-1) };
#endif
}

// export
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.attr(STR(MAX_NODES)) = MAX_NODES;
	m.def("forward", &gaussian_rbf_forward, "Gaussian RBF forward");
	m.def("backward", &gaussian_rbf_backward, "Gaussian RBF backward wtr x and weights");
}
