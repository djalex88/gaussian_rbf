"""
Weighted sum of Gaussian Radial Basis Functions (RBF)

Parameters
----------
x : tensor of shape (N, C, H, W)
	input image
weights : tensor of shape (C, num_nodes)
    weights
mu : tensor of shape (num_nodes,)
	means
sigma : tensor of dimention 0 (scalar)
	std deviation

Returns
-------
out : tensor of shape (N, C, H, W)
	output image

"""

import torch

class GaussianRBF(torch.autograd.Function):

	@staticmethod
	def forward(ctx, x, weights, mu, sigma):
		ctx.save_for_backward(x, weights, mu, sigma)
		y = torch.zeros_like(x)
		for j in range(weights.shape[1]):
			y+= torch.exp(-(x-mu[j])**2/(2*sigma**2)) * weights[:,j,None,None]
		return y

	@staticmethod
	def backward(ctx, grad_wrt_output):
		x, weights, mu, sigma = ctx.saved_tensors

		grad_x =       torch.zeros_like(x)       if ctx.needs_input_grad[0] else None
		grad_weights = torch.empty_like(weights) if ctx.needs_input_grad[1] else None

		for j in range(weights.shape[1]):
			a = (x-mu[j])/sigma
			b = torch.exp(-0.5*a**2)
			if ctx.needs_input_grad[0]:
				grad_x-= a * b * weights[:,j,None,None] * (1/sigma)
			if ctx.needs_input_grad[1]:
				grad_weights[:,j] = ( b * grad_wrt_output ).sum(dim=(0,2,3))
		if ctx.needs_input_grad[0]:
			grad_x*= grad_wrt_output

		return grad_x, grad_weights, None, None


try:
	import gaussian_rbf_cuda

except:
	print("Warning: CUDA implementation of Gaussian RBF is not available! Falling back to Python implementation.")
	GaussianRBF_CUDA = GaussianRBF

else:
	def bins(N, k):
		num_bins = N//k+1 if N%k else N//k
		x = N // num_bins
		for i in range(N % num_bins):
			yield x+1
		for i in range(N % num_bins, num_bins):
			yield x

	MAX_NODES = gaussian_rbf_cuda.MAX_NODES

	class GaussianRBF_CUDA(torch.autograd.Function):

		@staticmethod
		def forward(ctx, x, weights, mu, sigma):
			ctx.save_for_backward(x, weights, mu, sigma)
			num_nodes, = mu.shape
			if num_nodes > MAX_NODES:
				i, *v = bins(num_nodes, MAX_NODES)
				out = gaussian_rbf_cuda.forward(x, weights[:,:i].contiguous(), mu[:i], sigma)
				for k in v:
					out+= gaussian_rbf_cuda.forward(x, weights[:,i:i+k].contiguous(), mu[i:i+k], sigma)
					i+= k
			else:
				out = gaussian_rbf_cuda.forward(x, weights, mu, sigma)
			return out

		@staticmethod
		def backward(ctx, grad_wrt_output):
			x, weights, mu, sigma = ctx.saved_tensors
			num_nodes, = mu.shape
			if num_nodes > MAX_NODES:
				i, *v = bins(num_nodes, MAX_NODES)
				grad_x, dw = gaussian_rbf_cuda.backward(x, weights[:,:i].contiguous(), mu[:i], sigma, grad_wrt_output)
				grad_weights = [dw]
				for k in v:
					dx, dw = gaussian_rbf_cuda.backward(x, weights[:,i:i+k].contiguous(), mu[i:i+k], sigma, grad_wrt_output)
					grad_x+= dx ; grad_weights.append(dw)
					i+= k
				grad_weights = torch.cat(grad_weights, dim=1)
			else:
				grad_x, grad_weights = gaussian_rbf_cuda.backward(x, weights, mu, sigma, grad_wrt_output)
			return grad_x, grad_weights, None, None

