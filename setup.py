from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
	name="Gaussian RBF",
	description="Implements weighted sum of Gaussian Radial Basis Functions (RBF)",
	ext_modules=[
		CUDAExtension(
			name='gaussian_rbf_cuda',
			sources=['gaussian_rbf.cu'],
			extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-use_fast_math', '-DRBF_USE_ATOMIC_ADD']}
		)
	],
	py_modules=['gaussian_rbf.py'],
	cmdclass={'build_ext': BuildExtension},
)

