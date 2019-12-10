### import libraries
import numpy as np
from numpy.polynomial import legendre
from numba import cuda
from integrate_cuda import quad_funs

### get the roots and weights for Gauss Quadrature
### from Legendre function
order = 10
roots, weights = legendre.leggauss(order)

### after setting up the functions you want to integrate in config.py
### set up the ranges for each function you want to integrate
### in this example, we have five functions in the config

# Constant function
# - f(x,y,z) = 1
# - f(x,y,z) = 2

# Polynomial function
# - f(x,y,z) = x * y * z

# Trignometric and exponential function
# - f(x,y,z) = cos(x) + sin(y) + cos(z) + cos(x) * exp(y) * z^2

# Log Gamma function
# - f(x,y,z) = log(Gamma(x * y * z))

### set the integrating order of the five functions above
idx_funs = np.arange(5) + 1
idx_funs = np.tile(idx_funs, 2)
n_funs   = len(idx_funs)

### set ranges of domain for integration
range1 = [-1, 1, -1, 1, -1,   1]
range2 = [-1, 2, -2, 3,  1,   2]
range3 = [ 1, 2,  1, 2,  1,   2]
range4 = [ 1, 5,  2, 3,  0.5, 1]
ranges = np.r_[
    np.tile(range1, 4), range3,
    np.tile(range2, 4), range4,
    ].reshape(-1, 6)

### memory management: 
### copying the array and perform memory allocation in CUDA
roots_device    = cuda.to_device(roots)
weights_device  = cuda.to_device(weights)
idx_funs_device = cuda.to_device(idx_funs)
ranges_device   = cuda.to_device(ranges)

### memory allocation for the nodes of Gauss Quadrature for all functions 
n_nodes       = order
values        = np.zeros((n_funs, n_nodes * n_nodes * n_nodes))
values_device = cuda.to_device(values)

### Perform triple integration with Gauss Quadrature
res_gpu = quad_funs(idx_funs_device, roots_device, weights_device, ranges_device, values_device)

### Move the results from GPU back to CPU
res_cpu = res_gpu.copy_to_host()

#####################################################################################
# Integrating the example functions in this example and compare to the correct answers
# Calculating the answers by hand/wolfram alpha
#  8.0
# 16.0
#  0.0
# 16.1006
#  1.2871
# 15.0
# 30.0
#  5.625
# 92.9919
#  9.60443
ans = np.array([8.0, 16.0, 0.0, 16.1006, 1.2871, 15.0, 30.0, 5.625, 92.9919, 9.60443])

### compare the results to the answer in our example
### If succeed, should get:
### >>> Are the results correct: [ True  True  True  True  True  True  True  True  True  True]
print("Are the results correct:", np.isclose(res_cpu, ans))