### import libraries
import numpy as np
import math
from   numba import cuda

### set up your own functions that you want to integrate
@cuda.jit(device=True)
def ing3d_fun1(x, y, z):
    return 1

@cuda.jit(device=True)
def ing3d_fun2(x, y, z):
    return 2

@cuda.jit(device=True)
def ing3d_fun3(x, y, z):
    return x * y * z

@cuda.jit(device=True)
def ing3d_fun4(x, y, z):
    return math.cos(x) + math.sin(y) + math.cos(z) + math.cos(x) * math.exp(y) * math.pow(z,2)

@cuda.jit(device=True)
def ing3d_fun5(x, y, z):
    return math.lgamma(x * y * z)

### set up a wrapper to wrap up all the functions above
@cuda.jit(device=True)
def ing3d_wrapper(idx_fun, x, y, z):
    """This is a wrapper function that apply the function based on the index you specify; 
    the rules of mapping between numeric index and the device funtion is not fixed and 
    can be set by yourself"""
    if idx_fun == 1:
        return ing3d_fun1(x, y, z)
    if idx_fun == 2:
        return ing3d_fun2(x, y, z)
    if idx_fun == 3:
        return ing3d_fun3(x, y, z)
    if idx_fun == 4:
        return ing3d_fun4(x, y, z)
    if idx_fun == 5:
        return ing3d_fun5(x, y, z)
    else:
        return 0