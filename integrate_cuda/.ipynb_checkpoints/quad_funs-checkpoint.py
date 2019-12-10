### import libraries
import numpy  as np
import math
from   numba  import guvectorize
from   numba  import float32, float64
from  .quad3d import quad3d

@guvectorize([(float32[:,:], float32[:]),
              (float64[:,:], float64[:])], 
              '(n,m)->(n)', target="cuda")
def row_sum(array, out):
    """each row of array is a sequence of nodes from Gauss Quadrature,
    sum the value nodes in each row to get the results of integration
    each row represent the integration of a function over the domain"""
    nr = array.shape[0]
    nc = array.shape[1]
    
    ### sum of each row
    for idx_r in range(nr):
        out[idx_r] = 0
        for idx_c in range(nc):
            out[idx_r] += array[idx_r, idx_c] 
    

def quad_funs(idx_funs, roots, weights, ranges, values, threadsperblock = (32, 32), dim="3d"):
    """Wrapper function for help setting up the integration"""
    ### initialization
    n_fun  = idx_funs.shape[0] # number of functions you want to integrate
    n_node = roots.shape[0]    # number of nodes in Gauss Quadrature
    
    ### set the blocks of threads based on the given number of threads per block
    blockspergrid_x = math.ceil(n_fun     / threadsperblock[0]) 
    blockspergrid_y = math.ceil(n_node**3 / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    ### perform integration
    if dim=="3d":
        quad3d[blockspergrid, threadsperblock](idx_funs, roots, weights, ranges, values)
    
    ### the results of integration for each function is acquired 
    ### by summing the results of nodes for each row
    return row_sum(values) 