### import libraries
import numpy  as np
from   numba  import cuda

### import the configuration (config.py in the parent folder)
import sys, os
sys.path.append(os.getcwd() + '/..')
from config import ing3d_wrapper

@cuda.jit
def quad3d(idx_funs, roots, weights, ranges, values):
    """The kernel function that perform triple integrations
    The function is CUDA parallelize for the functions and nodes from Gauss quadrature"""
    ik, iv = cuda.grid(2)
    if (ik < values.shape[0]) & (iv < values.shape[1]):
        
        ### specify the index of function and the ranges of the domain
        idx_fun = idx_funs[ik]
        aranges = ranges[ik]
        
        ### extract the ranges of domain it is going to integrate
        xa = aranges[0]; xb = aranges[1] # x axis
        ya = aranges[2]; yb = aranges[3] # y axis
        za = aranges[4]; zb = aranges[5] # z axis
        
        ### Gauss Quadrature for setting nodes of custom domain of integration
        xh = (xb-xa) / 2; xc = (xb+xa) / 2
        yh = (yb-ya) / 2; yc = (yb+ya) / 2
        zh = (zb-za) / 2; zc = (zb+za) / 2
        
        ### get the 3D index from 1D array
        n_node = len(roots)
        ix = iv // (n_node*n_node)
        i  = iv %  (n_node*n_node)
        iy = i  // (n_node)
        iz = i  %  (n_node)
        
        ### set up the nodes for Gauss Quadrature
        xnode   = roots[ix] * xh + xc; xweight = weights[ix]
        ynode   = roots[iy] * yh + yc; yweight = weights[iy]
        znode   = roots[iz] * zh + zc; zweight = weights[iz]
        
        ### apply the function you want to integrate to each node
        values[ik, iv] = xh * yh * zh * xweight * yweight * zweight * ing3d_wrapper(idx_fun, xnode, ynode, znode)