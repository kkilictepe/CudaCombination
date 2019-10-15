"""
Created on Sat Sep 28 00:47:38 2019
@author: Kenan Kilictepe (Entegral / www.entegral.com.tr)
"""

import numba
from numba import cuda
import numpy as np
from pdb import set_trace


@cuda.jit(device=True)
def choose(n, k):
    if n < k:
        return 0
    if n == k:
        return 1

    delta = imax = 0
    if k < n-k:
        delta = n-k
        imax = k
    else:
        delta = k
        imax = n-k

    ans = numba.int64(delta + 1)

    for i in range(2, imax+1):
        ans = numba.int64((ans * (delta + i)) // i)

    return ans


@cuda.jit(device=True)
def largestV(a, b, x):

    v = numba.int64(a-1)
    while choose(v, b) > x:
        v -= 1

    return v


@cuda.jit
def cuda_calculateMth(n, k, d_result):
    pos = cuda.grid(1)     # pylint: disable=not-callable
    if pos >= len(d_result):
        return

    m = pos
    a = n
    b = k
    x = (choose(a, b) - 1) - m

    for i in range(k):
        d_result[pos][i] = largestV(a, b, x)
       
        x = x - choose(d_result[pos][i], b)
        a = d_result[pos][i]
        b -= 1
      
    for i in range(k):
        d_result[m][i] = (n-1) - d_result[m][i]
    
    




if __name__ == "__main__":

    n = 100
    k = 3

    totalcount = 1
    factorial = 1
    for i in range(k):
        totalcount *= (n-i)
        factorial *= (i+1)

    totalcount = totalcount // factorial

    result = np.zeros((totalcount, k), dtype="uint")
    temp = np.zeros(10, dtype="uint")

    d_result = cuda.to_device(result)

    threadsperblock = 512
    blockspergrid = (totalcount +
                     (threadsperblock - 1)) // threadsperblock

    cuda_calculateMth[blockspergrid, threadsperblock](  # pylint: disable=unsubscriptable-object
        n, k, d_result)
    result = d_result.copy_to_host()
    print(result)
