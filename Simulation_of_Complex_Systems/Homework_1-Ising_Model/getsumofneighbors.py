import numpy as np
import random
import matplotlib.pyplot as plt
from copy import copy
import pandas as pd
from matplotlib import colors

from scipy.ndimage import convolve, generate_binary_structure


# def getsumofneighbors(matrix, i, j):
#     matrixcopy = matrix.copy()
#     region = matrixcopy[max(0, i-1) : i+2,
#                     max(0, j-1) : j+2]
#     matrixcopy[i][j]==0
    
#     return np.sum(region) - matrix[i, j] # Sum the region and subtract center


def getsumofneighbors(matrix, i, j):

    matrixcopy = matrix.copy()
    try:
        matrixcopy[i+1,j+1] = 0
        matrixcopy[i+1,j-1] = 0
        matrixcopy[i-1,j+1] = 0
        matrixcopy[i-1,j-1] = 0
        matrixcopy[i,j] = 0
        region = matrixcopy[max(0, i-1) : i+2,
                        max(0, j-1) : j+2]
        return np.sum(region) 
    except:
        
        region = matrixcopy[max(0, i-1) : i+2,
                        max(0, j-1) : j+2]
        matrixcopy[i,j] = 0
        return np.sum(region) 
         

