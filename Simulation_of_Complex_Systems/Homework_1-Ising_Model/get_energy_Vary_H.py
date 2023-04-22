import numpy as np
import random
import matplotlib.pyplot as plt
from copy import copy
import pandas as pd
from getsumofneighbors import *

from scipy import optimize

from matplotlib import colors

from scipy.ndimage import convolve, generate_binary_structure

def get_energy_Vary_H(chainInit,sample,N,iterations,T,kB,J,intervalH):
    
    DF1 = {name: pd.DataFrame() for name in T}

    chaincopy = chainInit.copy()

    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    
    cmap = colors.ListedColormap(['purple',"yellow"])
    bounds=[-1,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    
    
    for k in T :
        print("T is", k)
        beta = 1/(kB*k)
        
        m = []
        
        for h in intervalH : 
            chain = chainInit
            print("h=", h)   
            # plt.imshow(chainInit)
            # plt.show()
            for j in range(0,iterations):
                
                    
               
                #Generate random 10 percent indices
                indicesX = np.array([])    
                indicesY = np.array([])
                for i in range(0,int(sample)): #Indices 10% of lattice 
                    indicesX = np.insert(indicesX,0,random.randrange(0,N)).astype(int)
                    indicesY = np.insert(indicesY,0,random.randrange(0,N)).astype(int)
                indices = np.array([indicesX,indicesY]).T.astype(int)

            
                for i in range(0,len(indices)): #Update each atome one by one
                    #print(indices[i])
                    #print(SNN[indices[i][0],indices[i][1]])
                    SNN = convolve(chain, kernel, mode='constant')  #Sum Nearest 
                    #                                                 #Neighbor Matrix
                    M = SNN[indices[i][0],indices[i][1]]
                    #M = getsumofneighbors(chain, indices[i][0], indices[i][1])
                    Eplus = -(h + J*M)
                    Eminus = +(h + J*M)
                    
                    Z = np.exp(-beta*Eplus) + np.exp(-beta*Eminus)
                    pPlus = (np.exp(-beta*Eplus))/Z
                    pMinus = (np.exp(-beta*Eminus))/Z
                    r = random.uniform(0,1)
                    
                    if r <= pPlus :
                        chain[indices[i][0],indices[i][1]] = +1

                    if pPlus < r <= pPlus+pMinus :
                        chain[indices[i][0],indices[i][1]] = -1
                    
                
                
                    # if chain.sum() == 0 :
                    #     print(chain)
                        
                    m.append((1/(N^2))*chain.sum()) 

        if k == T[0] :
            DF1[0] = m
          
        if k == T[1] :
            DF1[1] = m
            
        if k == T[2] :
            DF1[2] = m

    
    return DF1