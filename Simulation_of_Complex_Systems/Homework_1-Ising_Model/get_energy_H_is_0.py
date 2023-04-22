import numpy as np
import random
import matplotlib.pyplot as plt
from copy import copy
import pandas as pd
from matplotlib import colors

from scipy.ndimage import convolve, generate_binary_structure

def get_energy_H_is_0(chainInit,sample,N,iterations,T,kB,J,H):

    chaincopy = chainInit.copy()

    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    
    
    cmap = colors.ListedColormap(['purple',"yellow"])
    bounds=[-1,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    m = []
    n = []
    p = []
    for k in T :
        chain = chainInit.copy()
        beta = 1/(kB*k)

        for j in range(0,iterations):
            
            if j in [0,5,100,200,300,400,500,600,700,800,900]:
                print(j)
                fig = plt.figure()
                plt.imshow(chain,cmap=cmap, norm=norm)
                fig.suptitle("%1.0f" %N +" x %1.0f lattice" %N +"\nT =%1.1f,"%k +" iteration Nº %i"%j + " H =%1.1f" %H)
                plt.xlabel('x-axis', fontsize=16)
                plt.ylabel('y-axis', fontsize=16)
                save_results_to = '/Users/thiba/Desktop/Results' + str(k) + "/"
                plt.savefig(save_results_to + "image" + str(j) + ".jpeg", dpi = 300)
                #plt.show()
                plt.close(fig)
            if k == T[0] :
                m.append((1/(N^2))*chain.sum())                    
            if k == T[1] :
                n.append((1/(N^2))*chain.sum())
            if k == T[2] :
                p.append((1/(N^2))*chain.sum())
            

                    
            #Generate random 10 percent indices
            indicesX = np.array([])    
            indicesY = np.array([])
            for i in range(0,int(sample)):
                indicesX = np.insert(indicesX,0,random.randrange(0,N)).astype(int)
                indicesY = np.insert(indicesY,0,random.randrange(0,N)).astype(int)
            indices = np.array([indicesX,indicesY]).T.astype(int)

        
            for i in range(0,len(indices)):
                #print(indices[i])
                #print(SNN[indices[i][0],indices[i][1]])
                
                SNN = convolve(chain, kernel, mode='constant')  #Sum Nearest 
                                                                #Neighbor Matrix
                M = SNN[indices[i][0],indices[i][1]]
        
                Eplus = -(H + J*M)
                Eminus = +(H + J*M)
                
                Z = np.exp(-beta*Eplus) + np.exp(-beta*Eminus)
                pPlus = (np.exp(-beta*Eplus))/Z
                pMinus = (np.exp(-beta*Eminus))/Z
                r = random.uniform(0,1)
                
                if r <= pPlus :
                    chain[indices[i][0],indices[i][1]] = +1
                    #print(indices[i][0],indices[i][1],"= +1")
                if pPlus < r <= pPlus+pMinus :
                    chain[indices[i][0],indices[i][1]] = -1
                    #print(indices[i][0],indices[i][1],"= -1")
    

    
    return m,n,p