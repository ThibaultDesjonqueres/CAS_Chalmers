import numpy as np
import random
import matplotlib.pyplot as plt
from copy import copy
import pandas as pd

from matplotlib import colors
from getsumofneighbors import *
from scipy.ndimage import convolve, generate_binary_structure

def get_energy(chainInit,sample,N,iterations,T,kB,J,H):
    
    DF1 = {name: pd.DataFrame() for name in H}
    DF2 = {name: pd.DataFrame() for name in H}
    DF3 = {name: pd.DataFrame() for name in H}

    chaincopy = chainInit.copy()

    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    
    cmap = colors.ListedColormap(['purple',"yellow"])
    bounds=[-1,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    for k in T :

        beta = 1/(kB*k)
        
        m = []
        n = []
        p = []
        q = []
        RR = []
        
        for h in H : 
            print("T=", k,"H=", h)   
            chain = chainInit.copy()
            # plt.imshow(chainInit)
            # plt.show()
            for j in range(0,iterations):
                if j in [0,5,100,200,300,400,500,600,700,800,900]:
                    #print(j)
                    fig = plt.figure()
                    plt.imshow(chain, cmap=cmap, norm=norm)
                    fig.suptitle("%1.0f" %N +" x %1.0f lattice" %N +"\nT =%1.1f,"%k +" iteration Nº %i"%j + " H =%1.1f" %h)
                    plt.xlabel('x-axis', fontsize=16)
                    plt.ylabel('y-axis', fontsize=16)
                    save_results_to = '/Users/thiba/Desktop/Results' + str(k) + "/" + str(h) + "/"
                    plt.savefig(save_results_to + str(h) + "H="  + "it=" +str(j)+ ".jpeg", dpi = 300)
                    
                    plt.close(fig)
                    
                    

                        
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
                                                                    #Neighbor Matrix
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
                        
                    if h == H[0] :
                        m.append((1/(N^2))*chain.sum())                    
                    if h == H[1] :
                        n.append((1/(N^2))*chain.sum())
                    if h == H[2] :
                        p.append((1/(N^2))*chain.sum())
                    if h == H[3] :
                        q.append((1/(N^2))*chain.sum())                    
                    if h == H[4] :
                        RR.append((1/(N^2))*chain.sum())
                        

                    
    
        if k == T[0] :
            DF1[H[0]] = m
            DF1[H[1]] = n
            DF1[H[2]] = p
            DF1[H[3]] = q
            DF1[H[4]] = RR
        
        if k == T[1] :
            DF2[H[0]] = m
            DF2[H[1]] = n
            DF2[H[2]] = p
            DF2[H[3]] = q
            DF2[H[4]] = RR
        if k == T[2] :
            DF3[H[0]] = m
            DF3[H[1]] = n
            DF3[H[2]] = p
            DF3[H[3]] = q
            DF3[H[4]] = RR 

    
    return DF1,DF2,DF3