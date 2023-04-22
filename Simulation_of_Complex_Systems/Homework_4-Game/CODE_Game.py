import numpy as np 
from numpy import arctan2 as atan2, sin, cos
import matplotlib.pyplot as plt
from IPython import display
from scipy.constants import Boltzmann as kB 
from tkinter import *
from tkinter import ttk
from PIL import ImageGrab
from PIL import Image
from PIL import ImageTk as itk
import time
import pprint
from copy import copy
from sklearn.neighbors import NearestNeighbors
import random
from scipy.ndimage import convolve, generate_binary_structure
from IPython import display
import time
import collections
import pandas as pd1
import statistics as st
import pandas as pd
#%%
N = 7
T=0
R=0.9
P=1
S=1.5

def action(n,m,N):
    nn = np.random.randint(1, 2, size=n)
    nprime = np.random.randint(1, 2, size=N-n)-2
    NN = np.concatenate((nn, nprime))
    
    mm = np.random.randint(1, 2, size=m)
    mprime = np.random.randint(1, 2, size=N-m)-2
    MM = np.concatenate((mm, mprime))
    res=NN,MM
    return res

def actionMultiRandom(L,N): #LxL number of players (L=lattice size)
    strategy = np.random.randint(0, N, size=L*L)
    # print("strategy=", strategy)
    players=[]
    for i in strategy:
        nn = np.random.randint(1, 2, size=i)
        nprime = np.random.randint(1, 2, size=N-i)-2
        NN = np.concatenate((nn, nprime))
        players.append(NN)
    return players

def actionMultiNice(L,N): #LxL number of players (L=lattice size)
    strategy = np.random.randint(1, 2, size=L*L)
    # print("strategy=", strategy)
    players=[]
    for i in strategy:
        nn = np.random.randint(1, 2, size=i)
        nprime = np.random.randint(1, 2, size=N-i)
        NN = np.concatenate((nn, nprime))
        players.append(NN)
    return players


def game(n,m,R,S,T,P,N):
    yearsA = 0
    for i in range(0,N):
        if n == 7 and m==7 :
            yearsA += R
            continue
        if n == 7 and m==1 :
            yearsA += S  
            # n==1
            continue
        if n == 1 and m==7 :
            yearsA += T
            # m==1
            continue
        if n == 1 and m==1 :
            yearsA += P
            continue
    return yearsA

def pd(n1,n2,R,S,T,P,N):                 
    # Prisoner's dilemma with two agents with strategies n1 and n2
    r = min(n1,n2)
    if n1<n2:
        p1 = r*R +(N-1-r)
        p2 = r*R + S + (N-1-r)
    elif n1 == n2:
        p1 = r*R + (N-r)
        p2 = p1
    else:
        p1 = r*R + S + (N-1-r)
        p2 = r*R + (N-1-r)
    return p1


def pd2(n,m,R,S,T,P,N):                 
    # Prisoner's dilemma with two agents with strategies n1 and n2
    yearsA = []
    # yearsB = []
    # print(n,m,A)
    # print(n,m,B)

    A = action(n,m,N)[0]
    # print("A",A)
    B = action(n,m,N)[1]
        
    for i in range(0,N):

        if (A[i] == 1) and (B[i] == 1) :
            yearsA.append(R)
            # yearsB.append(R)
            # print("R",n,i,A[i],B[i])
            continue
            
            
        if (A[i] == 1) and (B[i] == -1) :
            yearsA.append(S)
            # yearsB.append(T)
            # print("S",n,i,A[i],B[i], "Betrayed")
            A = action(n,m,N)[1]
            continue
            
            
        if (A[i] == -1) and (B[i] == 1) :
            yearsA.append(T)
            # yearsB.append(S)
            # print("T",n,i,A[i],B[i])
            B = action(n,m,N)[0]
            continue
    
        if (A[i] == -1) and (B[i] == -1) :
            yearsA.append(P)
            # yearsB.append(P)
            # print("P",n,i,A[i],B[i])
            continue
    return np.array(yearsA).sum()
#%% 13.1.a
N = 10
T=0
R=0.5
P=1
S=1.5


m=6
n=8

totA = []
totB=[]


for n in range(0,N+1):
    
    yearsA = []
    yearsB =[]
    # print(n,m,A)
    # print(n,m,B)

    A = action(n,m,N)[0]
    B = action(n,m,N)[1]
    
    for i in range(0,N):

        if (A[i] == 1) and (B[i] == 1) :
            yearsA.append(R)
            yearsB.append(R)
            print("R",n,i,A[i],B[i])
            continue
            
            
        if (A[i] == 1) and (B[i] == -1) :
            yearsA.append(S)
            yearsB.append(T)
            print("S",n,i,A[i],B[i], "Betrayed")
            A = action(n,m,N)[1]
            continue
            
            
        if (A[i] == -1) and (B[i] == 1) :
            yearsA.append(T)
            yearsB.append(S)
            print("T",n,i,A[i],B[i])
            B = action(n,m,N)[0]
            continue
    
        if (A[i] == -1) and (B[i] == -1) :
            yearsA.append(P)
            yearsB.append(P)
            print("P",n,i,A[i],B[i])
            continue
        # print(yearsA)
    
    #print(np.sum(yearsA), np.sum(yearsB))
    print(yearsA)
    totA.append(np.sum(yearsA))
    totB.append(np.sum(yearsB))
nn = [i for i in range(0,N+1)]

plt.scatter(nn,totA)
plt.title("yearsInPrison Vs. n")
plt.xlabel("n")
plt.ylabel("Years in Prison")        

        
#%% 13.1.b  
N = 10
T=0
R=0.5
P=1
S=1.5
totA = []
totB=[]
index=[]
Map = np.zeros((11,11))
for m in range(0,N+1):
    for n in range(0,N+1):      
        yearsA = []
        yearsB =[]
        # print(n,m,A)
        # print(n,m,B)
        A = action(n,m,N)[0]
        B = action(n,m,N)[1]
        
        for i in range(0,N):
    
            if (A[i] == 1) and (B[i] == 1) :
                yearsA.append(R)
                yearsB.append(R)
                # print("R",n,i,A[i],B[i])
                continue
            if (A[i] == 1) and (B[i] == -1) :
                yearsA.append(S)
                yearsB.append(T)
                # print("S",n,i,A[i],B[i], "B Betrayed A")
                A = action(n,m,N)[1]
                continue
            if (A[i] == -1) and (B[i] == 1) :
                yearsA.append(T)
                yearsB.append(S)
                # print("T",n,i,A[i],B[i], "A Betrayed B")
                B = action(n,m,N)[0]
                continue  
            if (A[i] == -1) and (B[i] == -1) :
                yearsA.append(P)
                yearsB.append(P)
                # print("P",n,i,A[i],B[i])
                continue  
        totA.append(np.sum(yearsA))
        Map[n][m] = (np.sum(yearsA))
nn = [i for i in range(0,N+1)]
plt.imshow(Map,origin='lower')
plt.title("years in prison wrt. m,n")
plt.xlabel("m")
plt.ylabel("n")
plt.colorbar()

#%% 13.c  #Play R and S
N = 10
T=0
R=0.1  # R={0,1}
P=1
S=1 #R={1,2}
totA = []
totB=[]
index=[]
Map = np.zeros((11,11))
for m in range(0,N+1):
    for n in range(0,N+1):      
        yearsA = []
        yearsB =[]
        # print(n,m,A)
        # print(n,m,B)
        A = action(n,m,N)[0]
        B = action(n,m,N)[1]
        
        for i in range(0,N):
    
            if (A[i] == 1) and (B[i] == 1) :
                yearsA.append(R)
                yearsB.append(R)
                # print("R",n,i,A[i],B[i])
                continue
            if (A[i] == 1) and (B[i] == -1) :
                yearsA.append(S)
                yearsB.append(T)
                # print("S",n,i,A[i],B[i], "B Betrayed A")
                A = action(n,m,N)[1]
                continue
            if (A[i] == -1) and (B[i] == 1) :
                yearsA.append(T)
                yearsB.append(S)
                # print("T",n,i,A[i],B[i], "A Betrayed B")
                B = action(n,m,N)[0]
                continue  
            if (A[i] == -1) and (B[i] == -1) :
                yearsA.append(P)
                yearsB.append(P)
                # print("P",n,i,A[i],B[i])
                continue  
        totA.append(np.sum(yearsA))
        Map[n][m] = (np.sum(yearsA))
nn = [i for i in range(0,N+1)]
plt.imshow(Map,origin='lower')
plt.title("years in prison wrt. m,n, S="+str(S)+"R="+str(R))
plt.xlabel("m")
plt.ylabel("n")
plt.colorbar()

#%%  13.2.a.b
N=7
L = 30
T=0
R=0.9 # R={0,1}
P=1
S=1.5  #R={1,2}
iterations =15

board = np.ones((L,L), dtype=np.int8)*int(N)

# # Configuration 1
# board[int(L/2), int(L/2)]=0


# a = 3
# board[int(L/2), int(L/2)]=0
# board[a-1,a-1]=0
# board[L-a, L-a]=0

a = 9
board[a-1,a-1]=0
board[L-a, L-a]=0

# a = 9
# b=3
# board[a-1,a-1]=0
# board[L-a, L-a]=0
# board[b-1,b-1]=0
# board[L-b, L-b]=0


plt.ion()


# P = np.zeros((L,L))
pind = np.roll(np.arange(L),-1)   
mind = np.roll(np.arange(L),1)
for t in range(0,iterations):

    counter = np.zeros((L,L))
    A = np.zeros((L,L))
    
        
    for i in range(0, L):
        for j in range(0, L):
            # A[i,j] += pd(board[i,j],board[i-1,j],R,S,T,P,N)
            # try :
            #     A[i,j] += pd(board[i,j],board[i+1,j],R,S,T,P,N)
            # except Exception:
            #     A[i,j] += pd(board[i,j],board[0,j],R,S,T,P,N)                
            # A[i,j] += pd(board[i,j],board[i,j-1],R,S,T,P,N)
            # try:
            #     A[i,j] += pd(board[i,j],board[i,j+1],R,S,T,P,N)
            # except Exception:
            #     A[i,j] += pd(board[i,j],board[i,0],R,S,T,P,N)
                
            A[i,j] += pd2(int(board[i,j]),int(board[i-1,j]),R,S,T,P,N)
            try :
                A[i,j] += pd2(int(board[i,j]),int(board[i+1,j]),R,S,T,P,N)
            except Exception:
                A[i,j] += pd2(int(board[i,j]),int(board[0,j]),R,S,T,P,N)                
            A[i,j] += pd2(int(board[i,j]),int(board[i,j-1]),R,S,T,P,N)
            try:
                A[i,j] += pd2(int(board[i,j]),int(board[i,j+1]),R,S,T,P,N)
            except Exception:
                A[i,j] += pd2(int(board[i,j]),int(board[i,0]),R,S,T,P,N)
           
            # A[i,j] += game(board[i,j],board[i-1,j],R,S,T,P,N)
            # try :
            #     A[i,j] += game(board[i,j],board[i+1,j],R,S,T,P,N)
            # except :
            #     A[i,j] += game(board[i,j],board[0,j],R,S,T,P,N) 
            # A[i,j] += game(board[i,j],board[i,j-1],R,S,T,P,N)
            # try:
            #     A[i,j] += game(board[i,j],board[i,j+1],R,S,T,P,N)
            # except :
            #     A[i,j] += game(board[i,j],board[i,0],R,S,T,P,N)
            # counter=A.copy()

            # p1, p2 = pd(R,S,N,board[i,j],board[pind[i],j])
            # P[i,j] += p1
            # P[pind[i],j] += p2

            # p1, p2 = pd(R,S,N,board[i,j],board[i,pind[j]])
            # P[i,j] += p1
            # P[i,pind[j]] += p2

    new_board = np.zeros((L,L))
    # print("A=",A)
    for i in range(L):
        for j in range(L):
            mu=0
            if np.random.rand() < mu:    # mutate the strategy with probability mu
                board[i,j] = np.random.randint(N+1)  
            else:
                pp = [A[i,j], A[mind[i],j], A[i,mind[j]], A[pind[i],j], A[i,pind[j]]]
                ss = [board[i,j], board[mind[i],j], board[i,mind[j]], board[pind[i],j], board[i,pind[j]]]
                new_board[i,j] = np.random.choice([ss[i] for i in range(5) if pp[i]==min(pp)])

    board = new_board.copy()
    

    
    # pause_time = 0.002  # seconds between frames
    plt.imshow(board)
    plt.title("S="+str(S)+" R="+str(R)+" nb. defectors="+str(2))
    # display.display(plt.gcf())
    # display.clear_output(wait=False)
    # time.sleep(pause_time)
#%% 13.2.c


N=7
L = 30
T=0
R=0.9  # R={0,1}
P=1
S=1.5  #R={1,2}
iterations =20
# board = np.random.randint(2, size=(L, L))*2-1
board = np.ones((L,L))*0

board[int(L/2), int(L/2)]=7

plt.figure(1)
plt.imshow(board)

# a = 3
# board[int(L/2), int(L/2)]=7
# board[a-1,a-1]=7
# board[L-a, L-a]=7

# a = 9
# board[a-1,a-1]=1
# board[L-a, L-a]=1

# a = 9
# b=3
# board[a-1,a-1]=1
# board[L-a, L-a]=1
# board[b-1,b-1]=1
# board[L-b, L-b]=1

nbOfInitialDefector = 3
# board[np.random.randint(L,size=nbOfInitialDefector),np.random.randint(L,size=nbOfInitialDefector)]=1

plt.ion()

def pd(n1,n2,R,S,T,P,N):                 
    # Prisoner's dilemma with two agents with strategies n1 and n2
    r = min(n1,n2)
    if n1<n2:
        p1 = r*R +(N-1-r)
        p2 = r*R + S + (N-1-r)
    elif n1 == n2:
        p1 = r*R + (N-r)
        p2 = p1
    else:
        p1 = r*R + S + (N-1-r)
        p2 = r*R + (N-1-r)
    return p1

# P = np.zeros((L,L))
pind = np.roll(np.arange(L),-1)   
mind = np.roll(np.arange(L),1)
for t in range(0,iterations):

    counter = np.zeros((L,L))
    A = np.zeros((L,L))
    
        
    for i in range(0, L):
        for j in range(0, L):
            # A[i,j] += pd(board[i,j],board[i-1,j],R,S,T,P,N)
            # try :
            #     A[i,j] += pd(board[i,j],board[i+1,j],R,S,T,P,N)
            # except Exception:
            #     A[i,j] += pd(board[i,j],board[0,j],R,S,T,P,N)                
            # A[i,j] += pd(board[i,j],board[i,j-1],R,S,T,P,N)
            # try:
            #     A[i,j] += pd(board[i,j],board[i,j+1],R,S,T,P,N)
            # except Exception:
            #     A[i,j] += pd(board[i,j],board[i,0],R,S,T,P,N)
            
            A[i,j] += pd2(int(board[i,j]),int(board[i-1,j]),R,S,T,P,N)
            try :
                A[i,j] += pd2(int(board[i,j]),int(board[i+1,j]),R,S,T,P,N)
            except Exception:
                A[i,j] += pd2(int(board[i,j]),int(board[0,j]),R,S,T,P,N)                
            A[i,j] += pd2(int(board[i,j]),int(board[i,j-1]),R,S,T,P,N)
            try:
                A[i,j] += pd2(int(board[i,j]),int(board[i,j+1]),R,S,T,P,N)
            except Exception:
                A[i,j] += pd2(int(board[i,j]),int(board[i,0]),R,S,T,P,N)
           
            # A[i,j] += game(board[i,j],board[i-1,j],R,S,T,P,N)
            # try :
            #     A[i,j] += game(board[i,j],board[i+1,j],R,S,T,P,N)
            # except :
            #     A[i,j] += game(board[i,j],board[0,j],R,S,T,P,N) 
            # A[i,j] += game(board[i,j],board[i,j-1],R,S,T,P,N)
            # try:
            #     A[i,j] += game(board[i,j],board[i,j+1],R,S,T,P,N)
            # except :
            #     A[i,j] += game(board[i,j],board[i,0],R,S,T,P,N)
            # counter=A.copy()

            # p1, p2 = pd(R,S,N,board[i,j],board[pind[i],j])
            # P[i,j] += p1
            # P[pind[i],j] += p2

            # p1, p2 = pd(R,S,N,board[i,j],board[i,pind[j]])
            # P[i,j] += p1
            # P[i,pind[j]] += p2

    new_board = np.zeros((L,L))
    # print("A=",A)
    for i in range(L):
        for j in range(L):
            mu=0
            if np.random.rand() < mu:    # mutate the strategy with probability mu
                board[i,j] = np.random.randint(N+1)  
            else:
                pp = [A[i,j], A[mind[i],j], A[i,mind[j]], A[pind[i],j], A[i,pind[j]]]
                ss = [board[i,j], board[mind[i],j], board[i,mind[j]], board[pind[i],j], board[i,pind[j]]]
                new_board[i,j] = np.random.choice([ss[i] for i in range(5) if pp[i]==min(pp)])

    board = new_board.copy()
    

    plt.figure(2)
    # pause_time = 0.002  # seconds between frames
    plt.imshow(board)
    plt.title(t)
    # display.display(plt.gcf())
    # display.clear_output(wait=False)
    # time.sleep(pause_time)
        

#%%  13.2.d


N=7
L = 30
T=0
R=0.9  # R={0,1} R<0.55 grows  R = 0.7 stable R=0.9 1s take over
P=1
S=1.5  #R={1,2}
iterations =20
# board = np.random.randint(2, size=(L, L))*2-1
board = np.ones((L,L))*0

board[int(L/2), int(L/2)]=7
board[int((L-1)/2), int(L/2)]=7
board[int((L+1)/2), int(L/2)]=7
board[int(L/2), int((L-1)/2)]=7
board[int(L/2), int((L+1)/2)]=7

# plt.figure(1)
# plt.imshow(board)


# a = 3
# board[int(L/2), int(L/2)]=7
# board[a-1,a-1]=7
# board[L-a, L-a]=7

# a = 9
# board[a-1,a-1]=1
# board[L-a, L-a]=1

# a = 9
# b=3
# board[a-1,a-1]=1
# board[L-a, L-a]=1
# board[b-1,b-1]=1
# board[L-b, L-b]=1

# nbOfInitialDefector = 
# board[np.random.randint(L,size=nbOfInitialDefector),np.random.randint(L,size=nbOfInitialDefector)]=1

plt.ion()

def pd(n1,n2,R,S,T,P,N):                 
    # Prisoner's dilemma with two agents with strategies n1 and n2
    r = min(n1,n2)
    if n1<n2:
        p1 = r*R +(N-1-r)
        p2 = r*R + S + (N-1-r)
    elif n1 == n2:
        p1 = r*R + (N-r)
        p2 = p1
    else:
        p1 = r*R + S + (N-1-r)
        p2 = r*R + (N-1-r)
    return p1

# P = np.zeros((L,L))
pind = np.roll(np.arange(L),-1)   
mind = np.roll(np.arange(L),1)
for t in range(0,iterations):

    counter = np.zeros((L,L))
    A = np.zeros((L,L))
    
        
    for i in range(0, L):
        for j in range(0, L):
            # A[i,j] += pd(board[i,j],board[i-1,j],R,S,T,P,N)
            # try :
            #     A[i,j] += pd(board[i,j],board[i+1,j],R,S,T,P,N)
            # except Exception:
            #     A[i,j] += pd(board[i,j],board[0,j],R,S,T,P,N)                
            # A[i,j] += pd(board[i,j],board[i,j-1],R,S,T,P,N)
            # try:
            #     A[i,j] += pd(board[i,j],board[i,j+1],R,S,T,P,N)
            # except Exception:
            #     A[i,j] += pd(board[i,j],board[i,0],R,S,T,P,N)
            
            A[i,j] += pd2(int(board[i,j]),int(board[i-1,j]),R,S,T,P,N)
            try :
                A[i,j] += pd2(int(board[i,j]),int(board[i+1,j]),R,S,T,P,N)
            except Exception:
                A[i,j] += pd2(int(board[i,j]),int(board[0,j]),R,S,T,P,N)                
            A[i,j] += pd2(int(board[i,j]),int(board[i,j-1]),R,S,T,P,N)
            try:
                A[i,j] += pd2(int(board[i,j]),int(board[i,j+1]),R,S,T,P,N)
            except Exception:
                A[i,j] += pd2(int(board[i,j]),int(board[i,0]),R,S,T,P,N)
           
            # A[i,j] += game(board[i,j],board[i-1,j],R,S,T,P,N)
            # try :
            #     A[i,j] += game(board[i,j],board[i+1,j],R,S,T,P,N)
            # except :
            #     A[i,j] += game(board[i,j],board[0,j],R,S,T,P,N) 
            # A[i,j] += game(board[i,j],board[i,j-1],R,S,T,P,N)
            # try:
            #     A[i,j] += game(board[i,j],board[i,j+1],R,S,T,P,N)
            # except :
            #     A[i,j] += game(board[i,j],board[i,0],R,S,T,P,N)
            # counter=A.copy()

            # p1, p2 = pd(R,S,N,board[i,j],board[pind[i],j])
            # P[i,j] += p1
            # P[pind[i],j] += p2

            # p1, p2 = pd(R,S,N,board[i,j],board[i,pind[j]])
            # P[i,j] += p1
            # P[i,pind[j]] += p2

    new_board = np.zeros((L,L))
    # print("A=",A)
    for i in range(L):
        for j in range(L):
            mu=0
            if np.random.rand() < mu:    # mutate the strategy with probability mu
                board[i,j] = np.random.randint(N+1)  
            else:
                pp = [A[i,j], A[mind[i],j], A[i,mind[j]], A[pind[i],j], A[i,pind[j]]]
                ss = [board[i,j], board[mind[i],j], board[i,mind[j]], board[pind[i],j], board[i,pind[j]]]
                new_board[i,j] = np.random.choice([ss[i] for i in range(5) if pp[i]==min(pp)])

    board = new_board.copy()
    

    plt.figure(2)
    # pause_time = 0.002  # seconds between frames
    plt.imshow(board)
    plt.title("S="+str(S)+" R="+str(R))
    # display.display(plt.gcf())
    # display.clear_output(wait=False)
    # time.sleep(pause_time)
    
#%%  13.3.a


N=7
L = 30
T=0
R=0.86  #Tune 0.82 0.84 0.86
P=1
S=1.5  #R={1,2}
iterations =70

board = np.ones((L,L))*N
RR=int((L*L)/2)
board[np.random.randint(L, size=RR),np.random.randint(L, size=RR)] = 0

nbOfInitialDefector = 3
# board[np.random.randint(L,size=nbOfInitialDefector),np.random.randint(L,size=nbOfInitialDefector)]=1

plt.ion()


pind = np.roll(np.arange(L),-1)   
mind = np.roll(np.arange(L),1)
for t in range(0,iterations):

    counter = np.zeros((L,L))
    A = np.zeros((L,L))
    
        
    for i in range(0, L):
        for j in range(0, L):
            # A[i,j] += pd(board[i,j],board[i-1,j],R,S,T,P,N)
            # try :
            #     A[i,j] += pd(board[i,j],board[i+1,j],R,S,T,P,N)
            # except Exception:
            #     A[i,j] += pd(board[i,j],board[0,j],R,S,T,P,N)                
            # A[i,j] += pd(board[i,j],board[i,j-1],R,S,T,P,N)
            # try:
            #     A[i,j] += pd(board[i,j],board[i,j+1],R,S,T,P,N)
            # except Exception:
            #     A[i,j] += pd(board[i,j],board[i,0],R,S,T,P,N)
            
            A[i,j] += pd2(int(board[i,j]),int(board[i-1,j]),R,S,T,P,N)
            try :
                A[i,j] += pd2(int(board[i,j]),int(board[i+1,j]),R,S,T,P,N)
            except Exception:
                A[i,j] += pd2(int(board[i,j]),int(board[0,j]),R,S,T,P,N)                
            A[i,j] += pd2(int(board[i,j]),int(board[i,j-1]),R,S,T,P,N)
            try:
                A[i,j] += pd2(int(board[i,j]),int(board[i,j+1]),R,S,T,P,N)
            except Exception:
                A[i,j] += pd2(int(board[i,j]),int(board[i,0]),R,S,T,P,N)
           
    new_board = np.zeros((L,L))
    # print("A=",A)
    for i in range(L):
        for j in range(L):
            mu=0.01
            if np.random.rand() < mu:    # mutate the strategy with probability mu
                board[i,j] = random.choice([0, 7])
            else:
                pp = [A[i,j], A[mind[i],j], A[i,mind[j]], A[pind[i],j], A[i,pind[j]]]
                ss = [board[i,j], board[mind[i],j], board[i,mind[j]], board[pind[i],j], board[i,pind[j]]]
                
                new_board[i,j] = np.random.choice([ss[i] for i in range(5) if pp[i]==min(pp)])

    board = new_board.copy()
    

    plt.figure(2)
    # pause_time = 0.002  # seconds between frames
    plt.imshow(board) #+=7=yellow  
    # plt.colorbar()
    plt.title("S="+str(S)+" R="+str(R))
    # display.display(plt.gcf())
    # display.clear_output(wait=False)
    # time.sleep(pause_time)


#%%  13.3.d  


N=7
L = 30
T=0
R=0.8  #Tune 0.82 0.84 0.86
P=1
S=1.5  #S={1,2}
iterations =50

board = np.ones((L,L))*N
RR=int((L*L)/2)
board[np.random.randint(L, size=RR),np.random.randint(L, size=RR)] = 0

    
plt.imshow(board)
nbOfInitialDefector = 3

# board[np.random.randint(L,size=nbOfInitialDefector),np.random.randint(L,size=nbOfInitialDefector)]=1

plt.ion()


pind = np.roll(np.arange(0,L),-1)   
mind = np.roll(np.arange(0,L),1)
for S in np.linspace(1, 2, 20):
    for t in range(0,3):
        
        counter = np.zeros((L,L))
        A = np.zeros((L,L))
        
            
        for i in range(0, L):
            for j in range(0, L):
                # A[i,j] += pd(board[i,j],board[i-1,j],R,S,T,P,N)
                # try :
                #     A[i,j] += pd(board[i,j],board[i+1,j],R,S,T,P,N)
                # except Exception:
                #     A[i,j] += pd(board[i,j],board[0,j],R,S,T,P,N)                
                # A[i,j] += pd(board[i,j],board[i,j-1],R,S,T,P,N)
                # try:
                #     A[i,j] += pd(board[i,j],board[i,j+1],R,S,T,P,N)
                # except Exception:
                #     A[i,j] += pd(board[i,j],board[i,0],R,S,T,P,N)
                
                A[i,j] += pd2(int(board[i,j]),int(board[i-1,j]),R,S,T,P,N)
                try :
                    A[i,j] += pd2(int(board[i,j]),int(board[i+1,j]),R,S,T,P,N)
                except Exception:
                    A[i,j] += pd2(int(board[i,j]),int(board[0,j]),R,S,T,P,N)                
                A[i,j] += pd2(int(board[i,j]),int(board[i,j-1]),R,S,T,P,N)
                try:
                    A[i,j] += pd2(int(board[i,j]),int(board[i,j+1]),R,S,T,P,N)
                except Exception:
                    A[i,j] += pd2(int(board[i,j]),int(board[i,0]),R,S,T,P,N)
                

               
        new_board = np.zeros((L,L))
        # print("A=",A)
        for i in range(L):
            for j in range(L):
                mu=0.01
                if np.random.rand() < mu:    # mutate the strategy with probability mu
                    board[i,j] = random.choice([0, 7])
                else:
                    pp = [A[i,j], A[mind[i],j], A[i,mind[j]], A[pind[i],j], A[i,pind[j]]]
                    ss = [board[i,j], board[mind[i],j], board[i,mind[j]], board[pind[i],j], board[i,pind[j]]]
                    
                    Q = np.random.choice([ss[i] for i in range(5) if pp[i]==min(pp)])
                    new_board[i,j] = Q
    
        board = new_board.copy()
        
    
        fig = plt.figure(2)
        pause_time = 0.002  # seconds between frames
        plt.imshow(board)
        plt.title("S="+str("%.2f" % round(S, 2)))
        # save_results_to = '/Users/thiba/Desktop/Results' + str(k) + "/" + str(h) + "/"
        # save_results_to = '/Users/thiba/Desktop/Results13.3.e/'
        # plt.savefig(save_results_to + str(h) + "H="  + "it=" +str(j)+ ".jpeg", dpi = 300)
        # plt.savefig(save_results_to + "S=" + str("%.2f" % round(S, 2))+ "t=" + str(t)+ ".jpeg", dpi = 300)
        plt.close(fig)
        # display.display(plt.gcf())
        # display.clear_output(wait=False)
        # time.sleep(pause_time)


#%%  13.4.a




N=7
L = 30
T=0
R=0.73  #Tune 0.82 0.84 0.86
P=1
S=1.5  #S={1,2}
iterations =50

# board = np.ones((L,L))*N
# RR=int((L*L)/2)
# board[np.random.randint(L, size=RR),np.random.randint(L, size=RR)] = 0

board = np.ceil(np.random.rand(L,L)*(N+1))-1



# board[np.random.randint(L,size=nbOfInitialDefector),np.random.randint(L,size=nbOfInitialDefector)]=1

plt.ion()


pind = np.roll(np.arange(0,L),-1)   
mind = np.roll(np.arange(0,L),1)
for R in np.arange(0,1,0.1):
    for t in range(0,3):
        
        counter = np.zeros((L,L))
        A = np.zeros((L,L))
        
            
        for i in range(0, L):
            for j in range(0, L):
                A[i,j] += pd2(int(board[i,j]),int(board[i-1,j]),R,S,T,P,N)
                try :
                    A[i,j] += pd2(int(board[i,j]),int(board[i+1,j]),R,S,T,P,N)
                except Exception:
                    A[i,j] += pd2(int(board[i,j]),int(board[0,j]),R,S,T,P,N)                
                A[i,j] += pd2(int(board[i,j]),int(board[i,j-1]),R,S,T,P,N)
                try:
                    A[i,j] += pd2(int(board[i,j]),int(board[i,j+1]),R,S,T,P,N)
                except Exception:
                    A[i,j] += pd2(int(board[i,j]),int(board[i,0]),R,S,T,P,N)
                
        #         A[i,j] += pd(board[i,j],board[i-1,j],R,S,T,P,N)
        #         try :
        #             A[i,j] += pd(board[i,j],board[i+1,j],R,S,T,P,N)
        #         except Exception:
        #             A[i,j] += pd(board[i,j],board[0,j],R,S,T,P,N)                
        #         A[i,j] += pd(board[i,j],board[i,j-1],R,S,T,P,N)
        #         try:
        #             A[i,j] += pd(board[i,j],board[i,j+1],R,S,T,P,N)
        #         except Exception:
        #             A[i,j] += pd(board[i,j],board[i,0],R,S,T,P,N)
               
        # new_board = np.zeros((L,L))
                # p1, p2 = pd(R,S,N,board[i,j],board[pind[i],j])
                # A[i,j] += p1
                # # P[pind[i],j] += p2
        
                # p1, p2 = pd(R,S,N,board[i,j],board[i,pind[j]])
                # A[i,j] += p1
                # P[i,pind[j]] += p2
        
        new_board = np.zeros((L,L))
    
        for i in range(L):
            for j in range(L):
                mu=0.01
                if np.random.rand() < mu:    # mutate the strategy with probability mu
                    board[i,j] = np.random.randint(N+1) 
                else:
                    pp = [A[i,j], A[mind[i],j], A[i,mind[j]], A[pind[i],j], A[i,pind[j]]]
                    ss = [board[i,j], board[mind[i],j], board[i,mind[j]], board[pind[i],j], board[i,pind[j]]]
                    new_board[i,j] = np.random.choice([ss[i] for i in range(5) if pp[i]==min(pp)])
    
    
        board = new_board.copy()
        
    
        fig = plt.figure(2)
        pause_time = 0.002  # seconds between frames
        plt.imshow(board)
        plt.colorbar()
        plt.title("R="+str("%.2f" % round(R, 2)))
    
        display.display(plt.gcf())
        # save_results_to = '/Users/thiba/Desktop/Results33/'
        # plt.savefig(save_results_to + "R=" + str("%.2f" % round(R, 2))+ "t=" + str(t)+ ".jpeg", dpi = 300)
        plt.close(fig)
        
        display.clear_output(wait=False)
        time.sleep(pause_time)

#%%  13.4.b

def pd(R,S,N,n1,n2):                 
    # Prisoner's dilemma with two agents with strategies n1 and n2
    r = min(n1,n2)
    if n1<n2:
        p1 = r*R +(N-1-r)
        p2 = r*R + S + (N-1-r)
    elif n1 == n2:
        p1 = r*R + (N-r)
        p2 = p1
    else:
        p1 = r*R + S + (N-1-r)
        p2 = r*R + (N-1-r)
    return p1, p2
N=7
L = 30
T=0
R=0.4 #Tune 0.82 0.84 0.86
P=1
S=1.5  #S={1,2}
iterations =50
board = np.ceil(np.random.rand(L,L)*(N+1))-1


plt.ion()


# a = np.zeros((L,L))
# b = np.zeros((L,L))
# c = np.zeros((L,L))
# d = np.zeros((L,L))
# e = np.zeros((L,L))
# f = np.zeros((L,L))
# g = np.zeros((L,L))

hh=[]
aa =[]
bb =[]
cc =[]
dd =[]
ee =[]
ff =[]
gg =[]
pind = np.roll(np.arange(0,L),-1)   
mind = np.roll(np.arange(0,L),1)
# for R in np.arange(0,1,0.1):
for R in np.arange(0,1,0.1):

    for t in range(0,200):
        counter = np.zeros((L,L))
        A = np.zeros((L,L))
        for i in range(0, L):
            for j in range(0, L):
                
        #         A[i,j] += pd(board[i,j],board[i-1,j],R,S,T,P,N)
        #         try :
        #             A[i,j] += pd(board[i,j],board[i+1,j],R,S,T,P,N)
        #         except Exception:
        #             A[i,j] += pd(board[i,j],board[0,j],R,S,T,P,N)                
        #         A[i,j] += pd(board[i,j],board[i,j-1],R,S,T,P,N)
        #         try:
        #             A[i,j] += pd(board[i,j],board[i,j+1],R,S,T,P,N)
        #         except Exception:
        #             A[i,j] += pd(board[i,j],board[i,0],R,S,T,P,N)
               
        # new_board = np.zeros((L,L))
                p1, p2 = pd(R,S,N,board[i,j],board[pind[i],j])
                A[i,j] += p1
                # P[pind[i],j] += p2
        
                p1, p2 = pd(R,S,N,board[i,j],board[i,pind[j]])
                A[i,j] += p1
                # P[i,pind[j]] += p2
        
        new_board = np.zeros((L,L))
    
        for i in range(L):
            for j in range(L):
                mu=0.01
                if np.random.rand() < mu:    # mutate the strategy with probability mu
                    board[i,j] = np.random.randint(N+1) 
                else:
                    pp = [A[i,j], A[mind[i],j], A[i,mind[j]], A[pind[i],j], A[i,pind[j]]]
                    ss = [board[i,j], board[mind[i],j], board[i,mind[j]], board[pind[i],j], board[i,pind[j]]]
                    new_board[i,j] = np.random.choice([ss[i] for i in range(5) if pp[i]==min(pp)])
    
        board = new_board.copy()
        a = 0
        b = a
        c = a 
        d = a
        e = a
        f = a
        g = a
        h = a
        
        for i in range(L):
            for j in range(L):
                if board[i,j]==1:
                    a +=1
                if board[i,j]==2:
                    b +=1                    
                if board[i,j]==3:
                    c +=1    
                if board[i,j]==4:
                    d +=1
                if board[i,j]==5:
                    e +=1
                if board[i,j]==6:
                    f +=1
                if board[i,j]==7:
                    g +=1
                if board[i,j]==0:
                    h +=1    
        
        
        # fig = plt.figure(2)
        # pause_time = 0.002  # seconds between frames
        # plt.imshow(board, vmin=0, vmax=7)
        # plt.colorbar()
        # plt.title("R="+str("%.2f" % round(R, 2))+"t="+str(t))
    
        # display.display(plt.gcf())
        # save_results_to = '/Users/thiba/Desktop/Results3/'
        # plt.savefig(save_results_to + "R=" + str("%.2f" % round(R, 2))+ "t=" + str(t)+ ".jpeg", dpi = 300)
        # plt.savefig(save_results_to + "R=" + str("%.2f" % round(R, 2))+ ".jpeg", dpi = 300)
        # plt.close(fig)
        
        # display.clear_output(wait=False)
        # time.sleep(pause_time)
            
        hh.append(h/(L*L))    
        aa.append(a/(L*L))
        bb.append(b/(L*L))
        cc.append(c/(L*L))
        dd.append(d/(L*L))
        ee.append(e/(L*L))
        ff.append(f/(L*L))
        gg.append(g/(L*L))
    
    fig = plt.figure(2) 
    plt.plot(hh, label="0")
    plt.plot(aa, label="1")
    plt.plot(bb, label="2")
    plt.plot(cc, label="3")
    plt.plot(dd, label="4")
    plt.plot(ee, label="5")
    plt.plot(ff, label="6")
    plt.plot(gg, label="7")
    plt.axvline(x =200 , color = 'b')#, label = 'axvline - full height')
    plt.axvline(x =400 , color = 'b')#, label = 'axvline - full height')
    plt.axvline(x =600 , color = 'b')#, label = 'axvline - full height')
    plt.axvline(x =800 , color = 'b')#, label = 'axvline - full height')
    plt.axvline(x =1000 , color = 'b')#, label = 'axvline - full height')
    plt.axvline(x =1200 , color = 'b')#, label = 'axvline - full height')
    plt.axvline(x =1400 , color = 'b')#, label = 'axvline - full height')
    plt.axvline(x =1600 , color = 'b')#, label = 'axvline - full height')
    plt.axvline(x =1800 , color = 'b')#, label = 'axvline - full height')
    plt.axvline(x =2000 , color = 'b')#, label = 'axvline - full height')
    plt.title("13.4.b, R="+str("%.2f" % round(R, 2))+" 200 iterations")
    plt.legend()
    save_results_to = "/Users/thiba/Desktop/Results3/"
    plt.savefig(save_results_to + "R=" + str("%.2f" % round(R, 2))+ "t=" + str(t)+ ".jpeg", dpi = 300)
    plt.close(fig)



    
    #%%
#%%  13.5.a

def pdd(R,S,N,n1,n2):                 
    # Prisoner's dilemma with two agents with strategies n1 and n2
    r = min(n1,n2)
    if n1<n2:
        p1 = r*R +(N-1-r)
        p2 = r*R + S + (N-1-r)
    elif n1 == n2:
        p1 = r*R + (N-r)
        p2 = p1
    else:
        p1 = r*R + S + (N-1-r)
        p2 = r*R + (N-1-r)
    return p1, p2


N=7
L = 30
T=0
R=0.4 #Tune 0.82 0.84 0.86
P=1
S=1.5  #S={1,2}

board = np.ceil(np.random.rand(L,L)*(N+1))-1

plt.ion()

pind = np.roll(np.arange(0,L),-1)   
mind = np.roll(np.arange(0,L),1)
# for R in np.arange(0,1,0.1):
count=[0]
iterations =500
for R in np.linspace(0,1,4):
    VARA = []
    VARB = []
    VARC = []
    VARD = []
    VARE = []
    VARF = []
    VARG = []
    VARH = []
    counter = np.zeros((L,L))
    for S in np.linspace(1,2,4):
        hh=[]
        aa =[]
        bb =[]
        cc =[]
        dd =[]
        ee =[]
        ff =[]
        gg =[]
        count.append(count[-1]+1)
        for t in range(0,iterations):
            
            A = np.zeros((L,L))
            for i in range(0, L):
                for j in range(0, L):
                    
            #         A[i,j] += pd(board[i,j],board[i-1,j],R,S,T,P,N)
            #         try :
            #             A[i,j] += pd(board[i,j],board[i+1,j],R,S,T,P,N)
            #         except Exception:
            #             A[i,j] += pd(board[i,j],board[0,j],R,S,T,P,N)                
            #         A[i,j] += pd(board[i,j],board[i,j-1],R,S,T,P,N)
            #         try:
            #             A[i,j] += pd(board[i,j],board[i,j+1],R,S,T,P,N)
            #         except Exception:
            #             A[i,j] += pd(board[i,j],board[i,0],R,S,T,P,N)
                   
            # new_board = np.zeros((L,L))
                    p1, p2 = pdd(R,S,N,board[i,j],board[pind[i],j])
                    A[i,j] += p1
                    # P[pind[i],j] += p2
            
                    p1, p2 = pdd(R,S,N,board[i,j],board[i,pind[j]])
                    A[i,j] += p1
                    # P[i,pind[j]] += p2
            
            new_board = np.zeros((L,L))
        
            for i in range(L):
                for j in range(L):
                    mu=0.01
                    if np.random.rand() < mu:    # mutate the strategy with probability mu
                        board[i,j] = np.random.randint(N+1) 
                    else:
                        pp = [A[i,j], A[mind[i],j], A[i,mind[j]], A[pind[i],j], A[i,pind[j]]]
                        ss = [board[i,j], board[mind[i],j], board[i,mind[j]], board[pind[i],j], board[i,pind[j]]]
                        new_board[i,j] = np.random.choice([ss[i] for i in range(5) if pp[i]==min(pp)])
        
            board = new_board.copy()
            a = 0
            b = a
            c = a 
            d = a
            e = a
            f = a
            g = a
            h = a
            if t>100:
                for i in range(L):
                    for j in range(L):
                        if board[i,j]==1:
                            a +=1
                        if board[i,j]==2:
                            b +=1                    
                        if board[i,j]==3:
                            c +=1    
                        if board[i,j]==4:
                            d +=1
                        if board[i,j]==5:
                            e +=1
                        if board[i,j]==6:
                            f +=1
                        if board[i,j]==7:
                            g +=1
                        if board[i,j]==0:
                            h +=1    
            
                
            hh.append(h/(L*L))    
            aa.append(a/(L*L))
            bb.append(b/(L*L))
            cc.append(c/(L*L))
            dd.append(d/(L*L))
            ee.append(e/(L*L))
            ff.append(f/(L*L))
            gg.append(g/(L*L))
            
            df = pd.DataFrame((hh,aa,bb,cc,dd,ee,ff,gg))
            dfVar = df.var(axis=1)
            
            VARA.append(dfVar[1])
            VARB.append(dfVar[2])
            VARC.append(dfVar[3])
            VARD.append(dfVar[4])
            VARE.append(dfVar[5])
            VARF.append(dfVar[6])
            VARG.append(dfVar[7])
            VARH.append(dfVar[0])
        
        
 

        
        fig = plt.figure(2) 
        plt.plot(VARA, label="Var 1")
        plt.plot(VARB, label="Var 2")
        plt.plot(VARC, label="Var 3")
        plt.plot(VARD, label="Var 4")
        plt.plot(VARE, label="Var 5")
        plt.plot(VARF, label="Var 6")
        plt.plot(VARG, label="Var 7")
        plt.plot(VARH, label="Var 0")
        plt.ylim([0,1])
    # plt.axvline(x =200 , color = 'b')#, label = 'axvline - full height')
    # plt.axvline(x =400 , color = 'b')#, label = 'axvline - full height')
    # plt.axvline(x =600 , color = 'b')#, label = 'axvline - full height')
    # plt.axvline(x =800 , color = 'b')#, label = 'axvline - full height')
    # plt.axvline(x =1000 , color = 'b')#, label = 'axvline - full height')
    # plt.axvline(x =1200 , color = 'b')#, label = 'axvline - full height')
    # plt.axvline(x =1400 , color = 'b')#, label = 'axvline - full height')
    # plt.axvline(x =1600 , color = 'b')#, label = 'axvline - full height')
    # plt.axvline(x =1800 , color = 'b')#, label = 'axvline - full height')
    # plt.axvline(x =2000 , color = 'b')#, label = 'axvline - full height')
        plt.title("Variance R="+str("%.2f" % round(R, 2))+" S="+str("%.2f" % round(S, 2)))
        plt.legend()
        save_results_to = "/Users/thiba/Desktop/Results3/"
        plt.savefig(save_results_to + "R="+str("%.2f" % round(R, 2))+" S="+str("%.2f" % round(S, 2))+ ".jpeg", dpi = 300)
        plt.close(fig)
        
        # fig = plt.figure(2) 
        # plt.plot(hh, label="0")
        # plt.plot(aa, label="1")
        # plt.plot(bb, label="2")
        # plt.plot(cc, label="3")
        # plt.plot(dd, label="4")
        # plt.plot(ee, label="5")
        # plt.plot(ff, label="6")
        # plt.plot(gg, label="7")
        # # plt.axvline(x =200 , color = 'b')#, label = 'axvline - full height')
        # # plt.axvline(x =400 , color = 'b')#, label = 'axvline - full height')
        # # plt.axvline(x =600 , color = 'b')#, label = 'axvline - full height')
        # # plt.axvline(x =800 , color = 'b')#, label = 'axvline - full height')
        # # plt.axvline(x =1000 , color = 'b')#, label = 'axvline - full height')
        # # plt.axvline(x =1200 , color = 'b')#, label = 'axvline - full height')
        # # plt.axvline(x =1400 , color = 'b')#, label = 'axvline - full height')
        # # plt.axvline(x =1600 , color = 'b')#, label = 'axvline - full height')
        # # plt.axvline(x =1800 , color = 'b')#, label = 'axvline - full height')
        # # plt.axvline(x =2000 , color = 'b')#, label = 'axvline - full height')
        # plt.title("R="+str("%.2f" % round(R, 2))+" S="+str("%.2f" % round(S, 2)))
        # plt.legend()
        # save_results_to = "/Users/thiba/Desktop/Results3/"
        # plt.savefig(save_results_to + "R="+str("%.2f" % round(R, 2))+" S="+str("%.2f" % round(S, 2))+ ".jpeg", dpi = 300)
        # plt.close(fig)