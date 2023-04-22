import numpy as np
import random
import matplotlib.pyplot as plt
from copy import copy
import itertools
from scipy import optimize

from scipy.ndimage import convolve, generate_binary_structure
from itertools import product
from get_energy import *

from getsumofneighbors import *



#%%
Z = 1+1 + np.exp(-2)
kB = 1.380649*10e-23

K = 1

p1 = 1/Z
p2 = p1
pB = np.exp(-2)/Z


L = np.array([0,1])
R =  np.array([0,0])
M =  np.array([0,0])

a = 0


Is = []
L0s = []
R0s = []
M0s = []


for i in range(0,100000):
    r = random.uniform(0,1)
    
    Is.append(i)
    L0s.append(L[0]/i)
    R0s.append(R[0]/i)
    M0s.append(M[0]/i)

    if L[1] == 1:
        #print("L")

        if r <= p1 or p1 < r <= p2+p1 :  #Go to L
            L[a]= L[a]+ 1
            L[1] = 1
            R[1] = 0
            M[1] = 0
           # print(L,M,R,"Stay L")
            continue
            
        if p2+p1 < r <= p2+p1+pB :  #Go to M
            M[a]= M[a]+ 1
            L[1] = 0
            R[1] = 0
            M[1] = 1
            #print(L,M,R,"Go M")
            continue
    if R[1] == 1 :
        #print("R")
        if r <= p1 or p1 < r <= p2+p1 :
            R[a]= R[a]+ 1
            L[1] = 0
            R[1] = 1
            M[1] = 0
            #print(L,M,R,"Stay R")
            continue
            
        if p2+p1 < r <=  p2+p1+pB :
            M[a]= M[a]+ 1
            L[1] = 0
            R[1] = 0
            M[1] = 1 
            #print(L,M,R,"Go M")
            continue
    if M[1] == 1 :
        #print("M")
        
        if r <= p1 :
            L[a]= L[a]+ 1
            L[1] = 1
            R[1] = 0
            M[1] = 0
            #print(L,M,R,"Go L")
            continue
            
        if p1 < r <= p2+p1 :
            R[a]= R[a]+ 1
            L[1] = 0
            R[1] = 1
            M[1] = 0
            #print(L,M,R,"Go R")
            continue

        if p2+p1 < r <=  p2+p1+pB :
            M[a]= M[a]+ 1
            L[1] = 0
            R[1] = 0
            M[1] = 1  
            #print(L,M,R,"Stay M")
            continue
        

print(L,M,R)
fig = plt.plot()
plt.title("Average presence of particle in each box, E = 2.Kb.T")
plt.title("Distribution of the particle over time, E = 2.Kb.T")
plt.xlabel("Iteration Number", fontsize=15)
plt.ylabel("L/i , M/i, R/i", fontsize=15)
plt.scatter(Is,L0s, marker=".", color ="blue" )
plt.scatter(Is,M0s, marker=".", color ="orange" )
plt.scatter(Is,R0s, marker=".", color ="red" )
plt.legend(["Left","Middle","Right"])
plt.show()

#%% 2.1.b) # Make K vary [0.001, 3, 8, 10]
K = 8   # Make K vary, realize it changes frequency 
                    #of occurence, equilibrium distribution
#K = E/T           # When K= 0.0001, equiprobability of L M R
                    # When K 
                    #
Z = 1+1 + np.exp(-K)
kB = 1.380649*10e-23
p1 = 1/Z
p2 = p1
pB = np.exp(-K)/Z
L = np.array([0,1])
R =  np.array([0,0])
M =  np.array([0,0])
a = 0
Is = []
L0s = []
R0s = []
M0s = []

for i in range(0,100000):
    r = random.uniform(0,1)
    
    Is.append(i)
    L0s.append(L[0]/i)
    R0s.append(R[0]/i)
    M0s.append(M[0]/i)

    
    #print(L0s, Is)
    
    if L[1] == 1:
        #print("L")

        if r <= p1 or p1 < r <= p2+p1 :  #Go to L
            L[a]= L[a]+ 1
            L[1] = 1
            R[1] = 0
            M[1] = 0
           # print(L,M,R,"Stay L")
            continue
            
        if p2+p1 < r <= p2+p1+pB :  #Go to M
            M[a]= M[a]+ 1
            L[1] = 0
            R[1] = 0
            M[1] = 1
            #print(L,M,R,"Go M")
            continue
    
    if R[1] == 1 :
        #print("R")
        if r <= p1 or p1 < r <= p2+p1 :
            R[a]= R[a]+ 1
            L[1] = 0
            R[1] = 1
            M[1] = 0
            #print(L,M,R,"Stay R")
            continue
            
        if p2+p1 < r <=  p2+p1+pB :
            M[a]= M[a]+ 1
            L[1] = 0
            R[1] = 0
            M[1] = 1 
            #print(L,M,R,"Go M")
            continue
    
    if M[1] == 1 :
        #print("M")
        
        if r <= p1 :
            L[a]= L[a]+ 1
            L[1] = 1
            R[1] = 0
            M[1] = 0
            #print(L,M,R,"Go L")
            continue
            
        if p1 < r <= p2+p1 :
            R[a]= R[a]+ 1
            L[1] = 0
            R[1] = 1
            M[1] = 0
            #print(L,M,R,"Go R")
            continue

        if p2+p1 < r <=  p2+p1+pB :
            M[a]= M[a]+ 1
            L[1] = 0
            R[1] = 0
            M[1] = 1  
            #print(L,M,R,"Stay M")
            continue
        

print(L[0]/i,M[0]/i,R[0]/i)
#print(L[0]/M[0],R[0]/M[0])

print(L,M,R)
fig = plt.plot()
plt.title("Average presence of particle in each box, T<<<E")
#plt.title("Distribution of the particle over time")
plt.xlabel("Iteration Number", fontsize=15)
plt.ylabel("L/i , M/i, R/i", fontsize=15)
plt.scatter(Is,L0s, marker="_", color ="blue" )
plt.scatter(Is,M0s, marker="_", color ="orange" )
plt.scatter(Is,R0s, marker="_", color ="red" )
plt.legend(["Left","Middle","Right"])
plt.show()

#%%  2.1.c   #Escape Time



K = 0.001    # Make K vary, realize it changes frequency 
                    #of occurence, equilibrium distribution
#K = E/T           # When K= 0.0001, equiprobability of L M R
                    # When K 
                    #
Z = 1+1 + np.exp(-K)
kB = 1.380649*10e-23
p1 = 1/Z
p2 = p1
pB = np.exp(-K)/Z
L = np.array([0,1])
R =  np.array([0,0])
M =  np.array([0,0])
a = 0
Is = []
L0s = []
R0s = []
M0s = []

List = []

for j in range(0,100):
    for i in range(0,100000):
        r = random.uniform(0,1)
        
        Is.append(i)
        L0s.append(L[0]/i)
        R0s.append(R[0]/i)
        M0s.append(M[0]/i)
    
        
        #print(L0s, Is)
        
        if L[1] == 1:
            #print("L")
    
            if r <= p1 or p1 < r <= p2+p1 :  #Go to L
                L[a]= L[a]+ 1
                L[1] = 1
                R[1] = 0
                M[1] = 0
                List.append(i)
                break
            

List = np.asarray(List)

print("Average Escape Time", np.mean(List))



#%%
#Create x
N = 200
xSize = N
ySize = N
tot = xSize*ySize
x = np.random.randint(2, size=(xSize, ySize))*2-1   #z,x,
sample = int(0.1*tot)
#plt.imshow(x)
iterations = 1000

#indices = indices.tolist()
Tc = 2.269
T = np.array([1.0,Tc,5.0])
kB = 1  
J = 1

Q = get_energy_H_is_0(x,sample,N,iterations,T,kB,J,-5)


#%%
#Create x

xSize = N
ySize = N
tot = xSize*ySize
x = np.random.randint(2, size=(xSize, ySize))*2-1   #z,x,
sample = int(0.1*tot)
#plt.imshow(x)
iterations = 1000

#indices = indices.tolist()
Tc = 2.269
T = np.array([1.0,Tc,5.0])
kB = 1  
J = 1

intervalH = np.arange(-0.5,0.5,0.1)

S = get_energy_Vary_H(x,sample,N,iterations,T,kB,J, intervalH)

#%%
intervalH = np.linspace(-0.5,0.5,len(S[0]))
plt.figure(3)
plt.plot(intervalH,S[0],color="b")

p = np.polyfit(np.linspace(0.39,0.41,len(S[0])),S[0],1)
poly1d_fn = np.poly1d(p)
plt.plot(np.linspace(0.36,0.44,len(S[0])),poly1d_fn(np.linspace(0.38,0.44,len(S[0]))),color="black")
plt.title("Magnetisation Vs H, at T=1")
plt.xlabel("Magnetic Field")
plt.ylabel("Magnetization")
p1 = p[0]
plt.legend(["magnetization","Linear Regression Xi[m3kg-1] = %i" %p1])

a= np.linspace(0.02,0.2,len(S[0]))

intervalH = np.linspace(-0.5,0.5,len(S[0]))
plt.figure(4)
plt.plot(intervalH,S[1],color="r")
p = np.polyfit(a, S[1],1)
poly1d_fn = np.poly1d(p)
plt.plot(a,poly1d_fn(a),color="black")
plt.title("Magnetisation Vs H, at T=Tc")
plt.xlabel("Magnetic Field", fontsize=15)
plt.ylabel("Magnetization", fontsize=15)
p1 = p[0]
plt.legend(["magnetization","Linear Regression Xi[m3kg-1] = %i" %p1])


intervalH = np.linspace(-0.5,0.5,len(S[0]))
plt.figure(5)
plt.plot(intervalH,S[2],color="g")
p = np.polyfit(intervalH, S[2],1)
poly1d_fn = np.poly1d(p)
plt.plot(intervalH,poly1d_fn(intervalH),color="black")
plt.title("Magnetisation Vs H, at T=5")
plt.xlabel("Magnetic Field")
plt.ylabel("Magnetization")
p1 = p[0]
plt.legend(["magnetization","Linear Regression Xi[m3kg-1] = %i" %p1])



#%%
#Create x

xSize = N
ySize = N
tot = xSize*ySize
x = np.random.randint(2, size=(xSize, ySize))*2-1   #z,x,

sample = int(0.1*tot)
#plt.imshow(x)
iterations = 1000

#indices = indices.tolist()
Tc = 2.269
T = np.array([1.0,Tc,5.0])

kB = 1  
J = 1
H = np.array([-2,-1,0,1,2])


R = get_energy(x,sample,N,iterations,T,kB,J,H)
DD = {'T1': R[0], 'T2': R[1], 'T3': R[2]}
df = pd.DataFrame(data=DD)





#%%

plt.figure(6)
plt.plot(df["T1"][-2], color="b")
plt.plot(df["T1"][-1], color="r")
plt.plot(df["T1"][0], color="g")
plt.plot(df["T1"][1], color="orange")
plt.plot(df["T1"][2], color="purple")
plt.legend(["H=-2","H=-1","H=0","H=1","H=2"])
plt.ylabel("Magnetization", fontsize=15)
plt.xlabel("iterations", fontsize=15)
plt.title("Magnetization for 5 different H, at T=1, %1.0f"%N + " x %1.0f "%N + " lattice")

plt.figure(7)
plt.plot(df["T2"][-2], color="b")
plt.plot(df["T2"][-1], color="r")
plt.plot(df["T2"][0], color="g")
plt.plot(df["T2"][1], color="orange")
plt.plot(df["T2"][2], color="purple")
plt.legend(["H=-2","H=-1","H=0","H=1","H=2"])
plt.ylabel("Magnetization", fontsize=15)
plt.xlabel("iterations", fontsize=15)
plt.title("Magnetization for 5 different H, at T=Tc, %1.0f"%N + " x %1.0f "%N + " lattice")

plt.figure(8)
plt.plot(df["T3"][-2], color="b")
plt.plot(df["T3"][-1], color="r")
plt.plot(df["T3"][0], color="g")
plt.plot(df["T3"][1], color="orange")
plt.plot(df["T3"][2], color="purple")
plt.legend(["H=-2","H=-1","H=0","H=1","H=2"])
plt.ylabel("Magnetization", fontsize=15)
plt.xlabel("iterations", fontsize=15)
plt.title("Magnetization for 5 different H, at T=5, %1.0f"%N + " x %1.0f "%N + " lattice")



#%%

A = [x/-2 for x in df["T1"][-2]]
B = [x/-1 for x in df["T1"][-1]]
C = [x/1 for x in df["T1"][1]]
D = [x/2 for x in df["T1"][2]]
NAMESXi1 = {'T=1,H=-2': A, 'T=1,H=-1': B, 'T=1,H=1': C,'T=1,H=2': D}
Xi1 = pd.DataFrame(data=NAMESXi1)

A = [x/-2 for x in df["T2"][-2]]
B = [x/-1 for x in df["T2"][-1]]
C = [x/1 for x in df["T2"][1]]
D = [x/2 for x in df["T2"][2]]
NAMESXi2 = {'T=1,H=-2': A, 'T=1,H=-1': B, 'T=1,H=1': C,'T=1,H=2': D}
Xi2 = pd.DataFrame(data=NAMESXi2)

A = [x/-2 for x in df["T3"][-2]]
B = [x/-1 for x in df["T3"][-1]]
C = [x/1 for x in df["T3"][1]]
D = [x/2 for x in df["T3"][2]]
NAMESXi3 = {'T=1,H=-2': A, 'T=1,H=-1': B, 'T=1,H=1': C,'T=1,H=2': D}
Xi3 = pd.DataFrame(data=NAMESXi3)




    
    

