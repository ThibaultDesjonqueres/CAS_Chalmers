import numpy as np
import random
import matplotlib.pyplot as plt
from copy import copy
import itertools
from scipy import optimize

from scipy.ndimage import convolve, generate_binary_structure
from itertools import product
from get_energy import *
from get_energy_Vary_H import *
from get_energy_H_is_0 import *

import time



#Create x
N = 50
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

Q = get_energy_H_is_0(x,sample,N,iterations,T,kB,J,5)


#%%

start_time = time.time()


#%%

#Create x
N=200
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
a= np.linspace(0.12,0.2,len(S[0]))
p = np.polyfit(a,S[0],1)
poly1d_fn = np.poly1d(p)
plt.plot(a,poly1d_fn(a),color="black")
plt.title("Magnetisation Vs H, at T=1")
plt.xlabel("Magnetic Field")
plt.ylabel("Magnetization")
p1 = p[0]
plt.legend(["magnetization","Linear Regression Xi[m3kg-1] = %i" %p1])

a= np.linspace(0.12,0.2,len(S[0]))
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

print("Phase 2 Start")
print("--- %s seconds ---" % (time.time() - start_time))

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
plt.title("Magnetization for 5 different H, at T=1, %1.0f" %N + " x %1.0f " %N + " lattice")

plt.figure(7)
plt.plot(df["T2"][-2], color="b")
plt.plot(df["T2"][-1], color="r")
plt.plot(df["T2"][0], color="g")
plt.plot(df["T2"][1], color="orange")
plt.plot(df["T2"][2], color="purple")
plt.legend(["H=-2","H=-1","H=0","H=1","H=2"])
plt.ylabel("Magnetization", fontsize=15)
plt.xlabel("iterations", fontsize=15)
plt.title("Magnetization for 5 different H, at T=Tc, %1.0f" %N + " x %1.0f " %N + " lattice")

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




print("--- %s seconds ---" % (time.time() - start_time))