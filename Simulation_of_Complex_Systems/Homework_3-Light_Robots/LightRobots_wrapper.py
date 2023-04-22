import numpy as np
import random
import matplotlib.pyplot as plt
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt

"""
The only combination of length, L, and diffusion rate, D, 
that yields a unit of time is
Diffusion Time Scale = TD ~ L^2/D.

where t D r = R−1 is the characteristic timescale 
of the rotational diffusion, which
represents the average time it takes the particle
 to change its direction of
propagation. 
"""
V0 = 4
Vinf = 1


mu = 0
sigma = 1
a = random.gauss(mu, sigma)
Dr = 0.5
tau = 1/Dr
iterations = 1000
deltaT = 0.01
I = np.array([i for i in np.linspace(0,40,iterations)])
I = np.array([0 for i in np.linspace(0,40,iterations)])

v = Vinf + (V0 - Vinf)*np.exp(-I)

mean = 0
std = 1 
num_samples = iterations
w = np.random.normal(mean, std, size=num_samples) #Gauss. distrib.

x=np.ones([iterations])
y=np.ones([iterations])
theta=np.ones([iterations])

Ilim = []
index = []

plt.ion()

for i in range(0,iterations-1):
    index.append(i)
    x[i+1] = x[i] + v[i]*np.cos(theta[i])  #*np.sqrt(deltaT)
    y[i+1] = y[i] + v[i]*np.sin(theta[i])  #*np.sqrt(deltaT)
    theta[i+1] = theta[i] + np.sqrt(2/tau)*w[i]
    if v[i] == Vinf :
        Ilim.append(i)
        
    # plt.axis([-50,50,-50,50])
    # plt.scatter(x[i],y[i], label=i)
    # plt.show()
    # plt.legend()
    # plt.pause(0.005)
    # plt.clf()




if I[3] != 0 :  

    
    plt.title("I is NOT     zero")
    plt.plot(index,x[:-1], color="blue", label="x position")
    plt.plot(index,y[:-1], color="red", label="y position")
    plt.plot(index,theta[:-1], color="green",label="rotation")
    plt.xlabel("Light Intensity")
    plt.axvline(I[Ilim[0]], color = 'b', label = 'v = Vinf')
    plt.ylabel("Position and rotation")
    plt.legend()
    
if I[3] == 0 :   
    plt.title("I is zero")
    plt.plot(index,x[:-1], color="blue", label="x position")
    plt.plot(index,y[:-1], color="red", label="y position")
    plt.plot(index,theta[:-1], color="green",label="rotation")
    plt.xlabel("Light Intensity")
    plt.ylabel("Position and rotation")
    plt.legend()
    
#%%

V0 = 4
Vinf = 1


mu = 0
sigma = 1
a = random.gauss(mu, sigma)
Dr = 0.5
tau = 1/Dr
iterations = 5000
deltaT = 0.01


v = Vinf + (V0 - Vinf)*np.exp(-I)
L = V0*tau
mean = 0
std = 1 
num_samples = iterations
w = np.random.normal(mean, std, size=num_samples) #Gauss. distrib.



Ilim = []


plt.ion()

A = L/10
posX = []
posY = []
posTheta = []

Ilist = []
FuncI = []
for j in np.linspace(L/10,L*10,15) :
    index = []
    x=np.empty([iterations])
    I=np.empty([iterations])
    y=np.empty([iterations])
    theta=np.empty([iterations])
    
    FuncI=np.empty([iterations])
    for i in range(0,iterations-1):
        index.append(i)
        I[i] = (np.sin(2*np.pi*x[i]/j))**2
        FuncI[i] = (np.sin(2*np.pi*i/j))**2
        
        v = Vinf + (V0 - Vinf)*np.exp(-I)
        x[i] = x[i-1] + v[i]*np.cos(theta[i-1])  #*np.sqrt(deltaT)
        y[i] = y[i-1] + v[i]*np.sin(theta[i-1])  #*np.sqrt(deltaT)
        theta[i] = theta[i-1] + np.sqrt(2/tau)*w[i]

    
            
        # plt.axis([-50,50,-50,50])
        # plt.scatter(x[i],y[i], label=i)
        # plt.show()
        # plt.legend()
        # plt.pause(0.005)
        # plt.clf()
        
    posX.append(x)
    posY.append(y)
    posTheta.append(theta)
    Ilist.append(I)

# fig = plt.figure(1)
# plt.scatter(posX[0],posY[0], color="red", s=0.5)
# plt.scatter(posX[1],posY[1], color="blue", s=0.5)
# plt.scatter(posX[2],posY[2], color="orange", s=0.5)
#%%
figure = plt.figure(1)
plt.scatter(posX[0][:-1],Ilist[0][:-1],s=0.5, label="L/10", color="black")
# plt.scatter(posX[1][:-1],Ilist[1][:-1],s=0.5, label="", color="blue")
# plt.scatter(posX[2][:-1],Ilist[2][:-1],s=0.5, label="")
plt.scatter(posX[3][:-1],Ilist[3][:-1],s=0.5, label="", color="blue")
# plt.scatter(posX[4][:-1],Ilist[4][:-1],s=0.5, label="")
# plt.scatter(posX[5][:-1],Ilist[5][:-1],s=0.5, label="", color="purple")
# plt.scatter(posX[6][:-1],Ilist[6][:-1],s=0.5, label="", color="purple")
# plt.scatter(posX[7][:-1],Ilist[7][:-1],s=0.5, label="", color="cyan")
# plt.scatter(posX[8][:-1],Ilist[8][:-1],s=0.5, label="", color="magenta")
#plt.scatter(posX[9][:-1],Ilist[9][:-1],s=0.5, label="", color="black")
# plt.scatter(posX[10][:-1],Ilist[10][:-1],s=0.5, label="", color="yellow")
# plt.scatter(posX[11][:-1],Ilist[11][:-1],s=0.5, label="", color="green")
# plt.scatter(posX[12][:-1],Ilist[12][:-1],s=0.5, label="", color="orange")
# plt.scatter(posX[13][:-1],Ilist[13][:-1],s=0.5, label="", color="pink")
# plt.scatter(posX[14][:-1],Ilist[14][:-1],s=0.5, label="10*L", color="red")


plt.legend()
plt.show()
#%%
figure = plt.figure(3)
plt.plot(index[:-1],Ilist[14][:-2])
plt.show()

#%%

plt.plot(index, FuncI[:-1])

