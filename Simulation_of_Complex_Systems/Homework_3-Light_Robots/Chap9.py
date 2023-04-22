import numpy as np
import random
import matplotlib.pyplot as plt
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt
# generate random floating point values
from random import seed
from random import random

from numpy import ones,vstack
from numpy.linalg import lstsq

def chunks(list_in, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(list_in), n):
        # Create an index range for l of n items:
        yield list_in[i:i+n]
# then just do this to get your output

def line(x,y,i,j,h):    
    
    x_coords = (x[i][j],x[i][h])
    y_coords = (y[i][j],y[i][h])
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    return m,c

def pol(x,y,i,j,h,a):
    X = [x[i][j],x[i][h]]
    Y = [y[i][j],y[i][h]]
    coefficients = np.polyfit(X, Y, 1)
    # # Let's compute the values of the line...
    # polynomial = np.poly1d(coefficients)
    # x_axis = np.linspace(-0.002e-5,0.002e-5,100)
    # y_axis = polynomial(x_axis)

    # # ...and plot the points and the line
    # plt.plot(x_axis, y_axis)
    # plt.xlim(-a,a)
    # plt.ylim(-a,a)
    # # plt.plot( x[0], y[0], 'go' )
    # # plt.plot( x[1], y[1], 'go' )
    # plt.grid('on')
    # plt.show()
    return coefficients



#%% 9.1.a
import random
mu = 0
sigma = 1
a = random.gauss(mu, sigma)
iterations = 500
deltaT = 0.01
mean = 0
std = 1 
num_samples = iterations
wx = np.random.normal(mean, std, size=num_samples) #Gauss. distrib.
wy = np.random.normal(mean, std, size=num_samples)
wtheta = np.random.normal(mean, std, size=num_samples)
x=np.zeros([iterations])
y=np.zeros([iterations])
theta=0
index = []
plt.ion()

Dt = 2*10**(-13)
Dr = 0.5
v = 3*10**(-6)
for i in range(0,iterations-1):
    index.append(i)

    x[i+1] = (v*np.cos(theta)*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[i]
    y[i+1] = (v*np.sin(theta)*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[i]
    theta = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta
   
    figure = plt.figure(1)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    #plt.axis([-50,50,-50,50])
    plt.scatter(x[i]*(10**6),y[i]*(10**6), label=i)
    plt.show()
    plt.legend()
    plt.pause(0.005)
    plt.xlabel("x-coordinate")
    plt.title("One particle, motion in 2D plan")
    plt.ylabel("y-coordinate")
    plt.clf()


plt.scatter(x,y, s=0.5)


#%% 9.1.b

mu = 0
sigma = 1
a = random.gauss(mu, sigma)
iterations = 5000
deltaT = 0.01
mean = 0
std = 1 
num_samples = iterations


plt.ion()

#Dt = 2*10**(-3)
Dt = 2*10**(-13)
#Dt = 2*10**(-33)

Dr = 0.000001
#Dr = 0.5
Dr = 5
v = 3*10**(-6)

posX = []
posY = []
posTheta = []

for j in np.linspace(0,v,5):
    print(j)
    x=np.zeros(iterations+1)
    y=np.zeros(iterations+1)
    theta=np.zeros(iterations+1)
    index = []
    theta = 0
    wx = np.random.normal(mean, std, size=num_samples) #Gauss. distrib.
    wy = np.random.normal(mean, std, size=num_samples)
    wtheta = np.random.normal(mean, std, size=num_samples)
    
    for i in range(iterations):
        index.append(i)
    
        # x[i+1] = x[i] + (v*np.cos(theta[i]))*deltaT + (np.sqrt(2*Dt)*np.random.normal())*np.sqrt(deltaT)
        # y[i+1] = y[i] + (v*np.sin(theta[i]))*deltaT + (np.sqrt(2*Dt)*np.random.normal())*np.sqrt(deltaT)
        # theta[i+1] = theta[i] + (np.sqrt(2*Dr)*np.random.normal())*np.sqrt(deltaT)
        
        x[i+1] = (j*np.cos(theta)*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[i]
        y[i+1] = (j*np.sin(theta)*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[i]
        theta = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta
        
        # plt.axis([-50,50,-50,50]
        # plt.scatter(x[i],y[i], label=i)
        # plt.show()
        # plt.legend()
        # plt.pause(0.005)
        # plt.clf()
        
    posX.append(x)
    posY.append(y)
    posTheta.append(theta)

fir = plt.figure(1)
plt.scatter(posX[0],posY[0], s=5, color="red", label="v=0.0")
plt.scatter(posX[1],posY[1], s=5, color="blue", label="v=7.5e-07")
plt.scatter(posX[2],posY[2], s=5, color="orange", label="v=1.5e-06")
plt.scatter(posX[3],posY[3], s=5, color="green", label="v=2.25e-06")
plt.scatter(posX[4],posY[4], s=5, color="black", label="v=3e-06")
plt.legend(prop={'size': 16})
plt.xlabel("x-coordinate")
plt.title("One particle, motion in 2D plan, Dr =5")
plt.ylabel("y-coordinate")

#%% 9.1.c

mu = 0
sigma = 1
a = random.gauss(mu, sigma)
iterations = 5000
deltaT = 0.01
mean = 0
std = 1 
num_samples = iterations


plt.ion()

Dt = 2*10**(-5)
Dr = 0.5
v = 3*10**(-6)

posX = []
posY = []
posTheta = []

for j in np.linspace(0,v,5):
    print(j)
    x=np.zeros(iterations+1)
    y=np.zeros(iterations+1)
    theta=np.zeros(iterations+1)
    index = []
    theta = 0
    wx = np.random.normal(mean, std, size=num_samples) #Gauss. distrib.
    wy = np.random.normal(mean, std, size=num_samples)
    wtheta = np.random.normal(mean, std, size=num_samples)
    
    for i in range(iterations):
        index.append(i)
    
        # x[i+1] = x[i] + (v*np.cos(theta[i]))*deltaT + (np.sqrt(2*Dt)*np.random.normal())*np.sqrt(deltaT)
        # y[i+1] = y[i] + (v*np.sin(theta[i]))*deltaT + (np.sqrt(2*Dt)*np.random.normal())*np.sqrt(deltaT)
        # theta[i+1] = theta[i] + (np.sqrt(2*Dr)*np.random.normal())*np.sqrt(deltaT)
        
        x[i+1] = (j*np.cos(theta)*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[i]
        y[i+1] = (j*np.sin(theta)*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[i]
        theta = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta
        
        # plt.axis([-50,50,-50,50]
        # plt.scatter(x[i],y[i], label=i)
        # plt.show()
        # plt.legend()
        # plt.pause(0.005)
        # plt.clf()
        
    posX.append(x)
    posY.append(y)
    posTheta.append(theta)

fir = plt.figure(2)
plt.scatter(posX[0],posY[0], s=0.5, color="red",label="v=0.0")
plt.scatter(posX[1],posY[1], s=0.5, color="blue",label="v=7.5e-07")
plt.scatter(posX[2],posY[2], s=0.5, color="orange",label="v=1.5e-06")
plt.scatter(posX[3],posY[3], s=0.5, color="green",label="v=2.25e-06")
plt.scatter(posX[4],posY[4], s=0.5, color="black",label="v=3e-06")




#%% 9.2.a

mu = 0
sigma = 1
iterations = 500
deltaT = 0.01
mean = 0
std = 1 
Dt = 0.1*(10**(-6))**2
Dr = 1  #Rotational Diffusion
v = 3*10**(-6)
posX = []
posY = []
MSD_v0 = np.zeros((iterations+1))
trajectories = 100

theta=np.zeros((trajectories,iterations+1))
MSD_list = []
for h in np.linspace(0,v,4):
    print(h)
    x=np.zeros((trajectories,iterations+1))
    y=np.zeros((trajectories,iterations+1))
    MSD_v0 = np.zeros((iterations+1))
    for j in range(trajectories) :
        index = []
        theta = 0
        for i in range(iterations):
            index.append(i)
            x[j][i+1] = (h*np.cos(theta)*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[j][i]
            y[j][i+1] = (h*np.sin(theta)*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[j][i]
            theta = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta
    
    for i in range(iterations):
        MSD_v0[i+1] = np.sum(x[:,i+1]**2 + y[:,i+1]**2) / iterations
    MSD_list.append(MSD_v0) 
plt.loglog(MSD_list[0],color="blue",label="v=0.0")
plt.loglog(MSD_list[1],color="orange",label="v=1e-06")
plt.loglog(MSD_list[2],color="green",label="v=2e-06")
plt.loglog(MSD_list[3],color="black",label="v=3e-06")
plt.legend()
plt.title("MSD Ensemble")
plt.xlabel("t")
plt.xlabel("MSD(t)")   


#%%  9.2.b METHOD 2

mu = 0
sigma = 1
iterations = int(10e4)
deltaT = 0.01
mean = 0
std = 1 
Dt = 0.1*(10**(-6))**2
Dr = 1   #Rotational Diffusion
v = 3*10**(-6)
posX = []
posY = []
MSD_v0 = np.zeros((iterations+1))
trajectories = 100

theta=np.zeros((iterations+1))

MSD_list = []
#for h in np.linspace(0,v,4):
x=np.zeros((iterations+1))
y=np.zeros((iterations+1))
MSD_v0 = np.zeros(iterations+1)
theta = 0
portion=100
R=np.zeros((trajectories,iterations+1))
n=100
MSD = np.zeros((iterations-n+1,n))

#for j in range(trajectories):
for i in range(iterations):
    
    x[i+1] = (v*np.cos(theta)*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[i]
    y[i+1] = (v*np.sin(theta)*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[i]
    theta = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta

# for j in range(trajectories):  
    # for i in range(iterations):  
    #     print(x[j][0])
    #     R[j][i] = (x[j][i] - x[j][0])**2 - (y[j][i] - y[j][0])**2

for i in range(iterations-n+1):
    batchX = x[i:i+n]
    batchY = y[i:i+n]

    for h in range(len(batchX)):        
        MSD[i,h] = (batchX[h] - batchX[0])**2 + (batchY[h] - batchY[0])**2

MSDdf = pd.DataFrame(MSD)
#MSDdf = MSDdf.iloc[:,1:]
MSDmean = MSDdf.mean(axis=0)
index = [i for i in range(len(MSDmean))]
plt.loglog(index,MSDmean)
plt.legend()
plt.title("MSD time average")
plt.xlabel("t")
plt.xlabel("MSD(t)")  


    


#%%   9.3.a 

mu = 0
sigma = 1
iterations = 1000
deltaT = 0.01
mean = 0
std = 1 
N = 2
R = 1e-6  # Difference between two random consecutive
                    # position/2
x=np.zeros([iterations,N])
y=x.copy()
theta=x.copy()
boundarymax = 2e-6
for i in range(0,iterations):
    for j in range(0,N):
    	x[i][j] = (random()*2-1)*boundarymax   #*10e-6
    	y[i][j] = (random()*2-1)*boundarymax   #*10e-6


index = []
plt.ion()
Dt = 0.1*(10**(-6))**2
Dr = 1
v = 3*10**(-6)

boundarymax = 2e-6
a = (boundarymax + boundarymax*0.3)
b =[]
c = []
INITIALxj = []
INITIALxh = []
INITIALyj = []
INITIALyh = []
FINALxj = []
FINALxh = []
FINALyj = []
FINALyh = []
for i in range(0,iterations-1):
    for j in range(0,N) :
        if x[i][j] > boundarymax :
            x[i][j] = -boundarymax
            
        if x[i][j] < -boundarymax :
            x[i][j] = boundarymax
            
        if y[i][j] > boundarymax :
            y[i][j] = -boundarymax
            
        if y[i][j] < -boundarymax :
            y[i][j] = boundarymax
            

        c = [np.sqrt((x[i][j]-x[i][h])**2+(y[i][j]-y[i][h])**2) for h in range(N)]

        for h in range(0,N) :  
            if (0.000000000000001 < c[h] < R) :
                INITIALxj.append(x[i][j])
                INITIALxh.append(x[i][h])
                INITIALyj.append(y[i][j])
                INITIALyh.append(y[i][h])
                b = (np.sqrt((x[i][j]-x[i][h])**2+(y[i][j]-y[i][h])**2))
                print(b)
                minx = min(x[i][j],x[i][h])
                miny = min(y[i][j],y[i][h])

                
                if minx == x[i][j] :
                    xj_old = x[i][j]
                    x[i][j] = xj_old - b/2
                     
                    # if x[i][j] > boundarymax :
                    #     x[i][j] = x[i-1][j]
                        
                    # if x[i][j] < -boundarymax :
                    #     x[i][j] = x[i-1][j]
                        
                    xh_old = x[i][h]
                    x[i][h]= xh_old + b/2
                    
                    # if x[i][h] > boundarymax :
                    #     x[i][h] = x[i-1][h]
                        
                    # if x[i][h] < -boundarymax :
                    #     x[i][h] = x[i-1][h]
                        
                    
                if minx == x[i][h] :
                    xj_old = x[i][j]
                    x[i][j] = xj_old + b/2
                    # if x[i][j] > boundarymax :
                    #     x[i][j] = x[i-1][j]
                        
                    # if x[i][j] < -boundarymax :
                    #     x[i][j] = x[i-1][j]
                        

                    
                    xh_old = x[i][h]
                    x[i][h]= xh_old - b/2
                    # if x[i][h] > boundarymax :
                    #     x[i][h] = x[i-1][h]
                        
                    # if x[i][h] < -boundarymax :
                    #     x[i][h] = x[i-1][h]
                        

            #if miny == y[i][j]:
                print("init", y[i][j])
            #y[i][j] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][j]) + pol(x,y,i,j,h,boundarymax)[1]
                y[i][j] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][j]-xj_old) + y[i][j]  #+ pol(x,y,i,j,h,boundarymax)[1]

                print("final", y[i][j])
            # if y[i][j] > boundarymax :
            #     y[i][j] = y[i-1][j]
                
            # if y[i][j] < -boundarymax :
            #     y[i][j] = y[i-1][j]
                print("init", y[i][j])   
            # y[i][h] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][h]) + pol(x,y,i,j,h,boundarymax)[1]
                y[i][h] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][h]-xh_old) + y[i][h]  #- pol(x,y,i,j,h,boundarymax)[1]

                print("final", y[i][j])
                # if y[i][h] > boundarymax :
                #     y[i][h] = y[i-1][h]
                    
                # if y[i][h] < -boundarymax :
                #     y[i][h] = y[i-1][h]
                # if miny == y[i][h]:
                    
                #     print("init", y[i][j])
                # #y[i][j] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][j]) + pol(x,y,i,j,h,boundarymax)[1]
                #     y[i][j] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][j]-xj_old) + y[i][j] - pol(x,y,i,j,h,boundarymax)[1]

                #     print("final", y[i][j])
                # # if y[i][j] > boundarymax :
                # #     y[i][j] = y[i-1][j]
                    
                # # if y[i][j] < -boundarymax :
                # #     y[i][j] = y[i-1][j]
                #     print("init", y[i][j])   
                # # y[i][h] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][h]) + pol(x,y,i,j,h,boundarymax)[1]
                #     y[i][h] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][h]-xh_old) + y[i][h]  + pol(x,y,i,j,h,boundarymax)[1]

                #print("final", y[i][j])
                
                FINALxj.append(x[i][j])
                FINALxh.append(x[i][h])
                FINALyj.append(y[i][j])
                FINALyh.append(y[i][h])     
            
        index.append(i)
        x[i+1][j] = (v*np.cos(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[i][j]
        y[i+1][j] = (v*np.sin(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[i][j]
        theta[i+1][j] = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta[i][j]
        #print("i,j", i, j)
        
        

     

    if i >0 :    
        figure = plt.figure(1)
        # plt.xlim(-a,a)
        # plt.ylim(-a,a)
        #plt.axis([-50,50,-50,50])
        plt.axvline(x = boundarymax, color = 'b', label = 'axvline - full height')
        plt.axvline(x = -boundarymax, color = 'b', label = 'axvline - full height')
        plt.axhline(y = boundarymax, color = 'b', label = 'axvline - full height')
        plt.axhline(y = -boundarymax, color = 'b', label = 'axvline - full height')
        #plt.Circle((x[i][j],y[i][j]),2, color='r')
        plt.scatter(x[i][:],y[i][:], label="iterations %i" %i, s=100, marker="x")
        plt.show()
        plt.legend()
        plt.pause(0.005)
        plt.clf()

#%%
point = 1
plt.figure(1)
plt.scatter(INITIALxj[point],INITIALyj[point],color="red", label="particle a, intial")   
plt.scatter(FINALxj[point],FINALyj[point],color="blue", label="particle a, final")
plt.scatter(INITIALxh[point],INITIALyh[point],color="green", label="particle b, intial")
plt.scatter(FINALxh[point],FINALyh[point],color="orange", label="particle b, final")
plt.legend()
plt.title("Check particle Colision")
plt.ylabel("y")
plt.xlabel("x")  

# fig = plt.figure(2)
# plt.title("YOOHOO%i"%i)
# plt.scatter(x,y, s=0.5)

#%%   9.3.b 

mu = 0
sigma = 1
iterations = 1000
deltaT = 0.010 ### 0.01 seems optimal
mean = 0
std = 1 
N = 60
R = 1e-6  # Difference between two random consecutive
                    # position/2
x=np.zeros([iterations,N])
y=x.copy()
theta=x.copy()
boundarymax = 1e-6  
for i in range(0,iterations):
    for j in range(0,N):
    	x[i][j] = (random()*2-1)*10e-6  #boundarymax #### put 1e-6 instead
    	y[i][j] = (random()*2-1)*10e-6  #boundarymax


index = []
plt.ion()
Dt = 0.1*(10**(-6))**2
Dr = 0.1
v = 3*10**(-6)


a = (boundarymax + boundarymax*0.3)
b =[]
c = []

for i in range(0,iterations-1):
    for j in range(0,N) :
        if x[i][j] > boundarymax :
            x[i][j] = -boundarymax
            
        if x[i][j] < -boundarymax :
            x[i][j] = boundarymax
            
        if y[i][j] > boundarymax :
            y[i][j] = -boundarymax
            
        if y[i][j] < -boundarymax :
            y[i][j] = boundarymax
            

        c = [np.sqrt((x[i][j]-x[i][h])**2+(y[i][j]-y[i][h])**2) for h in range(N)]

        for h in range(0,N) :  
            if (0.000000000000001 < c[h] < R) :
                b = (np.sqrt((x[i][j]-x[i][h])**2+(y[i][j]-y[i][h])**2))
                
                minx = min(x[i][j],x[i][h])
                miny = min(y[i][j],y[i][h])

                
                if minx == x[i][j] :
                    xj_old = x[i][j]
                    x[i][j] = xj_old - b/2
                        
                    xh_old = x[i][h]
                    x[i][h]= xh_old + b/2
                if minx == x[i][h] :
                    xj_old = x[i][j]
                    x[i][j] = xj_old + b/2
                    xh_old = x[i][h]
                    x[i][h]= xh_old - b/2

                y[i][j] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][j]-xj_old) + y[i][j]  #+ pol(x,y,i,j,h,boundarymax)[1]
                y[i][h] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][h]-xh_old) + y[i][h]  #- pol(x,y,i,j,h,boundarymax)[1]

            
        index.append(i)
        x[i+1][j] = (v*np.cos(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[i][j]
        y[i+1][j] = (v*np.sin(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[i][j]
        theta[i+1][j] = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta[i][j]
        #print("i,j", i, j)
        
        

     

    if i >0 :    
        figure = plt.figure(1)
        # plt.xlim(-a,a)
        # plt.ylim(-a,a)
        #plt.axis([-50,50,-50,50])
        plt.axvline(x = boundarymax, color = 'b', label = 'axvline - full height')
        plt.axvline(x = -boundarymax, color = 'b', label = 'axvline - full height')
        plt.axhline(y = boundarymax, color = 'b', label = 'axvline - full height')
        plt.axhline(y = -boundarymax, color = 'b', label = 'axvline - full height')
        #plt.Circle((x[i][j],y[i][j]),2, color='r')
        plt.scatter(x[i][:],y[i][:], label="iterations %i" %i, s=100)
        plt.show()
        plt.title("Trajectories of %i particles"%N)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.pause(0.005)
        plt.clf()

#%%   9.4.a

mu = 0
sigma = 1
iterations = 1000
deltaT = 0.01
mean = 0
std = 1 
N = 2
R = 1e-06  # Difference between two random consecutive
                    # position/2

Rc = 2*R + R/2
x=np.zeros([iterations,N])
y=x.copy()
theta=x.copy()
boundarymax = 1e-6
for i in range(0,iterations):
    for j in range(0,N):
    	x[i][j] = (random()*2-1)*boundarymax      #*10e-6
    	y[i][j] = (random()*2-1)*boundarymax      #*10e-6

index = []
plt.ion()
Dt = 0.1*(10**(-6))**2
Dr = 1
v0 = 20*10**(-6)

a = (boundarymax + boundarymax*0.3)
b =[]
c = []
for i in range(0,iterations-1):
    for j in range(0,N) :
        if x[i][j] > boundarymax :
            x[i][j] = -boundarymax           
        if x[i][j] < -boundarymax :
            x[i][j] = boundarymax            
        if y[i][j] > boundarymax :
            y[i][j] = -boundarymax        
        if y[i][j] < -boundarymax :
            y[i][j] = boundarymax
        c = [np.sqrt((x[i][j]-x[i][h])**2+(y[i][j]-y[i][h])**2) for h in range(N)]
        for h in range(0,N) :  
            if (0.000000000000001 < c[h] < R) :
                b = (np.sqrt((x[i][j]-x[i][h])**2+(y[i][j]-y[i][h])**2)) 
                minx = min(x[i][j],x[i][h])
                miny = min(y[i][j],y[i][h])
                if minx == x[i][j] :
                    xj_old = x[i][j]
                    x[i][j] = xj_old - b/2
                    xh_old = x[i][h]
                    x[i][h]= xh_old + b/2                  
                if minx == x[i][h] :
                    xj_old = x[i][j]
                    x[i][j] = xj_old + b/2
                    xh_old = x[i][h]
                    x[i][h]= xh_old - b/2              
                y[i][j] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][j]-xj_old) + y[i][j]  #+ pol(x,y,i,j,h,boundarymax)[1]
                y[i][h] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][h]-xh_old) + y[i][h]  #- pol(x,y,i,j,h,boundarymax)[1]  
        
        for h in range(0,N) :  
            if (2*R < c[h] < Rc) :
                v = v0*(R**2)/(c[h]**2)
            else :
                v = v0
        
            x[i+1][j] = (v*np.cos(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[i][j]
            y[i+1][j] = (v*np.sin(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[i][j]
            theta[i+1][j] = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta[i][j]
            #print("i,j", i, j)


    if i >0 :    
        figure = plt.figure(1)
        # plt.xlim(-a,a)
        # plt.ylim(-a,a)
        #plt.axis([-50,50,-50,50])
        plt.axvline(x = boundarymax, color = 'b')
        plt.axvline(x = -boundarymax, color = "b")
        plt.axhline(y = boundarymax, color = 'b')
        plt.axhline(y = -boundarymax, color = 'b')
        #plt.Circle((x[i][j],y[i][j]),2, color='r')
        plt.scatter(x[i][:],y[i][:], label="iterations %i" %i, s=100, marker=".")
        plt.show()
        plt.legend()
        plt.pause(0.005)
        plt.clf()
        
        
#%%  9.4.b


mu = 0
sigma = 1
iterations = 1000
deltaT = 0.01
mean = 0
std = 1 
N = 50
R = 1e-06  # Difference between two random consecutive
                    # position/2

Rc = 2*R + R/5
x=np.zeros([iterations,N])
y=x.copy()
theta=x.copy()
boundarymax = 100e-6
for i in range(0,iterations):
    for j in range(0,N):
    	x[i][j] = (random()*2-1)*boundarymax    #*10e-6
    	y[i][j] = (random()*2-1)*boundarymax    #*10e-6


index = []
plt.ion()
Dt = 0.1*(10**(-6))**2
Dr = 1
v0 = 20*10**(-6)

a = (boundarymax + boundarymax*0.3)
b =[]
c = []
for i in range(0,iterations-1):
    for j in range(0,N) :
        if x[i][j] > boundarymax :
            x[i][j] = -boundarymax           
        if x[i][j] < -boundarymax :
            x[i][j] = boundarymax            
        if y[i][j] > boundarymax :
            y[i][j] = -boundarymax        
        if y[i][j] < -boundarymax :
            y[i][j] = boundarymax
        c = [np.sqrt((x[i][j]-x[i][h])**2+(y[i][j]-y[i][h])**2) for h in range(N)]
        for h in range(0,N) :  
            if (0.000000000000001 < c[h] < R) :
                b = (np.sqrt((x[i][j]-x[i][h])**2+(y[i][j]-y[i][h])**2)) 
                minx = min(x[i][j],x[i][h])
                miny = min(y[i][j],y[i][h])
                if minx == x[i][j] :
                    xj_old = x[i][j]
                    x[i][j] = xj_old - b/2
                    xh_old = x[i][h]
                    x[i][h]= xh_old + b/2                  
                if minx == x[i][h] :
                    xj_old = x[i][j]
                    x[i][j] = xj_old + b/2
                    xh_old = x[i][h]
                    x[i][h]= xh_old - b/2              
                y[i][j] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][j]-xj_old) + y[i][j]  #+ pol(x,y,i,j,h,boundarymax)[1]
                y[i][h] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][h]-xh_old) + y[i][h]  #- pol(x,y,i,j,h,boundarymax)[1]  
        
        for h in range(0,N) :  
            if (2*R < c[h] < Rc) :
                v = v0*(R**2)/(c[h]**2)
                x[i+1][j] = (v*np.cos(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[i][j]
                y[i+1][j] = (v*np.sin(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[i][j]
                theta[i+1][j] = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta[i][j]

            else :
                v = v0
                x[i+1][j] = (v*np.cos(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[i][j]
                y[i+1][j] = (v*np.sin(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[i][j]
                theta[i+1][j] = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta[i][j]
            #print("i,j", i, j)


    if i >0 :    
        figure = plt.figure(1)
        # plt.xlim(-a,a)
        # plt.ylim(-a,a)
        #plt.axis([-50,50,-50,50])
        plt.axvline(x = boundarymax, color = 'b', label = 'axvline - full height')
        plt.axvline(x = -boundarymax, color = 'b', label = 'axvline - full height')
        plt.axhline(y = boundarymax, color = 'b', label = 'axvline - full height')
        plt.axhline(y = -boundarymax, color = 'b', label = 'axvline - full height')
        #plt.Circle((x[i][j],y[i][j]),2, color='r')
        plt.scatter(x[i][:],y[i][:], label="iterations %i" %i, s=100, marker=".")
        plt.show()
        plt.legend()
        plt.pause(0.005)
        plt.clf()
        
        
        
        
        
        
#%%  9.5.a


mu = 0
sigma = 1
iterations = 100
deltaT = 4
mean = 0
std = 1 
N = 100
R = 1e-06  # Difference between two random consecutive
                    # position/2

Rc = 2*R + R/2
x=np.zeros([iterations,N])
y=x.copy()
theta=x.copy()

index = []
plt.ion()
Dt = 0.1*(10**(-6))**2
Dr = 1
V = 50*10**(-6)
boundarymax = 10000e-6
a = (boundarymax + boundarymax*0.3)
b =[]
c = []
for i in range(0,iterations):
    for j in range(0,N):
    	x[i][j] = (random()*2-1)*boundarymax
    	y[i][j] = (random()*2-1)*boundarymax

count=0
count2=0
countlist=[i for i in range(0,iterations-1,10)]
v0 = [q for q in np.linspace(0,V,len(countlist))]
for i in range(0,iterations-1):
    count+=1
    if count in countlist :
        count2+=1
        print(count2)
    for j in range(0,N) :
        if x[i][j] > boundarymax :
            x[i][j] = -boundarymax           
        if x[i][j] < -boundarymax :
            x[i][j] = boundarymax            
        if y[i][j] > boundarymax :
            y[i][j] = -boundarymax        
        if y[i][j] < -boundarymax :
            y[i][j] = boundarymax
        c = [np.sqrt((x[i][j]-x[i][h])**2+(y[i][j]-y[i][h])**2) for h in range(N)]
        for h in range(0,N) :  
            if (0.000000000000001 < c[h] < R) :
                b = (np.sqrt((x[i][j]-x[i][h])**2+(y[i][j]-y[i][h])**2)) 
                minx = min(x[i][j],x[i][h])
                miny = min(y[i][j],y[i][h])
                if minx == x[i][j] :
                    xj_old = x[i][j]
                    x[i][j] = xj_old - b/2
                    xh_old = x[i][h]
                    x[i][h]= xh_old + b/2                  
                if minx == x[i][h] :
                    xj_old = x[i][j]
                    x[i][j] = xj_old + b/2
                    xh_old = x[i][h]
                    x[i][h]= xh_old - b/2              
                y[i][j] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][j]-xj_old) + y[i][j]  #+ pol(x,y,i,j,h,boundarymax)[1]
                y[i][h] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][h]-xh_old) + y[i][h]  #- pol(x,y,i,j,h,boundarymax)[1]  
        
        for h in range(0,N) :  
            if (2*R < c[h] < Rc) :
                v = v0[count2]*(R**2)/(c[h]**2)
                x[i+1][j] = (v*np.cos(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[i][j]
                y[i+1][j] = (v*np.sin(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[i][j]
                theta[i+1][j] = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta[i][j]
            else :
                v = v0[count2]
                x[i+1][j] = (v*np.cos(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[i][j]
                y[i+1][j] = (v*np.sin(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[i][j]
                theta[i+1][j] = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta[i][j]
            #print("i,j", i, j)

    w = v0[count2]*10e6
    if i >0 :    
        figure = plt.figure(1)
        # plt.xlim(-a,a)
        # plt.ylim(-a,a)
        #plt.axis([-50,50,-50,50])
        plt.title("Vo=%1.3f" %w)
        plt.axvline(x = boundarymax, color = 'b', label = 'axvline - full height')
        plt.axvline(x = -boundarymax, color = 'b', label = 'axvline - full height')
        plt.axhline(y = boundarymax, color = 'b', label = 'axvline - full height')
        plt.axhline(y = -boundarymax, color = 'b', label = 'axvline - full height')
        #plt.Circle((x[i][j],y[i][j]),2, color='r')
        plt.scatter(x[i][:],y[i][:], label="iterations %i" %i, s=100, marker=".")
        plt.show()
        plt.legend()
        plt.pause(0.005)
        plt.clf()
        
        
#%%  9.5.b


mu = 0
sigma = 1
iterations = 100
deltaT = 0.01
mean = 0
std = 1 
N = 50
R = 1e-06  # Difference between two random consecutive
                    # position/2

Rc = 2*R + R/2
x=np.zeros([iterations,N])
y=x.copy()
theta=x.copy()

index = []
plt.ion()
Dt = 0.1*(10**(-6))**2
Dr = 1
V = 50*10**(-6)
boundarymax = 10000e-6
a = (boundarymax + boundarymax*0.3)
b =[]
c = []
for i in range(0,iterations):
    for j in range(0,N):
    	x[i][j] = (random()*2-1)*boundarymax
    	y[i][j] = (random()*2-1)*boundarymax

count=0
count2=0
countlist=[i for i in range(0,iterations-1,10)]
v0 = [q for q in np.linspace(0,V,len(countlist))]
for i in range(0,iterations-1):
    count+=1
    if count in countlist :
        count2+=1
        print(count2)
    for j in range(0,N) :
        if x[i][j] > boundarymax :
            x[i][j] = -boundarymax           
        if x[i][j] < -boundarymax :
            x[i][j] = boundarymax            
        if y[i][j] > boundarymax :
            y[i][j] = -boundarymax        
        if y[i][j] < -boundarymax :
            y[i][j] = boundarymax
        c = [np.sqrt((x[i][j]-x[i][h])**2+(y[i][j]-y[i][h])**2) for h in range(N)]
        for h in range(0,N) :  
            if (0.000000000000001 < c[h] < R) :
                b = (np.sqrt((x[i][j]-x[i][h])**2+(y[i][j]-y[i][h])**2)) 
                minx = min(x[i][j],x[i][h])
                miny = min(y[i][j],y[i][h])
                if minx == x[i][j] :
                    xj_old = x[i][j]
                    x[i][j] = xj_old - b/2
                    xh_old = x[i][h]
                    x[i][h]= xh_old + b/2                  
                if minx == x[i][h] :
                    xj_old = x[i][j]
                    x[i][j] = xj_old + b/2
                    xh_old = x[i][h]
                    x[i][h]= xh_old - b/2              
                y[i][j] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][j]-xj_old) + y[i][j]  #+ pol(x,y,i,j,h,boundarymax)[1]
                y[i][h] = pol(x,y,i,j,h,boundarymax)[0]*(x[i][h]-xh_old) + y[i][h]  #- pol(x,y,i,j,h,boundarymax)[1]  
        
        for h in range(0,N) :  
            if (2*R < c[h] < Rc) :
                v = v0[count2]*(R**2)/(c[h]**2)
            else :
                v = v0[count2]
        
            x[i+1][j] = (v*np.cos(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + x[i][j]
            y[i+1][j] = (v*np.sin(theta[i][j])*deltaT + ((2*Dt)**0.5)*np.random.normal(0,1)*(deltaT**0.5)) + y[i][j]
            theta[i+1][j] = ((2*Dr)**0.5)*np.random.normal(0,1)*(deltaT**0.5) + theta[i][j]
            #print("i,j", i, j)

    w = v0[count2]*10e6
    if i >0 :    
        figure = plt.figure(1)
        # plt.xlim(-a,a)
        # plt.ylim(-a,a)
        #plt.axis([-50,50,-50,50])
        plt.title("Vo=%1.3f" %w)
        plt.axvline(x = boundarymax, color = 'b', label = 'axvline - full height')
        plt.axvline(x = -boundarymax, color = 'b', label = 'axvline - full height')
        plt.axhline(y = boundarymax, color = 'b', label = 'axvline - full height')
        plt.axhline(y = -boundarymax, color = 'b', label = 'axvline - full height')
        #plt.Circle((x[i][j],y[i][j]),2, color='r')
        plt.scatter(x[i][:],y[i][:], label="iterations %i" %i, s=100, marker=".")
        plt.show()
        plt.legend()
        plt.pause(0.005)
        plt.clf()
