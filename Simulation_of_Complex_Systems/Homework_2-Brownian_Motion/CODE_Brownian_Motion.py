import numpy as np
import random
import matplotlib.pyplot as plt
from copy import copy
import pandas as pd
from statistics import mean 
import scipy as sp
import pylab
from fractions import Fraction
def square(list):
    return [i ** 2 for i in list]

def slice_per(source, step):
    return [source[i::step] for i in range(step)]

#%% 5.1.a Equal Probability

iterations = 100
epochs = 100
posEnd = []
distribution=[]
for j in range(0,epochs) :
    position=0
    posList = []
    for i in range(0,iterations) :
        rnd = np.sign(random.uniform(0,1) -0.5)
        position = position + rnd
        posList.append(position)
        distribution.append(rnd)
    posEnd.append(position)

            
    fig = plt.figure(1)
    plt.plot(posList, color="blue")
    plt.title("Trajectories, Equal Probability")
    plt.xlabel("x(t)")
    plt.ylabel("t-steps")
    
    plt.figure(2)
    plt.hist(posEnd, bins = 100,color="blue")
    plt.title("Final_Position_Equal_Probability")
    plt.xlabel("%i iterations"%iterations)
    plt.ylabel("p(x) after %i iterations"%iterations)
    
    fig = plt.figure(3)
    plt.hist(distribution, bins = 100,color="blue")
    plt.title("Equal probability")
    plt.xlabel("\u0394x")
    plt.ylabel("p(\u0394x)")
    
        
   #%%  5.1.a Gaussian
mu = 0
sigma = 1
iterations = 100
epochs = 50
posEnd = []

distribution = []

for j in range(0,epochs) :
    position=0
    posList = []
    for i in range(0,iterations) :
        posList.append(position)
        position = position + np.sign(random.gauss(mu, sigma))
        
        distribution.append(random.gauss(mu, sigma))
    posEnd.append(position)
    
    fig = plt.figure(1)
    plt.plot(posList, color="green")
    plt.title("Trajectories_Gauss_Distribution")
    plt.xlabel("x(t)")
    plt.ylabel("t-steps")
    fig = plt.figure(2)
    #plt.hist(posList, bins = 100,density=True)
    plt.hist(posEnd, bins = 100, color="green")
    plt.title("Final_Position_Gauss_Distribution")
    plt.xlabel("%i iterations"%iterations)
    plt.ylabel("p(x) after %i iterations"%iterations)
    fig = plt.figure(3)
    plt.hist(distribution, bins = 100, color="green")
    plt.title("Gaussian_Distribution_mean=0_var=1")
    plt.xlabel("\u0394x")
    plt.ylabel("p(\u0394x)")



#%%

p1 = 1/3
p2 = 1/3 
p3 = 1/3
distribution = []
distribution.append([p1,p1+p2, p1+p2+p3])
iterations = 50
epochs = 1000

posEnd = []

for j in range(0,epochs) :
    position=0
    posList = []
    for i in range(0,iterations) :
        rnd = random.uniform(0,1)
        if rnd < p1:
            posList.append(position)
            position = position - 1
            
        
        if  p1< rnd< p1+p2:
            posList.append(position)
            position = position + (1-np.sqrt(3))/2
            
        
        if p1+p2< rnd <p1+p2+p3:
            posList.append(position)
            position = position + (1+np.sqrt(3))/2
        posEnd.append(position)

            
    fig = plt.figure(1)
    plt.plot(posList, color="orange")
    plt.title("Trajectories_Asymetric_Distribution")
    plt.xlabel("x(t)")
    plt.ylabel("t-steps")
    fig = plt.figure(2)
    #plt.hist(posList, bins = 100,density=True)
    plt.hist(posEnd, bins = 100, color="orange")
    plt.title("Final_Position_Asymetric_Distribution")
    plt.xlabel("%i iterations"%iterations)
    plt.ylabel("p(x) after %i iterations"%iterations)
fig = plt.figure(3)
plt.hist(distribution, bins = 100, color="orange")
plt.title("Asymetric_Distribution")
plt.xlabel("\u0394x")
plt.ylabel("p(\u0394x)")

#%%  5.2.a


x = np.array([])
t = [0.01,0.05,0.1]
mu = 0
sigma = 1
#iterations = np.linspace(0,1000,1001)
iterations = 50
epochs = 50

distribution = []

c = ["blue","red","orange"]
for k in t:
    for j in range(0,epochs) :
        position=0
        posList = []
        for i in range(1,iterations+2) :
            posList.append(position)
            position = position + random.gauss(mu, sigma)*np.sqrt(k)
            
            distribution.append(random.gauss(mu, sigma))
                
        fig = plt.figure(t.index(k))
        plt.plot(posList, color=c[t.index(k)])
        plt.title("Trajectories, Gauss Distribution, t=%1.2f"%k)
        plt.xlabel("x(t)")
        plt.ylabel("t-steps")


#%%  5.2.b TRASH


x = np.array([])
t = [0.01,0.05,0.1]
mu = 0
sigma = 1
#iterations = np.linspace(0,1000,1001)
iterations = 5
epochs = 1000 #int(10E2)

distribution = []
MSD1 = []
MSD1av = []
MSD2= []
MSD2av = []
MSD3 = []
MSD3av = []

A = []
Index = [i for i in range(0,epochs*iterations)]

c = ["blue","red","orange"]

A = np.empty((epochs, iterations))
B = np.empty((epochs, iterations))
C = np.empty((epochs, iterations))

# print(A)
# print(A.shape)

for k in t:
    for j in range(0,epochs) :
        position=0
        posList = []
        index = []
        
        for i in range(0,iterations) :
            index.append(i)
            position = position + random.gauss(mu, sigma)*np.sqrt(k)

            
            if k==0.01:
                MSD1 =(abs(position-0)**2)
                A[j,i] = MSD1
            if k==0.05:
                MSD2=(abs(position-0)**2)
                B[j,i] = MSD2
                
            if k==0.1:
                MSD3=(abs(position-0)**2)
                C[j,i] = MSD3
        
        
Aav = np.mean(A, axis=0)
Bav = np.mean(B, axis=0)
Cav = np.mean(C, axis=0)
        
fig = plt.figure(5)
plt.plot(Aav)
plt.plot(Bav)
plt.plot(Cav)
plt.title("MSD1")
# fig = plt.figure(6)
# plt.plot(index,MSD2av)
# plt.title("MSD2")
# fig = plt.figure(7)
# plt.plot(index,MSD3av)
plt.title("MSD3")
    

#%%  5.2.b GOOOOOOOOD


x = np.array([])
t = [0.01,0.05,0.1]
mu = 0
sigma = 1
#iterations = np.linspace(0,1000,1001)
iterations = 50
epochs = int(10E4)

distribution = []
MSD1 = []
MSD1av = []
MSD2= []
MSD2av = []
MSD3 = []
MSD3av = []

A = []
Index = [i for i in range(0,epochs*iterations)]

c = ["blue","red","orange"]

A = np.empty((epochs, iterations))
B = np.empty((epochs, iterations))
C = np.empty((epochs, iterations))

D = []
E = []
F = []


for k in t:
    for j in range(0,epochs) :
        position=0
        posList = []
        index = []
        
        for i in range(0,iterations) :
            index.append(i)
            position = position + random.gauss(mu, sigma)*np.sqrt(k)

            
            if k==0.01:
                A[j,i] =(abs(position-0)**2)
                
            if k==0.05:
                B[j,i]=(abs(position-0)**2)
                
                
            if k==0.1:
                C[j,i] =(abs(position-0)**2)
                
                

dfA = pd.DataFrame(A)
dfB = pd.DataFrame(B)
dfC = pd.DataFrame(C)

A = dfA.mean(axis=0)
B = dfB.mean(axis=0)
C = dfC.mean(axis=0)
        
        
fig = plt.figure(5)
plt.plot(index,A, label="MSD_delta_t=0.01")
plt.plot(index,B, label="MSD_delta_t=0.05")
plt.plot(index,C, label="MSD_delta_t=0.1")
plt.legend()
plt.title("MSDs for 3 delta_t")
plt.xlabel("index (=iteration)")
plt.ylabel("Averaged MSD over 10e4 epochs")


#%%   5.3.a
mu = 0
sigma = 1
R = 1*10**(-6)
m = 1.11e-14
eta = 0.001
gamma=6*np.pi*R*eta
T=300 #K
kB = 1.380649*10**(-23)

tau = m/gamma
#tau = m/gamma
In = []
position = 0
epsilon = gamma
amountofsteps = 100  #=100 0r 100*100

#iterations = 100
k=tau/100  #StepSize in constant
A = []
B=[]
A.append(0)
A.append(0)
B.append(0)
B.append(0)
xm = [] 
x = []
deltaT = k
count=2
Kb=kB
  #####   COMMENT / UNCOMMENT FOR LOOP !!! #######
#for i in np.linspace(0,tau,amountofsteps):
for i in np.linspace(0,100*tau,100*amountofsteps):  #Change tau into 100*tau    
    In.append(i/tau)
    rnd = random.gauss(mu, sigma)

        
    xm.append((2+deltaT*(epsilon/m))/(1+deltaT*(epsilon/m))*A[count-1]-A[count-2]/(1+deltaT*(epsilon/m))+ np.sqrt(2*Kb*T*epsilon)/(m*(1+deltaT*(epsilon/m)))*np.power(deltaT,3/2)*rnd)
    A.append(xm[-1])
    x.append(B[count-1]+(np.sqrt((2*Kb*T*deltaT)/epsilon))*rnd)
    B.append(x[-1])
    count=count+1 


plt.figure(1)
plt.plot(In,x,color="blue", label ="without Intertia" )
plt.xlabel("Iteration (timescale big compared to tau)")
plt.ylabel("Position x [nm]")
plt.plot(In,xm,color="red", label ="with Intertia")
plt.legend()
plt.title("Trajectories, Brownian particle, Langevin equation with and without inertia")
plt.show()


#%%  5.3.b



mu = 0
sigma = 1
R = 1*10**(-6)
m = 1.11e-14
eta = 0.001
gamma=6*np.pi*R*eta
T=300 #K
kB = 1.380649*10**(-23)

tau = m/gamma
#tau = m/gamma
In = []
position = 0
epsilon = gamma
amountofsteps = 100  #=100 0r 100*100

k=tau/100  #StepSize in constant
A = []
B=[]

xm = [] 
x = []
deltaT = k
count=2
Kb=kB

df = pd.DataFrame()
NSM = []

C = []
D = []
Aav = []
Bav=[]

for j in range(0,1000):
    In = []
    count=2
    xm = [] 
    x = []
    A = []
    B=[]
    A.append(0)
    A.append(0)
    B.append(0)
    B.append(0)
    for i in np.linspace(0,j*tau,j*amountofsteps):
        
        In.append(i/tau)
        rnd = random.gauss(mu, sigma)
    
            
        #xm.append((2+deltaT*(epsilon/m))/(1+deltaT*(epsilon/m))*A[count-1]-A[count-2]/(1+deltaT*(epsilon/m))+ np.sqrt(2*Kb*T*epsilon)/(m*(1+deltaT*(epsilon/m)))*np.power(deltaT,3/2)*rnd)
        #A.append(xm[-1])
        xm = (2+deltaT*(epsilon/m))/(1+deltaT*(epsilon/m))*A[count-1]-A[count-2]/(1+deltaT*(epsilon/m))+ np.sqrt(2*Kb*T*epsilon)/(m*(1+deltaT*(epsilon/m)))*np.power(deltaT,3/2)*rnd
        A.append(xm)
        
        #x.append(B[count-1]+(np.sqrt((2*Kb*T*deltaT)/epsilon))*rnd)
        #B.append(x[-1])
        x=(B[count-1]+(np.sqrt((2*Kb*T*deltaT)/epsilon))*rnd)
        B.append(x)
        count=count+1 
    
    
    
    C.append(square(A[2:]))
    D.append(square(B[2:]))

del C[0]
del D[0]
  

dfxm = pd.DataFrame(C)
dfx = pd.DataFrame(D)
MSDxm = dfxm.mean()
MSDx = dfx.mean()
   



plt.figure(1)
plt.xlabel("t/tau")
plt.ylabel("MSD")
plt.loglog(In,MSDxm,color="blue", label="Trajectory with Inertia")
plt.loglog(In,MSDx,color="red", label="Trajectory without Inertia")
plt.legend()
plt.title("MSD with and without Inertia")

#%%  5.3.c   Part 1

mu = 0
sigma = 1
R = 1*10**(-6)
m = 1.11e-14
eta = 0.001
gamma=6*np.pi*R*eta
T=300 #K
kB = 1.380649*10**(-23)

tau = m/gamma
#tau = m/gamma
In = []
position = 0
epsilon = gamma
amountofsteps = 100  #=100 0r 100*100

#iterations = 100
k=tau/100  #StepSize in constant
A = []
B=[]
Av= []
Bv=[]
A.append(0)
A.append(0)
B.append(0)
B.append(0)
xm = [] 
x = []
deltaT = k
count=2
Kb=kB

chunksize = int(10*amountofsteps)
end = 100*chunksize
chunck = [i for i in range(0,end,chunksize)]
#chunck = [i for i in range(0,100*amountofsteps,amountofsteps)]
countChunck = 0
initxm=0
initx=0
for i in np.linspace(0,100*tau,100*amountofsteps): #Change tau into 100*tau    
        countChunck += 1 
        In.append(i/tau)
        rnd = random.gauss(mu, sigma)
        xm.append((2+deltaT*(epsilon/m))/(1+deltaT*(epsilon/m))*A[count-1]-A[count-2]/(1+deltaT*(epsilon/m))+ np.sqrt(2*Kb*T*epsilon)/(m*(1+deltaT*(epsilon/m)))*np.power(deltaT,3/2)*rnd)
        A.append(xm[-1])
        Av.append(abs(xm[-1]-initxm)**2)
        x.append(B[count-1]+(np.sqrt((2*Kb*T*deltaT)/epsilon))*rnd)
        B.append(x[-1])
        Bv.append(abs(x[-1]-initx)**2)
        count=count+1 
        if countChunck in chunck:
            initxm = xm[-1]
            initx = x[-1]
            print(countChunck)
        
            

slicesize=chunksize
A = slice_per(Av, slicesize)
B = slice_per(Bv, slicesize)


dfAav = pd.DataFrame(A)
dfBav = pd.DataFrame(B)
MSDxm = dfAav.mean(axis=1)
MSDx = dfBav.mean(axis=1)



plt.figure(1)
plt.plot(In,MSDxm*10**(19),color="blue", label="MSD_xm_1 (w/ Intertia) (.e-19)")
plt.plot(In,MSDx*10**(19),color="red", label="MSD_x_1 (no Intertia) (.e-19)")
#plt.title("time-averaged MSDs (averaged over a single, very long trajectory)")
plt.legend()
plt.title("MSD averaged over 1 long trajectory")
plt.xlabel("Iterations")
plt.ylabel("Averaged MSD (.e-19)")

#%%

MSDxm1 = MSDxm
MSDx1 = MSDx

#%%  5.3.c part 2

mu = 0
sigma = 1
R = 1*10**(-6)
m = 1.11e-14
eta = 0.001
gamma=6*np.pi*R*eta
T=300 #K
kB = 1.380649*10**(-23)

tau = m/gamma
#tau = m/gamma
In = []
position = 0
epsilon = gamma
amountofsteps = 100

k=tau/100  #StepSize in constant
A = []
B=[]

xm = [] 
x = []
deltaT = k
count=2
Kb=kB

df = pd.DataFrame()
NSM = []

C = []
D = []
Aav = []
Bav=[]

for j in range(0,100):
    In = []
    count=2
    xm = [] 
    x = []
    A = []
    B=[]
    A.append(0)
    A.append(0)
    B.append(0)
    B.append(0)
    for i in np.linspace(0,100*tau,100*amountofsteps):
        
        In.append(i/tau)
        rnd = random.gauss(mu, sigma)

        xm = (2+deltaT*(epsilon/m))/(1+deltaT*(epsilon/m))*A[count-1]-A[count-2]/(1+deltaT*(epsilon/m))+ np.sqrt(2*Kb*T*epsilon)/(m*(1+deltaT*(epsilon/m)))*np.power(deltaT,3/2)*rnd
        A.append(xm)

        x=(B[count-1]+(np.sqrt((2*Kb*T*deltaT)/epsilon))*rnd)
        B.append(x)
        count=count+1 
     
    
    C.append(square(list(map(abs, A[2:]))))
    D.append(square(list(map(abs, B[2:]))))

  

dfxm = pd.DataFrame(C)
dfx = pd.DataFrame(D)
MSDxm = dfxm.mean(axis=0)
MSDx = dfx.mean(axis=0)
   
fig1 = plt.figure(1)

plt.plot(MSDxm*10**(17),color="blue", label='MSDxm (w/ intertia)(.e-17)')
plt.plot(MSDx*10**(17),color="red", label='MSDx (no Intertia) (.e-17)')
plt.xlabel("iterations")
plt.ylabel("MSD averaged (.e-17)")
plt.legend()
plt.title("MSD averaged over many different trajectories")



#%%


mu = 0
sigma = 1
T = 300                
kB = 1.38e-23          
R = 1e-6               
kx = 1e-6              
ky = 5e-7              
gamma = 0.006*np.pi*R 
N = int(1e5)          
dt = 1e-3             

x = []   
y = []
x.append(0)
y.append(0) 
print(x)

for i in range(N-1):
    randx = random.gauss(mu, sigma) 
    x.append(x[i] - kx*x[i]*dt/gamma + np.sqrt(2*kB*T*dt/gamma)*randx)   
    randy = random.gauss(mu, sigma)
    y.append(y[i] - ky*y[i]*dt/gamma + np.sqrt(2*kB*T*dt/gamma)*randy)


plt.figure(1)
plt.title("2D trajectory of a Brownian particle held in a harmonic potential")
plt.scatter(x,y, s=0.1)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
#%%

mu = 0
sigma = 1
T = 300                
kB = 1.38e-23          
R = 1e-6               
kx = 1e-6              
ky = 5e-7              
gamma = 0.006*np.pi*R 
N = int(1e5)          
dt = 1e-3             

x = []   
y = []
x.append(0)
y.append(0) 

for i in range(N-1):
    randx = random.gauss(mu, sigma)
    x.append(x[i] - kx*x[i]*dt/gamma + np.sqrt(2*kB*T*dt/gamma)*randx)   
    randy = random.gauss(mu, sigma)
    y.append(y[i] - ky*y[i]*dt/gamma + np.sqrt(2*kB*T*dt/gamma)*randy)
    
x = np.array(x)
y = np.array(y)
Ux = (1/2)*kx*x**2
Uy = (1/2)*ky*y**2
pX = np.exp(-Ux/(kB*T))
pY = np.exp(-Uy/(kB*T))


weightsX = np.ones_like(x)/float(len(x))

plt.figure(1)
plt.xlabel("x [m]")
plt.ylabel("p(x)")
plt.plot(x,pX, label="p(x)")
plt.title("p(x) Vs x")
plt.legend()

plt.figure(2)
plt.xlabel("x [m]")
plt.ylabel("Position density")
plt.title("Histogram of x")
plt.legend()
plt.hist(x, bins = 100, weights=weightsX, label="Histogram of positions x")


weightsY = np.ones_like(y)/float(len(y))
plt.figure(3)
plt.xlabel("y [m]")
plt.ylabel("p(y)")
plt.plot(y,pY, label="p(y) Vs y")
plt.title("p(y) Vs y")
plt.legend()

plt.figure(4)
plt.xlabel("y [m]")
plt.ylabel("Position density")
plt.title("Histogram of y")
plt.legend()
plt.hist(y, bins = 100, weights=weightsY, label="Histogram of positions y")


#%%

mu = 0
sigma = 1
T = 300                
kB = 1.38e-23          
R = 1e-6               
kx = 1e-6              
ky = 5e-7              
gamma = 0.006*np.pi*R 
N = int(1e5)          
dt = 1e-3             

x = []   
y = []
x.append(0)
y.append(0) 

for i in range(N-1):
    randx = random.gauss(mu, sigma)
    x.append(x[i] - kx*x[i]*dt/gamma + np.sqrt(2*kB*T*dt/gamma)*randx)   
    randy = random.gauss(mu, sigma)
    y.append(y[i] - ky*y[i]*dt/gamma + np.sqrt(2*kB*T*dt/gamma)*randy)
    
x = np.array(x)
y = np.array(y)
Ux = (1/2)*kx*x**2
Uy = (1/2)*ky*y**2
pX = np.exp(-Ux/(kB*T))
pY = np.exp(-Uy/(kB*T))


weightsX = np.ones_like(x)/float(len(x))

plt.figure(1)
plt.plot(x,pX)
plt.hist(x, bins = 100, weights=weightsX)
#plt.hist(x,bins = 100, weights=weightsX)

weightsY = np.ones_like(y)/float(len(y))
plt.figure(2)
plt.plot(y,pY)
plt.hist(y, bins = 100, weights=weightsY)
#plt.hist(y, bins = 100, weights=weightsY)

plt.figure(3)
plt.plot(y,pX*pY)


#%%

mu = 0
sigma = 1
T = 300                
kB = 1.38e-23          
R = 1e-6               
kx = 1e-6              
ky = 5e-7              
gamma = 0.006*np.pi*R 
N = int(1e5)          
dt = 1e-3             

x = []   
y = []
x.append(0)
y.append(0) 
A = []
B= []
Cx = []
Cy = []
it = N-1
In = []
for i in range(it):
    In.append(i)
    randx = random.gauss(mu, sigma)
    x.append(x[i] - kx*x[i]*dt/gamma + np.sqrt(2*kB*T*dt/gamma)*randx)   
    A.append(x[-1]*x[-2])
    Cx.append(((kB*T)/kx)*np.exp(-(kx*(i/(it)))/(gamma)))
    randy = random.gauss(mu, sigma)
    y.append(y[i] - ky*y[i]*dt/gamma + np.sqrt(2*kB*T*dt/gamma)*randy)
    B.append(y[-1]*y[-2])
    Cy.append(((kB*T)/ky)*np.exp(-(ky*(i/(it)))/(gamma)))
    
A = np.array(A)
B = np.array(B)

A = np.mean(A)
B = np.mean(B)   

x = np.array(x)
y = np.array(y)

plt.figure(1)
plt.plot(Cx, color="blue", label="Cx")
plt.plot(Cy, color="green", label="Cy")
plt.legend()
plt.xlabel("time")
plt.ylabel("Cx, Cy")
plt.title("Autocorrelation functions Cx and Cy")





    
    
