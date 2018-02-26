# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:11:33 2018

@author: ibrahim
"""

import numpy as np
import matplotlib.pyplot as plt

n=256

# Different Kernels for the generation of textures
def f1(x,y):
    return ((x**2+y**2)<0.25)

def f2(x,y):
    return (np.abs(x)<0.5)*(np.abs(y)<0.5)

def f3(x,y):
    a=1.0
    b=10.0
    M = np.sqrt((x/a)**2+(y/b)**2)
    res = np.zeros_like(M)
    select = M<0.5
    res[select] = np.cos(np.pi*M[select])
    return res

def f4(x,y):
    a=1.0
    b=1.0
    M = np.sqrt((x/a)**2+(y/b)**2)
    res = np.zeros_like(M)
    select = M<0.5
    res[select] = np.cos(np.pi*M[select])**2
    return res

def f5(x,y):
    a=1.0
    b=1.0
    M = np.sqrt((x/a)**2+(y/b)**2)
    res = np.zeros_like(M)
    select = M<0.5
    res[select] = np.cos(np.pi*np.exp(M[select]))**2
    return res

def f6(x,y):
    a=10.0
    b=1.0
    M = np.sqrt((x/a)**2+(y/b)**2)
    res = np.zeros_like(M)
    select = M<0.5
    res[select] = np.cos(np.pi*np.exp(np.exp(M[select])))**2
    return res

def f7(x,y):
    a=1.0
    b=1.0
    M = np.sqrt((x/a)**2+(y/b)**2)
    res = np.zeros_like(M)
    select = M<0.5
    res[select] = np.exp(-1/(1-(2*M[select])**4))
    return res

#Function generating an array of Q values for a certain f
def Q(f):
    rmin = 1.0/n
    Xmax = 1.0
    Ymax = 1.0
    umax = (1.0/(rmin**2)-1)
    lbda = (2.0/np.pi)*umax*(Xmax+1)*(Ymax+1)
    N = np.random.poisson(lbda)
    ui = umax*np.random.uniform(size=N)
    ri = 1.0/np.sqrt(umax-ui)
    xi = -0.5 + np.random.uniform(size=N) * (Xmax+1)
    yi = -0.5 + np.random.uniform(size=N) * (Ymax+1)
    
    sigma = np.sqrt(0.08)
    mu = -0.5*0.008
    logWi = np.random.randn(N)*sigma+mu
    
#    Uncomment to use another distribution for W_i
#    T=0.08
#    logWi = np.log(np.random.uniform(size=N))*T+np.log(T+1)


	
    x = np.linspace(0, Xmax, n)
    y = np.linspace(0, Ymax, n)
    X, Y = np.meshgrid(x, y)
    
    logQ = np.zeros((n,n))
    for i in range(N):
        tmp = f((X-xi[i])/ri[i],(Y-yi[i])/ri[i])
        slct = (tmp!=0)
        logQ[slct] = logQ[slct] + tmp[slct]*logWi[i]
    Q = np.exp(logQ)
    C = rmin**(np.exp(mu+0.5*sigma**2)-1)
    return Q/C



# alpha-integration of a certain 2D-array Qr
def integrate(Qr,alpha):
    n = Qr.shape[0]
    x = np.concatenate((np.arange(0,1+n/2),np.arange(-n/2+1,0)))
    U,V = np.meshgrid(x,x)
    S = U**2+V**2
    S[0,0] = 1.0
    return np.real(np.fft.ifft2(np.fft.fft2(Qr)/(S**alpha)))


#Different examples for texture generation
Qres1 = integrate(Q(f1),0.05)
plt.figure()
plt.imshow(Qres1,cmap='gray')
plt.show()

#Qres3 = integrate(Q(f2),0.05)
#plt.figure()
#plt.imshow(Qres3,cmap='gray')
#plt.show()

#Qres4 = integrate(Q(f3),0.05)
#plt.figure()
#plt.imshow(Qres4,cmap='gray')
#plt.show()

#Qres5 = integrate(Q(f4),0.05)
#plt.figure()
#plt.imshow(Qres5,cmap='gray')
#plt.show()
#
#Qres6 = integrate(Q(f5),0.05)
#plt.figure()
#plt.imshow(Qres6,cmap='gray')
#plt.show()

#Qres7 = integrate(Q(f6),0.05)
#plt.figure()
#plt.imshow(Qres7,cmap='gray')
#plt.show()

#Qres8 = integrate(Q(f7),0.05)
#plt.figure()
#plt.imshow(Qres8,cmap='gray')
#plt.show()
#
#Qres9 = integrate(Q(f1),1.0)
#plt.figure()
#plt.imshow(Qres9,cmap='gray')
#plt.show()
#
#Qres10 = integrate(Q(f1),10.0)
#plt.figure()
#plt.imshow(Qres10,cmap='gray')
#plt.show()


#One can also make colored textures by combining different channels
#Qres1 = integrate(Q(f1),0.1)
#Qres2 = integrate(Q(f1),0.04)
#Qres3 = integrate(Q(f1),0.04)
#Image = np.zeros((n,n,3))
#Image[:,:,0] = 0.3*Qres1
#Image[:,:,1] = 0.3*Qres2
#Image[:,:,2] = 0.3*Qres3
#plt.imshow(Image)


