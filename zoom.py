# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 09:10:39 2018

@author: ibrahim
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:11:33 2018

@author: ibrahim
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def QlL(l,L,n,Xmax=1.0,Ymax=1.0):
    rmin = l
    rmax = 1.0*L
    umax = (1.0/(rmin**2)-1)
    umin = (1.0/(rmax**2)-1)
    lbda = (2.0/np.pi)*(umax-umin)*(Xmax+1)*(Ymax+1)
    N = np.random.poisson(lbda)
    ui = (umax-umin)*np.random.uniform(size=N)+umin
    ri = 1.0/np.sqrt(umax-ui+umin)
    xi = -0.5 + np.random.uniform(size=N) * (Xmax+1)
    yi = -0.5 + np.random.uniform(size=N) * (Ymax+1)
    
    sigma = np.sqrt(0.08)
    mu = -0.5*0.08
    logWi = np.random.randn(N)*sigma+mu
    
    x = np.linspace(0, Xmax, n)
    y = np.linspace(0, Ymax, n)
    X, Y = np.meshgrid(x, y)
    
    logQ = np.zeros((n,n))
    for i in range(N):
        slct = ((((X-xi[i])/ri[i])**2+((Y-yi[i])/ri[i])**2)<0.25)
        logQ[slct] = logQ[slct] + logWi[i]
    Q = np.exp(logQ)
    C = (rmin/rmax)**(np.exp(mu+0.5*sigma**2)-1)
    return Q/C

def Ql(l,n,Xmax=1.0,Ymax=1.0):
    
    return QlL(l,1.0,n,Xmax,Ymax)

def interpolx2(Im,n,Xmax,Ymax):
    x = np.linspace(0, Xmax, n)
    y = np.linspace(0, Ymax, n)
    X, Y = np.meshgrid(x, y)
    
    x_new = np.linspace(0, Xmax, n*2)
    y_new = np.linspace(0, Ymax, n*2)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    
    f = interpolate.interp2d(x, y, Im)
    return f(x_new,y_new)

def integrate(Qr,H):
    n = Qr.shape[0]
    x = np.concatenate((np.arange(0,1+n/2),np.arange(-n/2+1,0)))
    U,V = np.meshgrid(x,x)
    S = U**2+V**2
    S[0,0] = 1.0
    return np.real(np.fft.ifft2(np.fft.fft2(Qr)/(S**H)))

def derivate(Qr,H):
    n = Qr.shape[0]
    x = np.concatenate((np.arange(0,1+n/2),np.arange(-n/2+1,0)))
    U,V = np.meshgrid(x,x)
    S = U**2+V**2
    S[0,0] = 1.0
    return np.real(np.fft.ifft2(np.fft.fft2(Qr)*(S**H)))

def normalize(Inew,Iold):
    n_new = Inew.shape[0]
    n = Iold.shape[0]
    if((Inew.shape[1]!=n_new)or(Iold.shape[1]!=n)or((n_new/n)!=2**(int(np.log2(n_new/n))))):
        print("Error size of images")
        return
    for i in range(n):
        for j in range(n):
            tot_intensity = Iold[i,j]
            Inew[2*i:2*(i+1),2*j:2*(j+1)] = tot_intensity*Inew[2*i:2*(i+1),2*j:2*(j+1)]/np.sum(Inew[2*i:2*(i+1),2*j:2*(j+1)])
    return Inew

def superresolution(Im,H,alpha):
    n=Im.shape[0]
    Xmax=1.0
    r1=1.0/n
    I_interp = interpolx2(Im,n,Xmax,Xmax)
    plt.figure()
    plt.imshow(I_interp,cmap='magma')
    plt.plot()
    
    J1 = derivate(I_interp,H)
    
    Q_new = QlL(0.5*r1,r1,2*n,Xmax,Xmax)
    J2 = J1*Q_new+alpha*1*(Q_new-Q_new.mean())
    
    K2 = Im.mean() + integrate(J2,H)
    K2 = np.maximum(K2,np.zeros_like(K2))
    
    Inew = normalize(K2,Im)
    
    return Inew


# Example of use on a toy problem. This can of course be used with real data but one would have to estimate alpha and H (as well as devise a good distribution for the W_i). Look at the report for more details.

Xmax=1.0
n=32
Q = Ql(1.0/n,n,Xmax,Xmax)
alpha = 8
H=0.7
I0 = 23
I = I0+alpha*integrate(Q-Q.mean(),H)
plt.figure()
plt.imshow(I,cmap='magma')
plt.plot()

I2 = superresolution(I,H,alpha)
plt.figure()
plt.imshow(I2,cmap='magma')
plt.plot()


I4 = superresolution(I2,H,alpha)
plt.figure()
plt.imshow(I4,cmap='magma')
plt.plot()


I8 = superresolution(I4,H,alpha)
plt.figure()
plt.imshow(I8,cmap='magma')
plt.plot()

I16 = superresolution(I8,H,alpha)
plt.figure()
plt.imshow(I16,cmap='magma')
plt.plot()


I32 = superresolution(I16,H,alpha)
plt.figure()
plt.imshow(I32,cmap='magma')
plt.plot()



# Visualizing a small chunk of the image at different zoom levels
a = 8

im_paszoom = I[a:a+16,a:a+16]
plt.figure()
plt.imshow(im_paszoom,cmap='magma')
plt.plot()

im_zoom = I4[a*4:(a+16)*4,a*4:(a+16)*4]
plt.figure()
plt.imshow(im_zoom,cmap='magma')
plt.plot()

im_zoom = I8[a*8:(a+16)*8,a*8:(a+16)*8]
plt.figure()
plt.imshow(im_zoom,cmap='magma')
plt.plot()

im_zoom = I16[a*16:(a+16)*16,a*16:(a+16)*16]
plt.figure()
plt.imshow(im_zoom,cmap='magma')
plt.plot()

