import pdb

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
from sklearn.datasets import fetch_openml
import scipy
from scipy.spatial.distance import cdist
import pandas as pd
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
import os
import scipy.misc
import matplotlib.image as mpimg

def read_img(fpath, grayscale=False, bbox=None):
    if grayscale:
        img = ImageOps.grayscale(Image.open(fpath))
    else:
        img = Image.open(fpath)
    if bbox is not None:
        return np.asarray(img.crop(bbox).reduce(2))
    else:
        return np.asarray(img.reduce(2))
    
def do_pca(X, n_pca):
    print('Applying PCA')
    pca = PCA(n_components=n_pca, random_state=42)
    pca.fit(X)
    print('explained_variance_ratio:', pca.explained_variance_ratio_)
    print('sum(explained_variance_ratio):', np.sum(pca.explained_variance_ratio_))
    print('singular_values:', pca.singular_values_)
    X = pca.fit_transform(X)
    return X

class Datasets:
    def __init__(self):
        pass
    
    def linesegment(self, RES=100, noise=0):
        np.random.seed(42)
        L = 1
        xv = np.linspace(0, L, RES)[:,np.newaxis]
        yv = noise*np.random.randn(xv.shape[0],1)
        X = np.concatenate([xv,yv],axis=1)
        labelsMat = X[:,0][:,np.newaxis]
        ddX = np.minimum(X,L-X).flatten()
        print('X.shape = ', X.shape)
        return X, labelsMat, ddX
    
    def circle(self, RES=100, noise=0):
        np.random.seed(42)
        theta = np.linspace(0, 2*np.pi, RES)[:-1]
        xv = np.cos(theta)[:,np.newaxis]
        yv = np.sin(theta)[:,np.newaxis]
        X = np.concatenate([xv,yv], axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def closedcurve_1(self, n=1000, noise=0):
        t = 2.01*np.pi*np.random.uniform(0,1,n)
        x = np.cos(t)
        y = np.sin(2*t)
        z = np.sin(3*t)
        X = np.vstack([x,y,z])
        X = np.transpose(X)
        X += noise*np.random.randn(X.shape[0],X.shape[1])
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def rectanglegrid(self, ar=16, RES=100, noise=0, noise_type='uniform'):
        sideLx = np.sqrt(ar)
        sideLy = 1/sideLx
        RESx = int(sideLx*RES+1)
        RESy = int(sideLy*RES+1)
        x = np.linspace(0, sideLx, RESx)
        y = np.linspace(0, sideLy, RESy)
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,np.newaxis]
        yv = yv.flatten('F')[:,np.newaxis]
        X = np.concatenate([xv,yv], axis=1)
        if noise:
            np.random.seed(42)
            n = xv.shape[0]
            if noise_type == 'normal':
                n = xv.shape[0]
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X = X + noise*np.random.normal(0,1,(n,3))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
            
        labelsMat = X
        print('X.shape = ', X.shape)
        
        n = X.shape[0]
        ddX = np.zeros(n)
        for k in range(n):
            ddXx = np.min([X[k,0], sideLx-X[k,0]])
            ddXy = np.min([X[k,1], sideLy-X[k,1]])
            ddX[k] = np.min([ddXx, ddXy])
            
        return X, labelsMat, ddX
    
    def rectanglegrid_mog(self, ar=16, RES=10, n=100, sigma=0.015, noise=0, noise_type='uniform'):
        sideLx = np.sqrt(ar)
        sideLy = 1/sideLx
        RESx = int(sideLx*RES+1)
        RESy = int(sideLy*RES+1)
        x = np.linspace(0, sideLx, RESx)
        y = np.linspace(0, sideLy, RESy)
        xv_, yv_ = np.meshgrid(x, y)
        xv_ = xv_.flatten('F')
        yv_ = yv_.flatten('F')
        
        xv = []
        yv = []
        cluster_label = []
        np.random.seed(42)
        for i in range(xv_.shape[0]):
            X_r = np.random.multivariate_normal([xv_[i],yv_[i]], [[sigma,0],[0,sigma]], [n])
            #X_r[:,0] = np.clip(X_r[:,0], 0, sideLx)
            #X_r[:,1] = np.clip(X_r[:,1], 0, sideLy)
            xv += X_r[:,0].tolist()
            yv += X_r[:,1].tolist()
            cluster_label += [i]*n
        
        xv = np.array(xv)[:,np.newaxis]
        yv = np.array(yv)[:,np.newaxis]
        cluster_label = np.array(cluster_label)[:,np.newaxis]
        
        X = np.concatenate([xv,yv], axis=1)
        if noise:
            np.random.seed(42)
            n = xv.shape[0]
            if noise_type == 'normal':
                n = xv.shape[0]
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X = X + noise*np.random.normal(0,1,(n,3))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
            
        labelsMat = X
        print('X.shape = ', X.shape)
        
        n = X.shape[0]
        ddX = np.zeros(n)
        for k in range(n):
            ddXx = np.min([X[k,0], sideLx-X[k,0]])
            ddXy = np.min([X[k,1], sideLy-X[k,1]])
            ddX[k] = np.min([ddXx, ddXy])
            
        return X, labelsMat, ddX
    
    def barbell(self, RES=100):
        A1 = 0.425
        Rmax = np.sqrt(A1/np.pi)
        sideL1x = 1.5
        sideL1y = (1-2*A1)/sideL1x

        sideLx = sideL1x+4*Rmax
        sideLy = 2*Rmax

        RESx = int(np.ceil(sideLx*RES)+1)
        RESy = int(np.ceil(sideLy*RES)+1)
        x1 = np.linspace(0,sideLx,RESx)
        y1 = np.linspace(0,sideLy,RESy)
        x1v, y1v = np.meshgrid(x1,y1);
        x1v = x1v.flatten('F')[:,np.newaxis]
        y1v = y1v.flatten('F')[:,np.newaxis]
        x2v = np.copy(x1v)
        y2v = np.copy(y1v)
        
        mask1 = (((x1v-Rmax)**2+(y1v-Rmax)**2) < Rmax**2)|(((x1v-3*Rmax-sideL1x)**2+(y1v-Rmax)**2)<Rmax**2)
        mask2 = (x2v>=(2*Rmax))&(x2v<=(2*Rmax+sideL1x))&(y2v>(Rmax-sideL1y/2))&(y2v<(Rmax+sideL1y/2))
        x1v = x1v[mask1][:,np.newaxis]
        y1v = y1v[mask1][:,np.newaxis]
        x2v = x2v[mask2][:,np.newaxis]
        y2v = y2v[mask2][:,np.newaxis]
        xv = np.concatenate([x1v,x2v],axis=0)
        yv = np.concatenate([y1v,y2v],axis=0)
        X = np.concatenate([xv,yv],axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        
        n = X.shape[0]
        ddX = np.zeros(n)
        for k in range(n):
            x_ = X[k,0]
            y_ = X[k,1]
            if (x_<=2*Rmax) or (x_>=(2*Rmax+sideL1x)):
                if x_>=(2*Rmax+sideL1x):
                    x_=x_-2*Rmax-sideL1x
                    x_=2*Rmax-x_
                if (x_>=Rmax) and (y_>=Rmax) and (y_<=Rmax+sideL1y*(x_-Rmax)/(2*Rmax)):
                    ddX[k]=np.sqrt((x_-2*Rmax)**2+(y_-(Rmax+sideL1y/2))**2)
                elif (x_>Rmax) and (y_<=Rmax) and (y_>=Rmax-sideL1y*(x_-Rmax)/(2*Rmax)):
                    ddX[k]=np.sqrt((x_-2*Rmax)**2+(y_-(Rmax-sideL1y/2))**2)
                else:
                    ddX[k]=Rmax-np.sqrt((x_-Rmax)**2+(y_-Rmax)**2)
            else:
                ddX[k]=np.min([y_-(Rmax-sideL1y/2),Rmax+sideL1y/2-y_])
        ddX[ddX<1e-2] = 0
        return X, labelsMat, ddX
    
    def squarewithtwoholes(self, RES=100):
        sideLx = 1
        sideLy = 1
        RESx = sideLx*RES+1
        RESy = sideLy*RES+1
        x = np.linspace(0,sideLx,RESx);
        y = np.linspace(0,sideLy,RESy);
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,np.newaxis]
        yv = yv.flatten('F')[:,np.newaxis]
        X = np.concatenate([xv,yv], axis=1)
        hole1 = np.sqrt((X[:,0] - 0.5*np.sqrt(2))**2 + (X[:,1]-0.5*np.sqrt(2))**2) < 0.1*np.sqrt(2)
        hole2 = np.abs(X[:,0] - 0.2*np.sqrt(2)) + np.abs(X[:,1]-0.2*np.sqrt(2)) < 0.1*np.sqrt(2)
        
        Xhole1 = X[hole1,:]
        Xhole2 = X[hole2,:]
        ddX1 = np.min(cdist(X,Xhole1),axis=1)
        ddX1[ddX1<1e-2*1.2] = 0
        ddX2 = np.min(cdist(X,Xhole2),axis=1)
        ddX2[ddX2<1e-2*1.2] = 0
        ddXx = np.minimum(X[:,0],sideLx-X[:,0])
        ddXy = np.minimum(X[:,1],sideLy-X[:,1])
        ddX = np.minimum(ddXx,ddXy)
        ddX = np.minimum(ddX,ddX1)
        ddX = np.minimum(ddX,ddX2)
        
        X = X[~hole1 & ~hole2,:]
        ddX = ddX[~hole1 & ~hole2]
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat, ddX
    
    def spherewithhole(self, n=10000):
        Rmax = np.sqrt(1/(4*np.pi))
        indices = np.arange(n)
        indices = indices+0.5
        phiv = np.arccos(1 - 2*indices/n)[:,np.newaxis]
        thetav = (np.pi*(1 + np.sqrt(5))*indices)[:,np.newaxis]
        X = np.concatenate([np.sin(phiv)*np.cos(thetav), np.sin(phiv)*np.sin(thetav), np.cos(phiv)], axis=1)
        X = X*Rmax
        z0 = np.max(X[:,2])
        R_hole = Rmax/6
        hole = (X[:,0]**2+X[:,1]**2+(X[:,2]-z0)**2)<R_hole**2
        
        Xhole = X[hole,:]
        ddX = np.min(cdist(X,Xhole), axis=1)
        ddX[ddX<1e-2*1.2] = 0
        
        X = X[~hole,:]
        ddX = ddX[~hole]
        thetav = thetav[~hole][:,np.newaxis]
        phiv = phiv[~hole][:,np.newaxis]
        labelsMat = np.concatenate([np.mod(thetav,2*np.pi), phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, ddX
    
    def swissrollwithhole(self, RES=100):
        theta0 = 3*np.pi/2
        nturns = 2
        rmax = 2*1e-2
        sideL1 = integrate.quad(lambda x: rmax*np.sqrt(1+x**2), theta0, theta0*(1+nturns))[0]
        sideL2 = 1/sideL1
        RESt = int(np.ceil(sideL1*RES+1))
        tdistv = np.linspace(0,sideL1,RESt)
        tv = []
        for tdist in tdistv.tolist():
            tt = fsolve(lambda x: (0.5*rmax*(x*np.sqrt(1+x**2)+np.arcsinh(x)))-\
                                   0.5*rmax*(theta0*np.sqrt(1+theta0**2)+np.arcsinh(theta0))-\
                                   tdist,theta0*(1+nturns/2))
            tv.append(tt)
        tv = np.array(tv)    
        RESh = int(np.ceil(sideL2*RES+1))
        heightv = np.linspace(0,sideL2,RESh)[:,np.newaxis]
        heightv = np.tile(heightv,[RESt,1])
        heightv = heightv.flatten('F')[:,np.newaxis]
        tv = np.repeat(tv,RESh)[:,np.newaxis]
        X=np.concatenate([rmax*tv*np.cos(tv), heightv, rmax*tv*np.sin(tv)], axis=1)
        
        ddX11 = np.minimum(heightv, sideL2-heightv).flatten()
        ddX12 = np.tile(tdistv[:,np.newaxis], RESh).flatten()
        ddX12 = np.minimum(ddX12, sideL1-ddX12)
        ddX1 = np.minimum(ddX11, ddX12)

        y_mid = sideL2*0.5
        t_min = np.min(tv)
        t_max = np.max(tv)
        t_range = t_max-t_min
        t_mid = t_min + t_range/2
        x_mid = rmax*t_mid*np.cos(t_mid)
        z_mid = rmax*t_mid*np.sin(t_mid)
        hole = np.sqrt((X[:,0]-x_mid)**2+(X[:,1]-y_mid)**2+(X[:,2]-z_mid)**2)<0.1

        Xhole = X[hole,:]
        ddX2 = np.min(cdist(X,Xhole), axis=1)
        ddX2[ddX2<1e-2*1.2] = 0
        
        X = X[~hole,:]
        ddX = np.minimum(ddX1[~hole], ddX2[~hole])
        tv = tv[~hole]
        labelsMat = np.concatenate([tv, X[:,[1]]], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, ddX
    
    def noisyswissroll(self, RES=100, noise=0.01, noise_type = 'normal'):
        theta0 = 3*np.pi/2
        nturns = 2
        rmax = 2*1e-2
        sideL1 = integrate.quad(lambda x: rmax*np.sqrt(1+x**2), theta0, theta0*(1+nturns))[0]
        sideL2 = 1/sideL1
        RESt = int(np.ceil(sideL1*RES+1))
        tdistv = np.linspace(0,sideL1,RESt)
        tv = []
        for tdist in tdistv.tolist():
            tt = fsolve(lambda x: (0.5*rmax*(x*np.sqrt(1+x**2)+np.arcsinh(x)))-\
                                   0.5*rmax*(theta0*np.sqrt(1+theta0**2)+np.arcsinh(theta0))-\
                                   tdist,theta0*(1+nturns/2))
            tv.append(tt)
        tv = np.array(tv)    
        RESh = int(np.ceil(sideL2*RES+1))
        heightv = np.linspace(0,sideL2,RESh)[:,np.newaxis]
        heightv = np.tile(heightv,[RESt,1])
        heightv = heightv.flatten('F')[:,np.newaxis]
        tv = np.repeat(tv,RESh)[:,np.newaxis]
        X=np.concatenate([rmax*tv*np.cos(tv), heightv, rmax*tv*np.sin(tv)], axis=1)
        np.random.seed(42)
        if noise_type == 'normal':
            X = X+noise*np.random.normal(0,1,[X.shape[0],3])
        elif noise_type == 'uniform':
            X = X+noise*np.random.uniform(0,1,[X.shape[0],3])
        labelsMat = np.concatenate([tv, X[:,[1]]], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
        
    def sphere(self, n=10000, noise = 0):
        R = np.sqrt(1/(4*np.pi))
        indices = np.arange(n)+0.5
        phiv = np.arccos(1 - 2*indices/n)
        phiv = phiv[:,np.newaxis]
        thetav = np.pi*(1 + np.sqrt(5))*indices
        thetav = thetav[:,np.newaxis]
        X = np.concatenate([np.sin(phiv)*np.cos(thetav),
                            np.sin(phiv)*np.sin(thetav),
                            np.cos(phiv)], axis=1)
        X = X*R;
        np.random.seed(2)
        X = X*(1+noise*np.random.uniform(-1,1,(X.shape[0],1)))
        labelsMat = np.concatenate([np.mod(thetav,2*np.pi), phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def sphere_and_swissroll(self, n=5000, RES=70, noise1 = 0.01, noise2=0.015, sep=1):
        s1, l1, _ = self.sphere(n, noise=noise1)
        s2, l2, _ = self.noisyswissroll(RES=RES, noise=noise2)
        x_max = np.max(s1[:,0])
        x_min = np.min(s2[:,0])
        s2 = s2 + np.array([x_max-x_min,0,0]).reshape((1,3)) + sep
        X = np.concatenate([s1, s2], axis=0)
        labelsMat = np.concatenate([l1, l2], axis=0)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def multi_spheres(self, m=3, n=3000, noise = 0, sep=1):
        X = []
        labelsMat = []
        offset = 0
        for i in range(m):
            s1, l1, _ = self.sphere(n, noise)
            if i > 0:
                s1[:,0] += 1.25*offset - np.min(s1[:,0])
            offset = np.max(s1[:,0])
            X.append(s1)
            labelsMat.append(l1)
            
        
        X = np.concatenate(X, axis=0)
        labelsMat = np.concatenate(labelsMat, axis=0)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def sphere_mog(self, k=10, n=1000, sigma=0.1, noise = 0):
        R = np.sqrt(1/(4*np.pi))
        indices = np.arange(k)+0.5
        phiv_ = np.arccos(1 - 2*indices/k)
        thetav_ = np.mod(np.pi*(1 + np.sqrt(5))*indices, 2*np.pi)
        
        phiv = []
        thetav = []
        cluster_label = []
        np.random.seed(42)
        for i in range(k):
            X_r = np.random.multivariate_normal([phiv_[i],thetav_[i]], [[sigma,0],[0,sigma]], [n])
            phiv += X_r[:,0].tolist()
            thetav += X_r[:,1].tolist()
            cluster_label += [i]*n
        
        phiv = np.array(phiv)[:,np.newaxis]
        thetav = np.array(thetav)[:,np.newaxis]
        cluster_label = np.array(cluster_label)[:,np.newaxis]
        
        X = np.concatenate([np.sin(phiv)*np.cos(thetav),
                            np.sin(phiv)*np.sin(thetav),
                            np.cos(phiv)], axis=1)
        X = X*R;
        np.random.seed(2)
        X = X*(1+noise*np.random.uniform(-1,1,(X.shape[0],1)))
        labelsMat = np.concatenate([cluster_label,np.mod(thetav,2*np.pi), phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def spherewithanomaly(self, n=10000, epsilon=0.05, noise=0.05):
        R = np.sqrt(1/(4*np.pi))
        indices = np.arange(n)+0.5
        phiv = np.arccos(1 - 2*indices/n)
        phiv = phiv[:,np.newaxis]
        thetav = np.pi*(1 + np.sqrt(5))*indices
        thetav = thetav[:,np.newaxis]
        X = np.concatenate([np.sin(phiv)*np.cos(thetav),
                            np.sin(phiv)*np.sin(thetav),
                            np.cos(phiv)], axis=1)
        X = X*R;
        np.random.seed(2)
        X = X*(1+noise*np.random.uniform(-1,1,(X.shape[0],1)))
        labelsMat = np.concatenate([np.mod(thetav,2*np.pi), phiv], axis=1)
        
        # add anomaly at north pole
        k = np.argmin(np.abs(phiv))
        d_k_kp = np.sqrt(np.sum((X - X[k,:][np.newaxis,:])**2, axis=1))
        mask = (d_k_kp < epsilon)
        n_ = np.sum(mask)
        np.random.seed(42)
        X[mask,2:3] = X[mask,2:3] + np.random.normal(0,1,(n_,1))*noise
        
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def flattorus4d(self, ar=4, RES=100):
        sideLx=np.sqrt(ar)
        sideLy=1/sideLx
        Rout = sideLx/(2*np.pi)
        Rin = sideLy/(2*np.pi)
        RESx=int(sideLx*RES+1)
        RESy=int(sideLy*RES+1)
        x=np.linspace(0,sideLx,RESx)[:-1] # remove 2pi
        y=np.linspace(0,sideLy,RESy)[:-1] # remove 2pi
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,np.newaxis]/Rout
        yv = yv.flatten('F')[:,np.newaxis]/Rin
        X=np.concatenate([Rout*np.cos(xv), Rout*np.sin(xv), Rin*np.cos(yv), Rin*np.sin(yv)], axis=1)
        labelsMat = np.concatenate([xv, yv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def curvedtorus3d(self, n=10000, noise=0):
        Rmax=0.25;
        rmax=1/(4*(np.pi**2)*Rmax);
        X = []
        thetav = []
        phiv = []
        np.random.seed(42)
        k = 0
        while k < n:
            rU = np.random.uniform(0,1,3)
            theta = 2*np.pi*rU[0]
            phi = 2*np.pi*rU[1]
            if rU[2] <= (Rmax + rmax*np.cos(theta))/(Rmax + rmax):
                thetav.append(theta)
                phiv.append(phi)
                k = k + 1
        
        thetav = np.array(thetav)[:,np.newaxis]
        phiv = np.array(phiv)[:,np.newaxis]
        np.random.seed(42)
        noise = noise*np.random.uniform(-1,1,(phiv.shape[0],1))
        X = np.concatenate([(Rmax+(1+noise)*rmax*np.cos(thetav))*np.cos(phiv),
                             (Rmax+(1+noise)*rmax*np.cos(thetav))*np.sin(phiv),
                             (1+noise)*rmax*np.sin(thetav)], axis=1)
        labelsMat = np.concatenate([thetav, phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def kleinbottle4d(self, ar=4, RES=100):
        sideLx=np.sqrt(ar)
        sideLy=1/sideLx
        Rout = sideLx/(2*np.pi)
        Rin = sideLy/(2*np.pi)
        RESx=int(sideLx*RES+1)
        RESy=int(sideLy*RES+1)
        x=np.linspace(0,sideLx,RESx)[:-1] # remove 2pi
        y=np.linspace(0,sideLy,RESy)[:-1] # remove 2pi
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,np.newaxis]/Rout
        yv = yv.flatten('F')[:,np.newaxis]/Rin
        X=np.concatenate([(Rout+Rin*np.cos(yv))*np.cos(xv), (Rout+Rin*np.cos(yv))*np.sin(xv),
                          Rin*np.sin(yv)*np.cos(xv/2), Rin*np.sin(yv)*np.sin(xv/2)], axis=1)
        labelsMat = np.concatenate([xv, yv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def mobiusstrip3d(self, ar=4, RES=90):
        sideLx=np.sqrt(ar)
        sideLy=1/sideLx
        Rmax = sideLx/(2*np.pi)
        RESx=int(sideLx*RES+1+50)
        RESy=int(sideLy*RES+1)
        x=np.linspace(0,sideLx,RESx)[:-1] #remove 2pi
        y=np.linspace(-sideLy/2,sideLy/2,RESy)
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,np.newaxis]/Rmax
        yv = yv.flatten('F')[:,np.newaxis]
        X=np.concatenate([(1+0.5*yv*np.cos(0.5*xv))*np.cos(xv),
                         (1+0.5*yv*np.cos(0.5*xv))*np.sin(xv),
                         0.5*yv*np.sin(0.5*xv)], axis=1)   
        labelsMat = np.concatenate([xv, yv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def twinpeaks(self, n=10000, noise=0, ar=4):
        np.random.seed(42)
        s_ = 2
        t_ = 2*ar
        t = np.random.uniform(-t_/2,t_/2,(n,1))
        s = np.random.uniform(-s_/2,s_/2,(n,1))
        h = 0.3*(1-t**2)*np.exp(-t**2-(s+1)**2)-\
            (0.2*t-np.power(t,3)-np.power(s,5))*np.exp(-t**2-s**2)-\
            0.1*np.exp(-(t+1)**2-s**2)
        
        eta = noise * np.random.normal(0,1,(n,1))
        X = np.concatenate([t,s,h+eta],axis=1)
        labelsMat = np.concatenate([t,s], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
        
    
    def floor(self, fpath, noise=0.01, n_transmitters = 42, eps = 1):
        data = scipy.io.loadmat(fpath)
        X = data['X']
        np.random.seed(42)
        X = X + np.random.uniform(0, 1, X.shape)*noise
        t_inds = np.random.permutation(range(X.shape[0]))
        t_inds = t_inds[:n_transmitters]
        t_locs = X[t_inds,:]
        
        mask = np.ones(X.shape[0])
        mask[t_inds] = 0
        X = X[mask==1,:]
        labelsMat = X.copy()
        
        dist_bw_x_and_t = cdist(X, t_locs)
        X = np.exp(-(dist_bw_x_and_t**2)/(eps**2))
        print('X.shape = ', X.shape)
        return X, labelsMat, None
        
    
    def solidcuboid3d(self, l=0.5, w=0.5, RES=20):
        sideLx = l
        sideLy = w
        sideLz = 1/(l*w)
        RESx=int(sideLx*RES+1)
        RESy=int(sideLy*RES+1)
        RESz=int(sideLz*RES+1)
        x=np.linspace(0,sideLx,RESx)
        y=np.linspace(0,sideLy,RESy)
        z=np.linspace(0,sideLz,RESz)
        xv, yv, zv = np.meshgrid(x, y, z)
        xv = xv.flatten('F')[:,np.newaxis]
        yv = yv.flatten('F')[:,np.newaxis]
        zv = zv.flatten('F')[:,np.newaxis]
        X = np.concatenate([xv,yv,zv], axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def mnist(self, digits, n, n_pca=25, normalize=False):
        X0, y0 = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = []
        y = []
        for digit in digits:
            X_ = X0[y0 == str(digit),:]
            X_= X_[:n,:]
            X.append(X_)
            y.append(np.zeros(n)+digit)
            
        X = np.concatenate(X, axis=0)
        if normalize:
            X = X - np.mean(X,axis=0)[np.newaxis,:]
            X = X / (np.std(X,axis=0)[np.newaxis,:] + 1e-12)
        y = np.concatenate(y, axis=0)
        labelsMat = y[:,np.newaxis]
        
        if n_pca:
            X_new = do_pca(X,n_pca)
        else:
            X_new = X
        
        print('X_new.shape = ', X_new.shape)
        return X_new, labelsMat, X, [28,28] 
    
    def face_data(self, fpath, pc=False, n_pca=0):
        data = scipy.io.loadmat(fpath)
        if pc:
            X = data['image_pcs'].transpose()
        else:
            X = data['images'].transpose()
        labelsMat = np.concatenate([data['lights'].transpose(), data['poses'].transpose()], axis=1)
        
        if n_pca:
            X_new = do_pca(X,n_pca)
        else:
            X_new = X
        
        print('X.shape = ', X_new.shape)
        
        min_pose = np.min(labelsMat[:,1])
        min_light = np.min(labelsMat[:,0])
        max_pose = np.max(labelsMat[:,1])
        max_light = np.max(labelsMat[:,0])
        N = X.shape[0]
        ddX = np.zeros(N)
        for k in range(N):
            ddX1 = np.min([labelsMat[k,0]-min_light, max_light-labelsMat[k,0]])
            ddX2 = np.min([labelsMat[k,1]-min_pose, max_pose-labelsMat[k,1]])
            ddX[k] = np.min([ddX1, ddX2])
        
        return X_new, labelsMat, X, [64,64], ddX
    
    def puppets_data(self, dirpath, prefix='s1', n=None, bbox=None,
                     grayscale=False, normalize = False, n_pca=100):
        X = []
        labels = []
        fnames = []
        for fname in sorted(os.listdir(dirpath)):
            if prefix in fname:
                fnames.append(fname)
        
        if n is not None:
            fnames = fnames[:n]
            
        for fname in fnames:
            X_k = read_img(dirpath+'/'+fname, bbox=bbox, grayscale=grayscale)
            X.append(X_k.T.flatten())
            labels.append(int(fname.split('.')[0].split('_')[1])-100000)
        
        img_shape = X_k.shape
        X = np.array(X)
        labels = np.array(labels)[:,np.newaxis]-1
        labelsMat = np.concatenate([labels,labels], axis=1)
#         m1 = X.shape[0]/310
#         m2 = 91
#         m3 = 89
#         if prefix=='s1':
#             labelsMat = np.concatenate([np.mod(labels,m1), np.mod(labels,m2)], axis=1)
#         elif prefix=='s2':
#             labelsMat = np.concatenate([np.mod(labels,m2), np.mod(labels,m3)], axis=1)
        if normalize:
            X = X - np.mean(X,axis=0)[np.newaxis,:]
            X = X / (np.std(X,axis=0)[np.newaxis,:] + 1e-12)
            
        if n_pca:
            X_new = do_pca(X, n_pca)
        else:
            X_new = X
        print('X.shape = ', X_new.shape)
        return X_new, labelsMat, X, img_shape
        
    
    def soils88(self, labels_path, X_path):
        df2 = pd.read_csv(X_path, sep='\t')
        df2 = df2.sort_values(by='Unnamed: 0').reset_index(drop=True)
        sample_names = df2['Unnamed: 0'].tolist()
        X = df2.to_numpy()[:,1:]
        
        df1 = pd.read_csv(labels_path, sep='\t')
        mask = df1['sample_name'].apply(lambda x: x in sample_names)
        df1 = df1[mask].reset_index(drop=True)
        df1 = df1.sort_values(by='sample_name').reset_index(drop=True)
        labelsMat = df1.to_numpy()[:,1:]
        
        print('X.shape = ', X.shape)
        return X, labelsMat, None