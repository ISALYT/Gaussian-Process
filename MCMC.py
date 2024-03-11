# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:27:57 2019

@author: Luyuting
"""
import numpy as np
import corner
import emcee
import matplotlib.pyplot as plt
from numpy.linalg import inv,det
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
#from lightkurve import search_lightcurvefile,log
import lightkurve as lk
import pandas as pd
import os
from scipy.interpolate import griddata
import sys
from astropy.timeseries import LombScargle
#%%
def highpass(lc, width):
    ###CAN INCREASE  SPEED OF COMPUTATION

    dt = 29.43/(60*24)
    half_wday = round(width/2)
    

    last = max(lc.time) 
    front = np.arange(0,half_wday+0.0001,dt)*(-1)
    front_t = front[::-1]
    back_t = np.arange(last+dt,last+half_wday,dt)
    time_padded = np.append(front_t, lc.time)
    time_padded = np.append(time_padded, back_t)
    
    front_f = np.ones(len(front_t))*lc.flux[0]
    back_f = np.ones(len(back_t))*lc.flux[-1]
    flux_padded = np.append(front_f,lc.flux)
    flux_padded =  np.append(flux_padded,back_f)
    last_ind = np.where(time_padded == last)[0][0]
    
    c = np.zeros(len(lc.flux))

    for i in range(len(front_t),last_ind+1):
        ind = np.logical_and(time_padded> time_padded[i]-half_wday, time_padded < time_padded[i]+half_wday)
        j = i-len(front_t)
        c[j] = np.sum(flux_padded[ind])/len(flux_padded[ind])
    
    R = lc.flux - c
    lc.flux = R
    
    ##
    
    return lc,R,c  


    
def no_outlier(lc,sigma):
    if sigma == 5:
        sigma = 5
    lc_no_outlier =lc.remove_outliers(sigma)
     
    return lc_no_outlier
    
    
def boundary(lc,limitsigma):
    if limitsigma ==3:
        sigma = 3
    sigma = limitsigma
    lc_withedge = lc.remove_outliers(sigma)
    return lc_withedge

def fill_gap(lc):
    n,bins= np.histogram(lc.flux, 100, density=True)

    x,y = three_val(n,bins)
    noise_sig = (abs(x[0]-x[1])+abs(x[1]-x[2]))/2
    lctime = lc.time - lc.time[0]
    lcflux = lc.flux
    fluxerr = lc.flux_err 
    flux = np.array([])
    flux_err = np.array([])
    time = np.arange(0, lctime[-1]+0.01, 29.4/(60*24))
    half_width = 0.5*29.4/(24*60)
    for i in range(0,len(time)):
        upperbound = time[i] + half_width
        lowerbound = time[i] - half_width
        condition1 = lctime <=upperbound
        condition2 = lctime > lowerbound
        index = np.where(condition1 & condition2) #tuple to expand
        if len(index[0])>0:
            sample_flux = lcflux[index]
            value = np.mean(sample_flux)/len(sample_flux)
            flux = np.insert(flux, i, value)
            sample_fluxerr = fluxerr[index]
            value_err = np.mean(sample_fluxerr)/len(sample_fluxerr)
            flux_err = np.insert(flux_err, i, value_err)
        else:
            flux = np.insert(flux, i, 0+np.random.normal(0, noise_sig, 1))    ###changed 0 to 1  &&  +np.random.normal(0, 0.05, 1)
            flux_err = np.insert(flux_err, i, noise_sig)     ###consider whether it is proper
    lc.flux = flux
    lc.time = time
    lc.flux_err = flux_err
    return lc

def three_val(n,b):
    y = []
    x = []
    add = 0
    d = b[2]-b[1]
    for t in range(0,len(n)):
        add = sum(n[0:t])*d
        if add>0.16:
            lower_i = t
            break
    
    y.append(n[lower_i-1])
    x.append(b[lower_i-1])
    
    for t in range(0,len(n)):
        add = sum(n[0:t])*d
        if add>0.5:
            peak_i = t
            break
    
    y.append(n[peak_i-1])
    x.append(b[peak_i-1])

    for t in range(0,len(n)):
        add = sum(n[0:t])*d
        if add>0.84:
            upper_i = t
            break
    
    y.append(n[upper_i-1])
    x.append(b[upper_i-1])   ####
    return x,y

#%%
def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))




#%%
#GAUSSIAN
def fluxchoice(time,flux):
    subnum = 3000
    idx = np.random.choice(np.arange(len(flux)), subnum, replace=False)
    y = flux[idx].reshape(6,500)
    x = time[idx].reshape(6,500)
    
    return x, y
def gaussian(mu,sigma,x_in):
    x = x_in
    factor = 1/(sigma*np.sqrt(np.pi*2))
    term2 = np.exp(-0.5*((x-mu)/sigma)**2)
    val = factor * term2
    return val


def kernel(A,l,gamma,P,sig):  
    n = len(flux)
    K = np.zeros(shape = (n,n)) ##here shape = (n*m) means n rows, m columns
    ###Then compute K_ij where i indicates (i+1)th row, j indicates (j+1)th column 
    count = 1
    for i in range(0,n):
        
        for j in range(0,n):
            interval = flux[i] - flux[j]
            term1 = -(interval)**2/(2*l**2)
            term2 = -gamma**2*(np.sin(np.pi*(interval)/P))**2
            if i == j:
                K[i,j] = sig**2
            else:
                K[i,j] = A*np.exp(term1+term2)
            count = count + 1
        #print('%d/%d completion'%(count,n))
                
    return K
##--------likelihood-------

def loglikelihood(theta,time,flux):  #theta = [A,l,gamma,P,sig]
    lnA,lnl,lngamma,lnsig, lnP = theta
    A,l,gamma,sig,P = np.exp([lnA,lnl,lngamma,lnsig, lnP])
    x, y = fluxchoice(time,flux)
    lnL = 0
    for ind in range(6):
        influx = y[ind,:]
        intime = x[ind,:]
        K = kernel(A,l,gamma,P,sig)
        invC = inv(K)
        term1 = np.dot(invC,r)
        term1 = -0.5*np.dot(r,term1)
        term2 = -0.5*np.log(det(K))
        ###not sure whether log 2pi is ln or log10
        term3 = 0.5*len(flux)*np.log10(2*np.pi)
        temp =  term1 + term2 + term3  
        lnL = temp + lnL
    
    return lnL

def log_prior(theta):    ###log prior inf not quite understood
    lnA,lnl,lngamma,lnsig, lnP = theta
    if -20.0 < lnA < 0 and 2.0 < lnl < 20.0 and -10.0 < lngamma < 3.0 and -20.0< lnsig <0.0 and np.log(0.5)< lnP <np.log(50) :
        term1 = gaussian(-13.5,5.7,lnA)
        term2 = gaussian(7.2,1.2,lnl)
        term3 = gaussian(-2.3,1.4,lngamma)
        term4 = gaussian(-17,5,lnsig)
        term5 = 1
        lp = term1*term2*term3*term4*term5
        return lp
    return -np.inf


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    if np.isnan(lp):
        print('prior problem')
   
    lh = loglikelihood(theta,time,flux)
    if np.isnan(lh):
        print('lh problem')
    
    return lp + lh

def lightcurve(target):
    
    rpath = r'..\data'
    rpath = os.path.join(rpath,'KIC%s.csv'%target)
    df = pd.read_csv(rpath)
    lctime = np.array(df['time'])
    lcflux = np.array(df['flux'])
    lcerr = np.array(df['flux_err'])
    lc = lk.LightCurve(time = lctime, flux = lcflux, flux_err = lcerr)
    lc = no_outlier(lc,sigma=5)  
    lc,R,c = highpass(lc,width=50)
    lc = fill_gap(lc)     
    time = lc.time
    flux = lc.flux
    return time, flux


nwalkers = 10
ndim = 5
startlnA = -13 + np.random.randn(nwalkers)*10
startlnl = 7.2 + np.random.randn(nwalkers)*5
startlngamma = -2 + np.random.randn(nwalkers)*5
startsigma = -17 + np.random.randn(nwalkers)*10
startP = np.log(20)+ np.random.randn(nwalkers)*5
pos = np.zeros((nwalkers,ndim))
pos[:,0] = startlnA
pos[:,1] = startlnl
pos[:,2] = startlngamma
pos[:,3] = startsigma
pos[:,4] = startP
nsteps = 50

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,moves=[(emcee.moves.DEMove(), 0.4),(emcee.moves.DEMove(), 0.4), (emcee.moves.DESnookerMove(), 0.2)])
    sampler.run_mcmc(pos, nsteps, progress=True)


fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["ln(A)", "ln(l)", "ln($/gamma$)","ln($/sigma$)","ln(P)"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")    

figpath = r'../MCMC.png'
plt.savefig(figpath)
plt.close()   
    
    
flat_samples = sampler.get_chain(flat=True)
labels = ["ln(A)", "ln(l)", "ln($/gamma$)","ln($/sigma$)","ln(P)"]
fig = corner.corner(
    flat_samples, labels=labels
)
figpath = r'../contour.png'
plt.savefig(figpath)
plt.close()



