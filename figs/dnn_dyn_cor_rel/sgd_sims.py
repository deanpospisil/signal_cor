#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:08:20 2020

@author: dean
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.figure(figsize=(3,3))
m = 20
t = np.rad2deg(np.linspace(0, np.pi*2 - np.pi*2/m, m)[:,np.newaxis])
frac_tc = 0.5
w_teach = frac_tc*np.cos(np.deg2rad(t)) + (1-frac_tc)*np.cos(2*np.deg2rad(t))
w_teach = w_teach/(w_teach**2).sum()**0.5
w = np.random.normal(size=m)[:,np.newaxis]
w = w/(w**2).sum()**0.5
#w = np.zeros((m,1))
lr = .01

n_steps = 200

for i in range(n_steps):
    if i%40==0:
        plt.plot(t, w, c=cm.cool(i/n_steps))
    train_input = np.random.normal(size=m)[:,np.newaxis] #+ w_teach
    #train_input = w_teach
    target = w_teach.T.dot(train_input)
    train_output = w.T.dot(train_input)
    dw = 2*(train_output-target)*train_input    
    w = w - dw*lr


plt.plot(t, w_teach, c='k')  
plt.legend(list(np.array(range(5))*40 ) + ['Teaching weights',], 
           title='learning step', fontsize=9)
plt.xlabel('Phase')
plt.ylabel('Weights')

#%%
plt.figure(figsize=(9,3))
plt.suptitle('Input distribution examples',y=1.1)
plt.subplot(131)
plt.title('noise')
noise = np.random.normal(size=m)[:,np.newaxis]
noise = noise/(noise**2).sum()**0.5
plt.plot(t, noise)
plt.ylim(-.75,.75)

plt.subplot(132)
plt.title('noise + exemplar inputs')
ns = noise + w_teach*4
ns = ns/(ns**2).sum()**0.5
plt.plot(t, ns)
plt.ylim(-.75,.75)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.subplot(133)

plt.title('exemplar inputs')
plt.plot(t, w_teach)
plt.ylim(-.75,.75)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.tight_layout()


#%%
plt.figure(figsize=(9,3))
plt.suptitle('Teacher weights fraction tuning curve (FTC) stimulus aligned',y=1.1)
frac_tcs = np.linspace(0, 1, 5)

k=0
for i in range(5):
        plt.subplot(1,5, i+1)
        w_teach = frac_tcs[i]*np.cos(np.deg2rad(t)) + (1-frac_tcs[i])*np.cos(2*np.deg2rad(t))
        w_teach = w_teach/(w_teach**2).sum()**0.5
        plt.plot(t,w_teach)
        if i>0:
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
        else:
            plt.xlabel('phase (deg)')
            plt.ylabel('weights')
        plt.title('FTC=' + str(frac_tcs[i]) )


#%%

plt.figure(figsize=(9,3))
plt.suptitle('Stimuli',y=1.1)
phase = np.deg2rad(np.array([0, 90,180, 270]))
stim = np.cos(np.deg2rad(t) + phase)

k=0
for i in range(4):
        plt.subplot(1,4, i+1)
        plt.plot(t, stim[:,i])
        if i>0:
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
        else:
            plt.xlabel('phase (deg)')
            plt.ylabel('input')
        plt.title('phase=' + str(np.rad2deg(phase[i]).round(2) ))
            
#%%
import xarray as xr
def teach_sim(m=20, frac_sig=0, lr=0.01, n_steps=200, frac_tc=0):
    
    t = np.rad2deg(np.linspace(0, np.pi*2 - np.pi*2/m, m)[:,np.newaxis])
    
    w_teach = frac_tc*np.cos(np.deg2rad(t)) + (1-frac_tc)*np.cos(2*np.deg2rad(t))
    w_teach = w_teach/(w_teach**2).sum()**0.5
    
    w = np.random.normal(size=m)[:,np.newaxis] 
    w = w/(w**2).sum()**0.5
    ws = []
    for i in range(n_steps):
        
        noise = np.random.normal(size=m)[:,np.newaxis] 
        noise = noise/(noise**2).sum()**0.5
        train_input = (1-frac_sig)*noise + frac_sig*w_teach
        target = w_teach.T.dot(train_input)
        train_output = w.T.dot(train_input)
        dw = 2*(train_output-target)*train_input    
        w = w - dw*lr
        w = w/(w**2).sum()**0.5
        ws.append(w)

    return ws
    
frac_sigs = np.linspace(0, 0.5, 3)
#lrs = np.logspace(np.log10(0.5), np.log10(2), 5)
lrs = np.linspace(0.1, 3, 2)
frac_tcs = np.linspace(0, 1, 5)



sims = 100
n_steps = 1000

das = []

for i in range(2):
    da = xr.DataArray(np.zeros((m, n_steps, sims,  len(lrs), len(frac_sigs), len(frac_tcs))),
                 dims=['m', 'step', 'sim', 'lr', 'frac_sig', 'frac_tc'],
                 coords = [range(m), range(n_steps), range(sims),  
                           lrs, frac_sigs, frac_tcs])
    
    for sim_params in da.stack(c=['sim', 'lr', 'frac_sig', 'frac_tc']).indexes['c'].values:
        sim, lr, frac_sig, frac_tc = sim_params
        ws = teach_sim(m=20, frac_sig=frac_sig, 
                       lr=lr, n_steps=n_steps, frac_tc=frac_tc)
        ws = np.squeeze(np.array(ws)).T
        da.loc[:, :, sim, lr, frac_sig, frac_tc] = ws
    das.append(da)
    
    
#%%
def norm(x, dim):
    x = x - x.mean(dim)
    x = x/(x**2).sum(dim)**0.5
    return x 

def cor(x, y, dim):
   y = norm(y, dim) 
   x = norm(x, dim)
   r = x.dot(y, dim)
   return r
def fz(r):
    return(0.5*np.log((1+r)/(1-r)))

da = xr.concat(das, 'unit')
da.coords['unit'] = [0,1]


phase = np.deg2rad(t).T
stim = np.cos(np.deg2rad(t) + phase)
stim = xr.DataArray(stim, dims=['m', 'phase'],
             coords=[range(m), t.squeeze()])
resp = (da*stim).sum('m')
#%%
r = (cor(resp.sel(unit=0), resp.sel(unit=1), dim=('phase',)))

#r = (((da.sel(unit=0)-da.sel(unit=1)))**2).sum('m')
#%%
from cycler import cycler
custom_cycler = cycler(color=cm.viridis(np.linspace(0,1,5))) #or simply color=colorlist

fig = plt.figure()
ax = plt.gca()

ax.set_prop_cycle(custom_cycler)
plt.figure(figsize=(3,3))
rn = r.rename({'frac_sig':'Fraction exemplar input'})
axs = (rn.isel(lr=1, frac_tc=slice(0,3))).mean('sim').plot.line(
    x='step', row='Fraction exemplar input', add_legend=False);plt.semilogx()
plt.ylim(-0.2,1)
plt.legend([0,0.25, 0.5], title='FTC')
plt.xlabel('training step')
plt.ylabel('Correlation between neurons')

plt.title('Fraction exemplar input = ' + '0.5')

#plt.figure()
#r.isel(lr=1).mean('sim').plot.line(x='step');plt.semilogx()
#%%

t = np.rad2deg(np.linspace(0, np.pi*2 - np.pi*2/m, m)[:,np.newaxis])
w_teach = np.cos(np.deg2rad(t)) + np.cos(2*np.deg2rad(t))

plt.plot(w_teach)

#%%
da.isel(lr=1,  frac_tc=0, sim=0, unit=0, frac_sig=0).mean('step').plot()
#%%
plt.figure(figsize=(3,3))
resp.isel(lr=0, step=-1, frac_tc=1, sim=0, unit=0, frac_sig=0).plot()  
plt.title('Response to stimuli')
plt.ylim(-1,1)
#stim.sel(phase=0).plot()  