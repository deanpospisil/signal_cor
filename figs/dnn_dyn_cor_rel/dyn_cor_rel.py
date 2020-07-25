#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:56:13 2020

@author: dean
"""


#functions to generate stimuli given parameter set
import os, sys
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import xarray as xr
from itertools import product
from torchvision import models
import torch.nn as nn
import pandas as pd

def norm(x, dim):
    x = x - x.mean(dim)
    x = x/(x**2).sum(dim)**0.5
    return x 

def cor(x, y, dim):
   y = norm(y, dim) 
   x = norm(x, dim)
   r = x.dot(y, dim)
   return r

def rc(x):
    reverse = (slice(None, None, -1),) * x.ndim
    return x[reverse].conj()
def auto_corr(x, y, dim, pad=None):
    xn = norm(x, dim);
    yn = norm(y, dim);
    
    if pad is None:
        cor = np.fft.ifftn(np.fft.fftn(xn)*(np.fft.fftn(rc(yn))))
    else:
        xn = np.pad(xn, pad_width=pad)
        yn = np.pad(yn, pad_width=pad)
        cor = np.fft.ifftn(np.fft.fftn(xn)*(np.fft.fftn(rc(yn))))

    return cor

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def sinusoid_2d(nx, ny, x_0, y_0, sf, ori, phase, bg=0):
    
    x_coords = np.arange(0, nx, dtype=np.float64) - x_0
    y_coords = np.arange(0, ny, dtype=np.float64) - y_0
    xx, yy = np.meshgrid(x_coords, y_coords)
    mu_0, nu_0 = pol2cart(sf, np.deg2rad(ori + 90)) 
    s = np.sin(2*np.pi*(mu_0*xx + nu_0*yy) + np.deg2rad(phase + 90))
    s = s + bg
    
    return s

def window(radius, x_0, y_0, nx, ny):
    x_coords = np.arange(0, nx, dtype=np.float128) - x_0
    y_coords = np.arange(0, ny, dtype=np.float128) - y_0
    xx, yy = np.meshgrid(x_coords, y_coords)
    d = (xx**2 + yy**2)**0.5
    w = np.zeros((int(nx), int(ny)))
    w[d<=radius] = 1
    return w

def cos_window(radius, x_0, y_0, nx, ny):
    x_coords = np.arange(0, nx, dtype=np.float128) - x_0
    y_coords = np.arange(0, ny, dtype=np.float128) - y_0
    xx, yy = np.meshgrid(x_coords, y_coords)
    d = (xx**2 + yy**2)**0.5
    w = np.cos(d*np.pi*(1/radius)) + 1
    w[d>radius] = 0
    return w

def colorize(s, lum, by, rg):
    s_c = s[..., np.newaxis]
    l_by_rg = np.array([[1, 1, 1],
                 [1/3, 1/3, -2/3],
                 [1, -1, 0]]
                 )
    l_by_rg = (l_by_rg/((l_by_rg**2).sum(1, keepdims=True)**0.5))
    rgb = l_by_rg[0]*lum + l_by_rg[1]*by + l_by_rg[2]*rg
    
    s_c = s_c*rgb[np.newaxis,np.newaxis]
    return s_c


def sine_chrom(nx, ny, x_0, y_0, sf, ori, phase, lum, by, rg, bg=0):
    s = sinusoid_2d(nx, ny, x_0, y_0, sf, ori, phase, bg=0)
    s_c = colorize(s, lum, by, rg)
    
    return scale_im(s_c)

def sine_chrom_dual(nx, ny, x_0, y_0, sf, ori, phase, 
                    lum, by, rg, bg=0,
                    sf2=None, rel_ori=None, phase2=None, 
                    lum2=None, by2=None, rg2=None, make_window=False, radius=None):
    
    if sf2 is None:
        sf2 = sf
    if rel_ori is None:
        rel_ori = 0
    if phase2 is None:
        phase2 = phase
    if lum2 is None:
        lum2 = lum
    if by2 is None:
        by2 = by
    if rg2 is None:
        rg2 = rg

    s1 = sine_chrom(nx, ny, x_0, y_0, sf, ori, phase, lum, by, rg, bg=0)
    s2 = sine_chrom(nx, ny, x_0, y_0, sf2, ori+rel_ori, phase2, 
                    lum2, by2, rg2)
    s = s1+s2
    if make_window:
        w = window(radius, x_0, y_0, nx, ny)
        s = w[..., np.newaxis]*s
    
    
    return s
def norm_im(im):
  im = im - im.min()
  im = im/im.max()
  return im
def scale_im(im):
  im = im - im.min()
  im = 2*im/im.max()
  im = im-1
  return im


mod = models.alexnet(pretrained=True).features[:1]
w = list(mod.parameters())[0].detach().numpy()
w_da = xr.DataArray(w, dims=('unit', 'channel', 'row', 'col'))
n_units = w.shape[0]
w_da_noise = w_da.copy(deep=True)
w_da_noise[...] = np.random.normal(size=w_da.shape, scale=0.1)

#%%
nx = ny = 11


stims = []

ori = list(np.linspace(0, 180-180/64, 64))
phase = list(np.linspace(0, 360-360/8, 8))
sf = list(np.logspace(np.log10(0.1), np.log10(.25), 8))
contrast = [1,]
lbr = [1,0,0]
make_window = False
param_nms = ['ori', 'sf', 'phase']
params = [ori, sf, phase]
for i, p in enumerate(params):
    if not type(p) is list:
        params[i] = [p,]
cart_prod_params = np.array(list(product(*params)))
da = xr.DataArray(np.zeros(tuple(len(p) for p in params )), 
         dims=param_nms,
         coords=params )

da_stims = da.squeeze(drop=True).expand_dims({'row':range(11), 
                                                  'col':range(11), 
                                                  'channel':range(3)})
da_stims = da_stims.transpose('ori', 'sf', 'phase', 'row', 'col', 'channel').copy()
x_0 = y_0 = 5


stim =[]
for p in (cart_prod_params):
    #plt.figure()
    ori, sf, phase = p
    im = sine_chrom_dual(nx, ny, x_0, y_0, sf, ori, phase, 
                        lbr[0], lbr[1], lbr[2], bg=0,
                        sf2=sf, rel_ori=0, phase2=phase, 
                        lum2=lbr[0], by2=lbr[1], rg2=lbr[2], 
                        make_window=make_window)
    #plt.imshow(norm_im(w[...,np.newaxis]*im))
    stim.append(im)
stims.append(stim)
stims = np.array(stims).squeeze()
rs = []
for stim, param in zip(stims, cart_prod_params):
    ori, sf, phase = param
    da_stims.loc[ori, sf, phase] = stim.copy()
    
      #%%
w_da = w_da/(w_da**2).sum(('channel', 'row', 'col'))**0.5
#da_stims = da_stims/(da_stims**2).sum(('channel', 'row', 'col'))**0.5
da_sig = da_stims.dot(w_da)
da_noise = da_stims.dot(w_da_noise)

#%%


n_units = len(da_sig.coords['unit'].values)       
unit_coords = list(product(range(n_units),range(n_units),))
mod_cor = xr.DataArray(np.zeros((n_units, n_units, 4, 2)),
                       dims=['unit_r', 'unit_c', 'vars', 'sn'],
                       coords=[range(n_units), range(n_units), 
                               ['cor', 'dyn', 'ori_ind', 'phase_ind'], 
                               ['s','n']])

sf_ind = 0
dim=('phase', 'ori')
for ind1, ind2 in tqdm((unit_coords)):
    for i, da in enumerate([da_sig, da_noise]):
        u1 = (da.isel(unit=ind1, sf=sf_ind).squeeze()**2).sum('phase')**0.5
        u2 = (da.isel(unit=ind2, sf=sf_ind).squeeze()**2).sum('phase')**0.5
        corr = auto_corr(u1, u2, dim=('ori'), pad=None)
        r = np.max(np.real(corr))
        mod_cor[ind1, ind2, 0, i] = r
        mod_cor[ind1, ind2, 1, i] = u1.std()*u2.std()
        mod_cor[ind1, ind2, 2:, i] = np.array(np.unravel_index(np.argmax(corr), corr.shape))
        


#%%
dfs = [mod_cor[...,i,:].to_dataframe(name=str(mod_cor.coords['vars'][i].values)).drop('vars', axis=1)
       for i in range(len(mod_cor.coords['vars']))]

df = pd.concat(dfs, 1)
m_inds = np.array([np.array(a) for a in df.index.values])
drop_inds = m_inds[:,0]<m_inds[:,1]
df_d = df[drop_inds]
df_d = df_d.reorder_levels([2,0,1])


#%%
def fz(r):
    return 0.5*(np.log((1+r)/(1-r)))
from scipy import stats

plt.figure(figsize=(4,3))
df = df_d.loc['s']
rs = []
for df in [df_d.loc['s'], df_d.loc['n']]:
    r,p = (stats.spearmanr(df['dyn'], df['cor']**2))
    rs.append(r)
    plt.scatter(df['dyn'], df['cor']**2, s=1);plt.semilogx();
    print(p)
plt.xlim(0.01, 1000)
plt.ylim(0,1.1)
plt.xlabel('Dynamic range')
plt.ylabel('$r^2_{ER}$')
plt.title('Trained $r=$' + str(np.round(rs[0],2)) + 
          ', untrained $r=$' + str(np.round(rs[1],2)))

inds = []
plt.legend(['Trained', 'Untrained'])

df = df_d.loc['s']
for i, ind in enumerate([0,-12,-100, 0,-12, -100]):
    if i<=2:
         ranks = (df['cor'].rank() + df['dyn'].rank()).sort_values()[::-1]
    else:
         ranks = (df['cor'].rank() - df['dyn'].rank()).sort_values()[::-1]


    u1, u2 = ranks.index.values[ind]
    inds.append([u1,u2])
    plt.scatter(df['dyn'][u1,u2], df['cor'][u1,u2]**2, s=10, c='r');plt.semilogx();


#%%
j=0
plt.figure(figsize=(3,8))
for ind in inds:
    u1, u2 = ind
    u1r = (da_sig.isel(unit=u1, sf=sf_ind).squeeze()**2).sum('phase')**0.5
    u2r = (da_sig.isel(unit=u2, sf=sf_ind).squeeze()**2).sum('phase')**0.5
    j+=1
    plt.subplot(6, 2, j)
    f = norm_im(w[u1])
    plt.imshow(np.transpose(f, (1,2,0)));plt.xticks([]);plt.yticks([])
    plt.title('Dyn='+str((u1r.var().values).round(2)) + 
              ', $r_{ER}^2$=' + str((df['cor'][u1,u2]**2).round(2)))
    #plt.imshow(f.mean(0), cmap='gray')
    j+=1
    plt.subplot(6, 2, j)
    f = norm_im(w[u2])
    plt.imshow(np.transpose(f, (1,2,0)));plt.xticks([]);plt.yticks([])
    plt.title('Dyn='+str((u2r.var().values).round(2)))
    #plt.imshow(f.mean(0), cmap='gray')

plt.tight_layout()
plt.savefig('example_filters.pdf')


#%%
j=0
plt.figure(figsize=(3,8))
for ind in inds:
    u1, u2 = ind
    j+=1
    plt.subplot(6, 2, j)
    a = da_sig.isel(unit=u1, sf=sf_ind)
    sta = (da_stims.isel(sf=sf_ind)*a).mean(('ori', 'phase'))
    plt.imshow(norm_im(sta));plt.xticks([]);plt.yticks([])
    plt.title('Dyn='+str(df['dyn'][u1,u2].round(2)) + 
              ', $r_{ER}^2$=' + str((df['cor'][u1,u2]**2).round(2)))
    j+=1
    plt.subplot(6, 2, j)
    a = da_sig.isel(unit=u2, sf=sf_ind)
    sta = (da_stims.isel(sf=sf_ind)*a).mean(('ori', 'phase'))
    plt.imshow(norm_im(sta));plt.xticks([]);plt.yticks([])



plt.tight_layout()
#plt.savefig('example_filters.pdf')

#%%
j=0
plt.figure(figsize=(4,8))
xticks = np.linspace(0, 180, 5)
for ind in inds:
    u1, u2 = ind
    j+=1

    plt.subplot(6, 2, j)
    u1r = da_sig.isel(unit=u1, sf=sf_ind).sel( phase=[0, 90, 180, 270], method='nearest')
    u1r = (u1r**2).sum('phase')**0.5
    u1r.plot.line(x='ori', add_legend=False);plt.title('');plt.gca().set_xticklabels([]);plt.xlabel('')
    if j==1:
        ''#plt.legend(['0','90','180','270'], loc='lower right', title='Phase (deg)')
    plt.gca().set_xticks(xticks)
    if j==11:
        
        plt.gca().set_xticklabels(np.round(xticks).astype(int))
        plt.xlabel('Orientation (deg)')
        plt.ylabel('Response')
    #plt.imshow(f.mean(0), cmap='gray')
    j+=1
    plt.subplot(6, 2, j)
    u2r = da_sig.isel(unit=u2, sf=sf_ind).sel( phase=[0, 90, 180, 270], method='nearest')
    u2r = (u2r**2).sum('phase')**0.5
    u2r.plot.line(x='ori', add_legend=False);plt.title('');plt.gca().set_xticks(xticks)

    plt.gca().set_xticklabels([]);plt.xlabel('');
    plt.title('Dyn='+str(df['dyn'][u1,u2].round(2)) + 
              ', $r_{ER}^2$=' + str((df['cor'][u1,u2]**2).round(2)))

        
    

plt.tight_layout()
plt.savefig('example_filters_resp.pdf')

#%% matching stim for a given unit
for ind in inds:
    plt.figure()
    shifts = df.loc[u1,u2]
    b_inds = np.unravel_index(np.argmax(u1r.values), np.shape(u1r))
    u1_stim = da_stims.isel(sf=sf_ind, ori=b_inds[0], phase=b_inds[1])
    s_stim = da_stims.roll({'phase':int(shifts['phase_ind']), 'ori':int(shifts['ori_ind'])})
    u2_match_stim = s_stim.isel(sf=sf_ind, ori=b_inds[0], phase=b_inds[1])
    
    plt.imshow(norm_im(u1_stim))
    plt.figure()
    plt.imshow(norm_im(u2_match_stim))


#%% now plot stim
from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(6., 6.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 8),  # creates 2x2 grid of axes
                 axes_pad=0.05,  # pad between axes in inch.
                 )

ims = da_stims.isel(sf=0).sel(ori=np.linspace(0, 180-180/8, 8), 
                              phase=[0, 90, 180, 270], method='nearest').stack(c = ('phase', 'ori')).transpose('c', 'row', 'col', 'channel')
for i, ax, im in zip(range(len(grid)), grid, ims):
    # Iterating over the grid returns the Axes.
    ax.imshow(norm_im(im));ax.set_xticks([]);ax.set_yticks([]);
    i,j = np.unravel_index(i, (4, 8))
    if i==0:
        ax.set_title(int(im.coords['c'].values.item()[1]))
    if j==0:
        ax.set_ylabel(int(im.coords['c'].values.item()[0]))
    
    if i==0 and j==0:
         ax.set_ylabel('phase (deg) \n' + str(int(im.coords['c'].values.item()[0])))
         ax.set_title( 'ori (deg) \n' + str(int(im.coords['c'].values.item()[1])))
plt.savefig('example_stim_phase_ori.pdf')       
#%%
from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(8., 3.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 8),  # creates 2x2 grid of axes
                 axes_pad=0.05,  # pad between axes in inch.
                 )

ims = da_stims.sel(ori=0,phase=0, method='nearest').transpose('sf', 'row', 'col', 'channel')
for i, ax, im in zip(range(len(grid)), grid, ims):
    # Iterating over the grid returns the Axes.
    ax.imshow(norm_im(im));ax.set_xticks([]);ax.set_yticks([]);
    sf = im.coords['sf'].values.item()
    
    ax.set_title(np.round(1/sf,1))
    if i==0:
        ax.set_title('Spatial Period\n ' + str(np.round(1/sf,1)) + str(' (pix)'))
        
plt.savefig('period.pdf')       



#%%
plt.figure(figsize=(3,9))
plt.subplot(311)
f = norm_im(w[55])
plt.imshow(np.transpose(f, (1,2,0)));plt.xticks([]);plt.yticks([])
plt.title('Filter')
plt.subplot(312)

u1 = da_sig.isel(unit=55, sf=sf_ind).sel( phase=[0, 90, 180, 270], method='nearest')
u1.plot.line(x='ori')
plt.title('Response')
plt.legend([0, 90, 180, 270], title='phase', loc='lower left')

plt.subplot(313)
(((u1**2)**0.5).mean('phase')).plot.line(x='ori')
plt.title('Avg. Response magnitude across phase')
plt.tight_layout()
