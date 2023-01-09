import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import matplotlib.colors as colors
import seaborn as sns
from scipy import stats
sns.set_style("ticks")
hfont = {'fontname':'Myriad Pro'}
from matplotlib import colors

# define colormaps
colorlist = ["sea blue", "burnt orange","amber", "windows blue",  "greyish", "faded green", "dusty purple"]
colorlist1 = ["sea blue", "burnt orange","amber"]
colorlist2 = ["windows blue",  "greyish", "faded green", "dusty purple"]
colorlist3 = ["sea blue", "burnt orange","amber", "dusty purple"]
cmap_all  = colors.ListedColormap(sns.xkcd_palette(colorlist))
cmap1  = colors.ListedColormap(sns.xkcd_palette(colorlist1))
cmap2  = colors.ListedColormap(sns.xkcd_palette(colorlist2))
cmap3  = colors.ListedColormap(sns.xkcd_palette(colorlist3))

def plot_data_vs_fit(xdata, ydata, fit, xlabel, ylabel, title, **kwargs):
    # plot data and fit 
    f = plt.figure(figsize = (7,5))
    plt.scatter(xdata, ydata, label='data')
    plt.plot(xdata, fit, label='fit', color='orange')
    if 'ymin' in kwargs and 'ymax' in kwargs:
        ymin = kwargs['ymin']
        ymax = kwargs['ymax']
        plt.ylim(ymin, ymax)
    plt.xlabel(xlabel,fontsize = 12, **hfont)
    plt.ylabel(ylabel,fontsize = 12, **hfont)
    plt.title(title,fontsize = 18, **hfont)
    ax = plt.gca()
    sns.despine()
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    return f, ax

def qq_plot_res(fit_res, title):
    # make qq plots from residuals 
    f = plt.figure()
    ax1 = f.add_subplot(1,1,1)
    resplot = stats.probplot(fit_res, dist="norm", plot=plt)
    ax1.set_ylabel('residuals',fontsize = 12)
    ax1.set_xlabel('quantiles',fontsize = 12)
    ax1.set_title(title,fontsize = 18)
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)

    
def stdres_vs_fitted(fit, fit_res, title):
    # plot standardized residuals vs fitted data
    f = plt.figure()
    ax1 = f.add_subplot(1,1,1)
    plt.scatter(fit, fit_res/np.std(fit_res))
    x1, x2 = plt.gca().get_xlim()
    ax1.axhline(y=0, xmin=0, xmax=x2, ls='--')
    ax1.set_ylabel('standardized residuals',fontsize = 12)
    ax1.set_xlabel('fitted',fontsize = 12)
    ax1.set_title(title,fontsize = 18)   

def format_scientnot():
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
    plt.locator_params(axis='x', nbins=7)   

def plot_multivar(list_res, title, figname, savepath, cm_choice):
    if cm_choice == 0:
        cm = cmap_all
    elif cm_choice == 1:
        cm = cmap1
    else:
        cm = cmap2

    f, ax = plt.subplots(figsize=(7, 7)) 
    list_res.plot(kind='bar', ax = ax, width = 0.8, cmap = cm)
    plt.ylabel('r2')
    plt.legend(fontsize = 12)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
        
    plt.ylim(0,0.9)
        
    plt.title(title,fontsize = 20)
    f.savefig(os.path.join(savepath, figname + '.png'), bbox_inches='tight')
    f.savefig(os.path.join(savepath, figname + '.eps'), bbox_inches='tight', format = 'eps')


def plot_chow_by_bp(thischowdf, title):
    colormap = plt.cm.viridis

    thiscolorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(thischowdf['area'].unique()))]

    f, ax = plt.subplots(figsize=(10,10))
    thiscdict = dict(zip(thischowdf['area'].unique(), thiscolorlist))

    for g in thischowdf['area'].unique():
        ax.plot(thischowdf['threshold'][thischowdf['area']==g], thischowdf['chow_pval'][thischowdf['area']==g], 'o-', c = thiscdict[g], label = g)
    ax.set_ylim(0,1.1)
    ax.axhline([0.05], c='k', ls = '--')
    ax.set_title('Chow Test - ' + title + ' areas')
    ax.set_xlabel('# starters breakpoint', fontsize = 12)
    ax.set_ylabel('p_value', fontsize = 12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.07))

    return f, ax