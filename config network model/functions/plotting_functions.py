import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.ticker as mtick


def plot_data_vs_fit(xdata, ydata, fit, xlabel, ylabel, title, **kwargs):
    # plot data and fit 
    f = plt.figure(figsize = (7,5))
    plt.scatter(xdata, ydata, label='data')
    plt.plot(xdata, fit, label='fit', color='orange')
    if 'ymin' in kwargs and 'ymax' in kwargs:
        ymin = kwargs['ymin']
        ymax = kwargs['ymax']
        plt.ylim(ymin, ymax)
    plt.xlabel(xlabel,fontsize = 12)
    plt.ylabel(ylabel,fontsize = 12)
    plt.title(title,fontsize = 18)
    ax = plt.gca()
    sns.despine()
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    return f, ax

def format_scientnot():
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
    plt.locator_params(axis='x', nbins=7)   



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


def plot_input_vs_starter(df, islog, title):
    # make single plot # inputs vs # starters (natural or logscale)
    f, ax = plt.subplots(figsize=(7, 7))
    ax.plot(df['num_starters'], df['num_inputs']['mean'], 'x')
    ax.errorbar(df['num_starters'], df['num_inputs']['mean'], df['num_inputs']['std'])
    
    if islog == 0:
        ax.set_xlabel('# starters')
        ax.set_ylabel('# inputs')
    else:
        ax.set_xlabel('Log(# starters)')
        ax.set_ylabel('Log (# inputs)')
    
    ax.set_title(title)
    plt.show()
    return f


def group_and_plot(df_res, title):
    # group by number of starters
    result = df_res.groupby(['num_starters'], as_index=False).agg(
                      {'num_inputs':['mean','std']})
    
    # same for log-transformed data
    df_res_log = np.log10(df_res)
    result_log = df_res_log.groupby(['num_starters'], as_index=False).agg(
                      {'num_inputs':['mean','std']})
    
    # plot
    f = plot_input_vs_starter(result, 0, title)
    
    f_log = plot_input_vs_starter(result_log, 1, title)
    return f, f_log

def make_plot_sp(df, islog, d_non_dep, Ntarg, param):
    # param : parameters used for grouping (in df, this is the only model param with non-unique values)
    # d_non_dep : dictionary with names and values of other parameters (e.g. d _non_dep= {'nspines'  : 10, 'gamma' : 1.5})
    f, ax = plt.subplots(figsize=(7, 7))
    gdf = df.groupby([param])
    # calculate mean and std per number of starters
    plots = gdf.plot(x = 'num_starters', y = ('num_inputs mean'), yerr = ('num_inputs std'), ax = ax)

    legendNames = list(gdf.groups.keys())  
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1, 1.01), fontsize = 15)
    leg.set_title(param, prop={'size':15})
    for txt, name in zip(ax.legend_.texts, legendNames):
        txt.set_text(name)
        s_title = 'starter size : ' + str(Ntarg) 
        for k, v in d_non_dep.items():
            s_title += ', '+k+': ' + str(v)       
        ax.set_title(s_title, fontsize = 15)

    if islog == 0:    
        ax.set_xlabel('# starters', fontsize = 15)
        ax.set_ylabel('# inputs', fontsize = 15)
    else:
        ax.set_xlabel('Log(# starters)', fontsize = 15)
        ax.set_ylabel('Log(# inputs)', fontsize = 15)
    return gdf, f, ax
