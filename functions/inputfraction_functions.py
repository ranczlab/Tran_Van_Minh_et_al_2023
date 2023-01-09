import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import LinearModel

def rss_fit(mod_y, data_y):
    return np.sum(np.square(mod_y - data_y))

def chow_test(data_y1, data_y2, data_c, bestfit1, bestfit2, bestfit_c):
    N1 = len(data_y1)
    N2 = len(data_y2)
    k = 2
    rss_1 = rss_fit(bestfit1, data_y1)
    rss_2 = rss_fit(bestfit2, data_y2)
    rss_c = rss_fit(bestfit_c, data_c)
    chow_stat = ((rss_c - (rss_1 + rss_2))/k)/((rss_1 + rss_2)/(N1+N2-2*k))
    pvalue = stats.f.sf(chow_stat, k, N1+N2-2*k)
    return chow_stat, pvalue

def make_input_frac(df, areas):
    df_fraction = df.copy()
    for area in list(areas):
        df_fraction[area] = df[area]/df['input']
    return df_fraction

def make_convergence_index(df, areas):
    df_ci = df.copy()
    for area in list(areas):
        df_ci[area] = df[area]/df['starter']
    return df_ci  


def fit_linear_model(datax, datay):
    mod_lm = LinearModel()
    this_par = mod_lm.guess(datay, x = datax)
    this_result_lm = mod_lm.fit(datay, this_par, x = datax)
    fit_params = []
    fit_params.append((this_result_lm.params['intercept'].value, this_result_lm.params['slope'].value))
    return this_result_lm, fit_params

def split_per_bp(df, area, bp):
    df_temp_above_area = df[area][df['starter']>bp]
    df_temp_above_starter = df['starter'][df['starter']>bp]
    df_temp_below_area = df[area][df['starter']<=bp]
    df_temp_below_starter = df['starter'][df['starter']<=bp]

    return df_temp_below_area, df_temp_below_starter, df_temp_above_area, df_temp_above_starter

def fit_per_starternum(df, area, bp):
    thisdf_temp_below_area, thisdf_temp_below_starter, thisdf_temp_above_area, thisdf_temp_above_starter = split_per_bp(df, area, bp)

    fit_params_lm_below = []
    fit_params_lm_above = []
    fit_params_lm_all = []
    
    # linear fits 
    this_result_lm, fit_params_lm_all = fit_linear_model(df['starter'].values, df[area].values)
    
    if thisdf_temp_above_area.shape[0]>1 and not thisdf_temp_above_area.empty:
        datax_above = thisdf_temp_above_starter.values
        datay_above = thisdf_temp_above_area.values
        this_result_lm_above, fit_params_lm_above = fit_linear_model(datax_above, datay_above)
    else:
        fit_params_lm_above.append((np.nan, np.nan))
    
    if thisdf_temp_below_area.shape[0]>1 and not thisdf_temp_below_area.empty:
        datax_below = thisdf_temp_below_starter.values
        datay_below = thisdf_temp_below_area.values
        this_result_lm_below, fit_params_lm_below = fit_linear_model(datax_below, datay_below)
    else:
        fit_params_lm_below.append((np.nan, np.nan))  

    return this_result_lm, this_result_lm_above, this_result_lm_below, fit_params_lm_all, fit_params_lm_above, fit_params_lm_below

def fit_lm_thresh(df, area, starternum): 
# make two linear fits of inputs fraction vs starter depending on starter numbers (above and below starternum)

  # split df by starter numbers  
    df_above_thresh_y, df_above_thresh_x, df_below_thresh_y, df_below_thresh_x = split_per_bp(df, area, starternum)
  # linear fits - full range
    results_all, pars_all = fit_linear_model(df['starter'].values, df[area].values)
  # fit above starternum
    results_above, pars_above = fit_linear_model(df_above_thresh_x, df_above_thresh_y)
  # fit below starternum            
    results_below, pars_below = fit_linear_model(df_below_thresh_x, df_below_thresh_y)
    
  # chow test
    chow_stat, chow_pval = chow_test(df_below_thresh_y, df_above_thresh_y, df[area].to_numpy(), results_below.best_fit, results_above.best_fit, results_all.best_fit)
    
    return pars_all, pars_above, pars_below, chow_stat, chow_pval

def fit_lm_thresh_allareas(df, starternum, arealist):
    parsall = []
    parsabove = []
    parsbelow = []
    chowresults = []
    for thisarea in arealist:
        thispars_all, thispars_above, thispars_below, thischow_stat, thischow_pval = fit_lm_thresh(df, thisarea, starternum)
        parsall_dict = {'area' : thisarea, 'threshold': starternum, 'intercept' : thispars_all[0][0], 'slope' : thispars_all[0][1]}
        parsabove_dict = {'area' : thisarea, 'threshold': starternum, 'intercept' : thispars_above[0][0], 'slope' : thispars_above[0][1]}
        parsbelow_dict = {'area' : thisarea, 'threshold': starternum, 'intercept' : thispars_below[0][0], 'slope' : thispars_below[0][1]}
        chow_dict = {'area': thisarea, 'threshold': starternum, 'chow_stat': thischow_stat, 'chow_pval': thischow_pval}
        
        parsall.append(parsall_dict)
        parsabove.append(parsabove_dict)
        parsbelow.append(parsbelow_dict)
        chowresults.append(chow_dict)
        
    return pd.DataFrame(parsall), pd.DataFrame(parsabove), pd.DataFrame(parsbelow), pd.DataFrame(chowresults)   


def find_breakpoint(df, arealist, bplist):
    # iterate through a list of proposed breakpoints
    parsalldf = pd.DataFrame()
    parsabovedf = pd.DataFrame()
    parsbelowdf = pd.DataFrame()
    chowdf = pd.DataFrame()
    for bp in bplist:
        thisparsall, thisparsabove, thisparsbelow, thischow = fit_lm_thresh_allareas(df, bp, arealist)
        
        parsalldf = pd.concat([parsalldf, thisparsall], axis = 0).reset_index(drop = True)
        parsabovedf = pd.concat([parsabovedf, thisparsabove], axis = 0).reset_index(drop = True)
        parsbelowdf = pd.concat([parsbelowdf, thisparsbelow], axis = 0).reset_index(drop = True)
        chowdf = pd.concat([chowdf, thischow], axis = 0).reset_index(drop = True)
        
    return parsalldf, parsabovedf, parsbelowdf, chowdf


def plot_fraction_and_chow(listareas, listdf, listnames, ymin, ymax, bp, savepath, title):
    # for all areas in listareas, make subplot with fits for each area and overlay of areas to compare

    savepathfig = os.path.join(savepath,'figures')
    savepathcsv = os.path.join(savepath,'csv')

    df_fit_params_final = pd.DataFrame() # to save parameters
    df_chowtest_final = pd.DataFrame()

    for area in listareas:
    
        f = plt.figure(figsize=(10,5))
        ax = {}
        # numdf = len(listdf)
        # ax[numdf+1] = f.add_subplot(1, numdf+2, numdf+1)
        # ax[numdf+2] = f.add_subplot(1, numdf+2, numdf+2)
        
        this_df_pooled_area = pd.DataFrame()
        this_chow_pooled_area = pd.DataFrame()
        for idx, thisdf in enumerate(listdf):
            chow_results = []
            thisdf_temp_below_area, thisdf_temp_below_starter, thisdf_temp_above_area, thisdf_temp_above_starter = split_per_bp(thisdf, area, bp)
            this_result_lm, this_result_lm_above, this_result_lm_below, fit_params_lm_all, fit_params_lm_above, fit_params_lm_below = fit_per_starternum(thisdf, area, bp)
        
            # save fit parameters
            df_fit_params_above = pd.DataFrame(fit_params_lm_above, columns = ['intercept', 'slope'])
            df_fit_params_above = pd.concat([df_fit_params_above], keys = ['starter+'], axis = 1)
            
            df_fit_params_below = pd.DataFrame(fit_params_lm_below, columns = ['intercept', 'slope'])
            df_fit_params_below = pd.concat([df_fit_params_below], keys = ['starter-'], axis = 1)
            
            df_fit_params_all = pd.DataFrame(fit_params_lm_all, columns = ['intercept', 'slope'])
            df_fit_params_all = pd.concat([df_fit_params_all], keys = ['startersall'], axis = 1)
            
            this_df_fit_params = pd.concat([df_fit_params_all, df_fit_params_below, df_fit_params_above], axis = 1)
            this_df_fit_params = pd.concat([this_df_fit_params], axis = 1, keys = [listnames[idx]])
            this_df_pooled_area = pd.concat([this_df_pooled_area, this_df_fit_params], axis = 1)
            
            # calculate chow test
            thischow, thispvalue = chow_test(thisdf_temp_below_area, thisdf_temp_above_area, thisdf[area].to_numpy(), this_result_lm_below.best_fit, this_result_lm_above.best_fit, this_result_lm.best_fit)
            chow_results.append((thischow, thispvalue))
            
            # save chow-test statistics and p-value
            df_this_chow = pd.DataFrame(chow_results, columns = ['Chow stat', 'p-value'])
            df_this_chow = pd.concat([df_this_chow], axis = 1, keys = [listnames[idx]])
            this_chow_pooled_area = pd.concat([this_chow_pooled_area, df_this_chow], axis = 1)
            
            # plots
            ax[idx+1] = f.add_subplot(1, 1, 1)
            ax[idx+1].scatter(thisdf['starter'], thisdf[area], marker = "s", color='grey', label = 'all data points')
            ax[idx+1].plot(thisdf['starter'].values, this_result_lm.best_fit, color='grey',ls = '--')

            ax[idx+1].set_xlabel('#starters')
            ax[idx+1].set_ylabel('fraction of total inputs')
            ax[idx+1].set_title(listnames[idx])
            if area in ['VIS', 'Thal', 'Dist_ctx', 'all_VIS']:
                ax[idx+1].set_ylim(ymin, 0.5)
            elif (area == 'VISp') or (area == 'RSP'):
                ax[idx+1].set_ylim(ymin, ymax)
            else:
                ax[idx+1].set_ylim(ymin, 0.12)

            if not thisdf_temp_above_area.empty :
                ax[idx+1].scatter(thisdf_temp_above_starter.values, thisdf_temp_above_area.values, marker = "+", label = ', # starters > ' + str(bp) ,color='orange')
            if thisdf_temp_above_area.shape[0]>1:
                ax[idx+1].plot(thisdf_temp_above_starter.values, this_result_lm_above.best_fit, color='orange', ls = '--')        
            ax[idx+1].plot(thisdf_temp_below_starter.values, this_result_lm_below.best_fit, color='blue', ls = '--')        

            if not thisdf_temp_below_area.empty:
                ax[idx+1].scatter(thisdf_temp_below_starter.values, thisdf_temp_below_area.values, marker = "x", label = ', # starters <= ' + str(bp),color='blue')
            plt.legend()
            


            f.suptitle(area)

            plt.savefig(os.path.join(savepathfig, 'input_fraction_' + title +'_starternum_' + listnames[idx] + '_input_' + area + '.png'))
            plt.savefig(os.path.join(savepathfig, 'input_fraction_' + title +'_starternum_' + listnames[idx] + '_input_' + area + '.eps'), format ='eps')
        
        this_df_pooled_area = pd.concat([this_df_pooled_area], axis = 1, keys = [area])
        df_fit_params_final = pd.concat([df_fit_params_final, this_df_pooled_area], axis = 1)
        this_chow_pooled_area = pd.concat([this_chow_pooled_area], axis = 1, keys = [area])
        df_chowtest_final = pd.concat([df_chowtest_final, this_chow_pooled_area], axis = 1)
    df_fit_params_final = df_fit_params_final.reorder_levels([1,0,2,3], axis = 1)
    df_chowtest_final =  df_chowtest_final.reorder_levels([1,0,2], axis = 1)
    
    return this_chowtest_final, df_fit_params_final