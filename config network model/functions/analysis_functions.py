import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def flin(x, A, B): #linear fit
    return A*x + B


def filter_df_numstart(df_cond, numstart, df_path):
    # filter df with list of conditions per number of targets
    df_cond_starters = df_cond[df_cond['targets'] == numstart]

    this_df_starters = pd.DataFrame()
    for file in df_cond_starters['file']:
        thisdf = pd.read_hdf(os.path.join(df_path, file))
        this_df_starters = pd.concat([this_df_starters, thisdf])
        del thisdf
    return this_df_starters


def make_log_df(df):
    df_log = df.copy()
    
    df_log['num_starters'] = np.log10(df_log['num_starters'])
    df_log['num_inputs'].replace(0, 1, inplace=True)
    df_log['num_inputs'] = np.log10(df_log['num_inputs'])

    return df_log


def make_res_df_param(df, param):
    # assumes multiple values for param, unique values for other parameters
    col_list = list(df.columns)
    var_list = ['num_starters', 'num_inputs']
    var_list.extend([param])
    nonvar_list = [x for x in col_list if x not in set(var_list)]
    # print(var_list)
    # print(nonvar_list)
    this_df_res = pd.DataFrame()
    
    for par in nonvar_list:
        if len(np.unique(df[par])) > 1:
            print('non unique conditions for parameter ' + par)
    
    param_range = df[param].unique()
    for thisp in param_range:
        thisres =  df[df[param] == thisp].groupby(['num_starters'], as_index=False).agg({'num_inputs':['mean','std']})
        thisres[param] = thisp
        for par in nonvar_list:
            thisres[par] = df[par].unique()[0]
        this_df_res = pd.concat([this_df_res, thisres], axis = 0)
    this_df_res.columns = [' '.join(col).strip() for col in this_df_res.columns.values]
    return this_df_res

def make_log_fits(gdf_log, strdep, d_non_dep, Ntarg):
    # for each group, fit the log-transformed data between minimal number of inputs and 85% of the max number of inputs
    # strdep : string of unique parameter
    # d_non_dep : dictionary with names and values of other parameters (e.g. d _non_dep= {'nspines'  : 10, 'gamma' : 1.5})
    
    fits_list = []
    for thisdeg in gdf_log.groups.keys():
        this_res = gdf_log.get_group(thisdeg)
        thrs = 0.9 * this_res['num_inputs mean'].max()
        thrsm = this_res['num_inputs mean'].min()
        x = this_res['num_starters'][(this_res['num_inputs mean'] < thrs) & (this_res['num_inputs mean']> thrsm)]
        y = this_res['num_inputs mean'][(this_res['num_inputs mean'] < thrs) & (this_res['num_inputs mean']> thrsm)]
        if x.shape[0]>=2:
            popt, pcov = curve_fit(flin, x, y)
            this_fit_res = {'targets': Ntarg, 'slope' : popt[0], 'intercept' : popt[1], strdep: thisdeg}
            for k, v in d_non_dep.items():
                this_fit_res.update({k : v})
        else:
            this_fit_res = {'targets': Ntarg, 'slope' : np.nan, 'intercept' : np.nan,  strdep: thisdeg}
            for k, v in d_non_dep.items():
                this_fit_res.update({k : v})
        fits_list.append(this_fit_res)
    
    fits_df = pd.DataFrame(fits_list)

    return fits_df