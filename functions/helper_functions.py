import os
import numpy as np
import pandas as pd

def dir_check(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def import_main_dataset(pathdata):
    df_raw = pd.read_hdf(os.path.join(pathdata, 'df_filt_ipsi.h5'))
    df_raw = df_raw.sort_values(by='starter').reset_index(drop = True)
    areas = list(df_raw.columns[10:])
    areas_with_pre = areas + ['input']
    df_raw_v1 = df_raw[(df_raw.area == 'V1') & (df_raw.keep != 0)]
    df_raw_pm = df_raw[(df_raw.area == 'PM') & (df_raw.keep != 0)]
    df_all = df_raw[df_raw.keep != 0]
    df_v1 = df_raw_v1.copy()
    df_pm = df_raw_pm.copy()    
    df_v1.set_index('group', inplace=True)
    df_pm.set_index('group', inplace=True)
    df_all.set_index('group', inplace=True)
    
    return areas, areas_with_pre, df_raw, df_all, df_v1, df_pm


def make_log_df(df):
    # make dataframe with log10 - transformed data (# inputs and # starters)
    # INPUT :  - df, dataframe containing natural scale data - columns input and starter are to be transformed
    # OUTPUT : - df_log
    df_log = df.copy()
    df_log['input'] = np.log10(df_log['input'])
    df_log['starter'] = np.log10(df_log['starter'])
    df_log = df_log.replace(-np.inf, np.nan) # if some starter numbers are 0, log10 gives -inf. Replace by NaN.
    return df_log

def make_log_df_full(df, areas):
    # make dataframe with log10 - transformed data (# inputs for each area and # starters)
    # INPUT :  - df, dataframe containing natural scale data - columns input and starter are to be transformed
    # OUTPUT : - df_log
    df_log = df.copy()
    df_log[list(areas)] = np.log10(df_log[list(areas)])
    try:
        df_log['pre'] = np.log10(df_log['pre'])
        df_log['input'] = np.log10(df_log['input'])
        df_log['starter'] = np.log10(df_log['starter'])
    except:
        pass
    df_log = df_log.replace(-np.inf, np.nan) # if some starter numbers are 0, log10 gives -inf. Replace by NaN.
    return df_log

def save_dflist_to_csv(dflist, filelist, savepath):
    # INPUTS :  - list of dataframes to save
    #           - list of file names (to save as)
    #           - directory to save csvs

    for i, df in enumerate(dflist):
        df.to_csv(os.path.join(savepath, filelist[i] + '.csv'))


def create_subfolders(savepath):
    # create classic subfolder structure (figures, DF, csv) in given folder
    savepath_fig = os.path.join(savepath, 'figures')
    dir_check(savepath_fig)
    savepath_df = os.path.join(savepath, 'DF')
    dir_check(savepath_df)
    savepath_csv = os.path.join(savepath, 'csv')
    dir_check(savepath_csv)
    return savepath_fig, savepath_df, savepath_csv

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

