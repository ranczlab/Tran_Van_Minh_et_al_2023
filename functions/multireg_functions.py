import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

def p_adjust_bh(p):
    # Benjamini-Hochberg p-value correction for multiple hypothesis testing
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def multireg_all_areas(X, df, tiplist):
    # calculate multivariate linear regression and save results
    list_res = []
    list_r2 = []
    for thistip in tiplist:
        thisarea = thistip[4:]
        thisres = sm.OLS(df[thistip].astype(float), X.astype(float)).fit()
        list_res.append({'area' : thisarea, 'summary' : thisres.summary()})
        list_r2.append({'area' : thisarea, 'r2' : thisres.rsquared, 'r2_adj' : thisres.rsquared_adj, 'p-value' : thisres.f_pvalue})
        del thisres
    df_res = pd.DataFrame(list_res)
    df_r2 = pd.DataFrame(list_r2)
    df_r2['p_adj'] = p_adjust_bh(df_r2['p-value'])
    return df_res, df_r2

def run_multireg(df, list_prms, TIP_list):
    X = df[list_prms]
    X = sm.add_constant(X)

    this_df_res = pd.DataFrame()
    this_df_r2 = pd.DataFrame()

    this_df_res, this_df_r2 = multireg_all_areas(X, df, TIP_list)
    return this_df_res, this_df_r2