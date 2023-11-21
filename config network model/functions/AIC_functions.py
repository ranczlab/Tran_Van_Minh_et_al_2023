import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
import helper_functions as hfn
import plotting_functions as plotfn
from scipy import stats
from lmfit.models import PowerLawModel
from lmfit.models import LinearModel


def expfunc(x, a, b):
    return a * np.exp(- x/b) 

def import_data(pathdata):
    df_raw = pd.read_excel(os.path.join(pathdata, 'data.xlsx'))
    try:
        df_raw = df_raw[['ID', 'starter', 'input', 'condition', 'cell_type']]
    except:
        df_raw = df_raw[['ID', 'starter', 'input', 'condition']]
    df_raw = df_raw.sort_values(by='starter').reset_index(drop = True)
    return df_raw

def import_data_areas(pathdata):
    df_raw = pd.read_excel(os.path.join(pathdata, 'data_areas.xlsx'))
    # df_raw = df_raw[['ID', 'starter', 'input', 'condition']]
    df_raw = df_raw.sort_values(by='starter').reset_index(drop = True)
    return df_raw

def calc_AICc_r2(fitAIC, fitnvarsys, fitndata, fitres, ydata):
    # calculate AICc from fit results
    # OUTPUTS: AICc, r**2
    AICc = fitAIC+ ((2*(fitnvarsys+1)**2) + 2*(fitnvarsys+1)/(fitndata - (fitnvarsys+1) -1))
    rsquare = 1-fitres.var()/ np.var(ydata)
    return AICc, rsquare

def AIC_RSS(k, n, rss): #rss= chi2
    AIC=2*k+n*np.log(rss/n)#2*k+n*np.log(rss)
    return AIC2

def fit_summary_df(result, ydata, area):
    # calculate AICc and R^2
    AICc, rsquare = calc_AICc_r2(result.aic, result.nvarys, result.ndata, result.residual, ydata)
    fit_summary = []
    fit_summary.append((area, result.aic, AICc, result.chisqr, result.bic, rsquare))
    df_AIC = pd.DataFrame(fit_summary, columns = ['area', 'AIC', 'AICc', 'Chi2','BIC','R2'])
    return df_AIC

def fit_and_saveprms_noplot(datax, datay, thisarea, model, title, figtitle, save_path, y1, y2, isexp, logscale):
    # create subfolders to save figures and data
    save_path_fig, save_path_df, save_path_csv = hfn.create_subfolders(save_path)

    # make fit
    if isexp == 1:
        pars = model.make_params(a = 1000, b = -500)
    else:
        pars = model.guess(datay, x = datax)
        if model.name == 'Model(lognormal)':
            pars['amplitude'].set(6000, vary = True)
            pars['sigma'].set(5, vary = True)
            pars['center'].set(18, vary = True)

    result = model.fit(datay, pars, x = datax)

    # calculate AICc and R^2
    AICc, rsquare = calc_AICc_r2(result.aic, result.nvarys, result.ndata, result.residual, datay)
    df_AIC = fit_summary_df(result, datay, thisarea)

    # create dataframes with fit parameters 
    fit_params = []
    fit_params.append(thisarea)
    fit_params.extend([result.params[key].value for key in result.params.keys()])
    prm_columns = ['area']
    prm_columns.extend([key for key in result.params.keys()])
    df_params = pd.DataFrame([fit_params], columns = prm_columns)    

    df_fit = pd.concat([df_AIC, df_params], axis=1)
    df_fit = df_fit.loc[:, ~df_fit.columns.duplicated()]
    
    df_fit.to_hdf(os.path.join(save_path_df, 'df_' + figtitle + '.h5'), key = figtitle, mode = 'w')
    df_fit.to_csv(os.path.join(save_path_csv, 'df_' + figtitle + '.csv'))
    
    return df_fit, result, pars, AICc, rsquare

def fit_and_saveprms(datax, datay, thisarea, model, title, figtitle, save_path, y1, y2, isexp, logscale):
    # create subfolders to save figures and data
    save_path_fig, save_path_df, save_path_csv = hfn.create_subfolders(save_path)
    if logscale == 1:
        xlabel = r'$\mathrm{Log}_{10}(starters)$'
        ylabel = r'$\mathrm{Log}_{10}(inputs)$'
    else:
        xlabel = 'starters'
        ylabel = 'inputs'

    # make fit
    if isexp == 1:
        pars = model.make_params(a = 1000, b = -500)
    else:
        pars = model.guess(datay, x = datax)
        if model.name == 'Model(lognormal)':
            pars['amplitude'].set(6000, vary = True)
            pars['sigma'].set(5, vary = True)
            pars['center'].set(18, vary = True)

    result = model.fit(datay, pars, x = datax)

    print(result.fit_report())

    # plot
    plotfn.plot_data_vs_fit(datax, datay, result.best_fit, xlabel, ylabel, title, ymin = y1, ymax = y2)
    if logscale == 0:
        plotfn.format_scientnot() # format axis to use scientific notation
    plt.savefig(os.path.join(save_path_fig, figtitle + '.png'))
    plt.savefig(os.path.join(save_path_fig, figtitle + '.eps'), format = 'eps')

    # calculate AICc and R^2
    AICc, rsquare = calc_AICc_r2(result.aic, result.nvarys, result.ndata, result.residual, datay)
    df_AIC = fit_summary_df(result, datay, thisarea)

    # create dataframes with fit parameters 
    fit_params = []
    fit_params.append(thisarea)
    fit_params.extend([result.params[key].value for key in result.params.keys()])
    prm_columns = ['area']
    prm_columns.extend([key for key in result.params.keys()])
    df_params = pd.DataFrame([fit_params], columns = prm_columns)    

    df_fit = pd.concat([df_AIC, df_params], axis=1)
    df_fit = df_fit.loc[:, ~df_fit.columns.duplicated()]
    
    df_fit.to_hdf(os.path.join(save_path_df, 'df_' + figtitle + '.h5'), key = figtitle, mode = 'w')
    df_fit.to_csv(os.path.join(save_path_csv, 'df_' + figtitle + '.csv'))
    
    return df_fit, result, pars, AICc, rsquare

def fit_all_areas(df, listareas, model, isexp, logscale, strmodel, savepath, targetc): 
    dictmodel = {'pl' : 'power-law', 'lin' : 'linear', 'exp' : 'exponential', 'quad' : 'quadratic'}
    # targetc = 0 for whole brain
    # targetc = 1 for V1
    # targetc = 2 for PM
    strtitle = dictmodel[strmodel]
    if logscale == 0:
        strscale = 'linear'
        strsctitle = 'lin'
    else:
        strscale = 'log'
        strsctitle = 'log'
    if targetc == 0:
        strtarget = 'alltargets'
    elif targetc == 1:
        strtarget = 'V1'
    elif targetc == 2:
        strtarget = 'PM'
    
    df_results = pd.DataFrame()
    for area in listareas:
        if logscale == 0:
            ymax = math.ceil(df[area].max() / 5000.0) * 5000.0
        else:
            ymax = math.ceil(df[area].max())
        if area == 'input':
            strarea = 'allinputs'
        else:
            strarea = area
        datax = df['starter'].values
        datay = df[area].values
        title = strscale + ' scale, ' + strtitle +' fit, target : '+ strtarget +', input = ' + strarea
        figtitle = strsctitle + 'scale_' + strmodel +'fit_'+ strtarget + '_input_' + strarea
        y1 = 0
    
        thisdf_fit, thisresult, thispars, thisAICc, thisrsquare = fit_and_saveprms(datax, datay, area, model, title, figtitle, savepath, y1, ymax, isexp, logscale)
        df_results = pd.concat([df_results, thisdf_fit], axis = 0)    
    return df_results


def lognormal10(x, mu, sigma):
    a = np.log10(np.e) / (x * sigma * np.sqrt(2*np.pi))
    b = -((np.log10(x)-np.log10(mu))**2)/(2*sigma**2)

    return float(a*np.exp(b))

def lognorm10pdf(x, mu, sigma):
    pdf = [lognormal10(x[i], mu[i], sigma) for i in range(len(x))]
    return pdf

def fit_log_lm_noplot(area, df):
    mod_lm = LinearModel()
    # linear fit of log transformed data
    y = df[area].values
    x = df['starter'].values
    thispars = mod_lm.guess(np.log10(y), x = np.log10(x))
    thisresult = mod_lm.fit(np.log10(y), thispars, x = np.log10(x))

    a_lm = 10**(thisresult.params['intercept'].value)#np.exp(thisresult.params['linintercept'].value)
    b_lm = thisresult.params['slope'].value
    sd_lm = np.std(np.log10(y) - (np.log10(a_lm) + b_lm * np.log10(x)), ddof = 1)
    k = 3 # Number of parameters : Here 2 parameters for the fit + error 
    n=len(x)
    l_logn = np.sum(np.log(lognorm10pdf(y,  a_lm * x ** b_lm, sd_lm)))#(stats.lognorm.logpdf(y, s = sd_lr, scale = a_lr * x ** b_lr)) # scale = np.log(mu)!!!
    
    AIC_logn = 2 * k - 2 * l_logn
    AICc_logn = 2 * k - 2 * l_logn + 2 * k * (k + 1) / (n - k - 1)
    return thisresult, a_lm, b_lm, l_logn, AIC_logn, AICc_logn

def fit_unt_pl_noplot(area, df):
    mod_pl = PowerLawModel()
    thisdf = df.dropna()
    thispars = mod_pl.guess(thisdf[area].values, x = thisdf['starter'].values)

    thisresult = mod_pl.fit(thisdf[area].values, thispars, x=thisdf['starter'].values)

    a_nlr = thisresult.params['amplitude'].value
    b_nlr = thisresult.params['exponent'].value
    sd_nlr = np.std(thisdf[area].values - a_nlr * (thisdf['starter'].values)**b_nlr)
    k = 3 # Number of parameters : Here 2 parameters for the fit + error 
    n = len(thisdf['starter'].values)

    l_norm = np.sum(stats.norm.logpdf(thisdf[area].values, loc = a_nlr*(thisdf['starter'].values)**b_nlr, scale = sd_nlr)) # scale = np.log(mu)!!!
    AIC_norm = 2 * k - 2 * l_norm
    AICc_norm = 2 * k - 2 * l_norm + 2 * k * (k + 1) / (n - k - 1)

    return thisresult, a_nlr, b_nlr,l_norm, AIC_norm, AICc_norm

def fit_log_lm(area, df):
    mod_lm = LinearModel()
    # linear fit of log transformed data
    y = df[area].values
    x = df['starter'].values
    thispars = mod_lm.guess(np.log10(y), x = np.log10(x))
    thisresult = mod_lm.fit(np.log10(y), thispars, x = np.log10(x))
    #print(thisresult.fit_report())

    if area == 'input':
        plotfn.plot_data_vs_fit(np.log10(x), np.log10(y), thisresult.best_fit,'starters','inputs','linear fit, Log scale, all targets, all inputs')
    else:
        plotfn.plot_data_vs_fit(np.log10(x), np.log10(y), thisresult.best_fit,'starters','inputs','linear fit, Log scale, all targets, inputs from '+ area)

    a_lm = 10**(thisresult.params['intercept'].value)#np.exp(thisresult.params['linintercept'].value)
    b_lm = thisresult.params['slope'].value
    sd_lm = np.std(np.log10(y) - (np.log10(a_lm) + b_lm * np.log10(x)), ddof = 1)
    k = 3 # Number of parameters : Here 2 parameters for the fit + error 
    n=len(x)
    l_logn = np.sum(np.log(lognorm10pdf(y,  a_lm * x ** b_lm, sd_lm)))#(stats.lognorm.logpdf(y, s = sd_lr, scale = a_lr * x ** b_lr)) # scale = np.log(mu)!!!
    
    AIC_logn = 2 * k - 2 * l_logn
    AICc_logn = 2 * k - 2 * l_logn + 2 * k * (k + 1) / (n - k - 1)
    return thisresult, a_lm, b_lm, l_logn, AIC_logn, AICc_logn

def fit_unt_pl(area, df):
    mod_pl = PowerLawModel()
    thisdf = df.dropna()
    thispars = mod_pl.guess(thisdf[area].values, x = thisdf['starter'].values)

    thisresult = mod_pl.fit(thisdf[area].values, thispars, x=thisdf['starter'].values)
    print(thisresult.fit_report())

    if area == 'input':
        plotfn.plot_data_vs_fit(thisdf['starter'].values, thisdf[area].values, thisresult.best_fit,'starters','inputs','power-law fit, linear scale, all targets, all inputs')
    else:
        plotfn.plot_data_vs_fit(thisdf['starter'].values, thisdf[area].values, thisresult.best_fit,'starters','inputs','power-law fit, linear scale, all targets, inputs from '+ area)

    a_nlr = thisresult.params['amplitude'].value
    b_nlr = thisresult.params['exponent'].value
    sd_nlr = np.std(thisdf[area].values - a_nlr * (thisdf['starter'].values)**b_nlr)
    k = 3 # Number of parameters : Here 2 parameters for the fit + error 
    n = len(thisdf['starter'].values)

    l_norm = np.sum(stats.norm.logpdf(thisdf[area].values, loc = a_nlr*(thisdf['starter'].values)**b_nlr, scale = sd_nlr)) # scale = np.log(mu)!!!
    AIC_norm = 2 * k - 2 * l_norm
    AICc_norm = 2 * k - 2 * l_norm + 2 * k * (k + 1) / (n - k - 1)

    return thisresult, a_nlr, b_nlr,l_norm, AIC_norm, AICc_norm

def analyze_residuals_noplot(df, saveqq, savepl, listareas, isnorm, targetc):
    # isnorm = 1 for natural scale (H0 : log-normal distribution of residuals)
    # isnorm = 0 for log scale (H0 : normal distribution of residuals)
    # targetc = 0 for whole brain
    # targetc = 1 for V1
    # targetc = 2 for PM

    result = []
    result_params = []
    
    if isnorm == 0:
        strfit = 'linear'
        strscale = 'log'
    else:
        strfit = 'powerlaw'
        strscale = 'linear'

    if targetc == 0:
        strtarget = 'alltargets'
    elif targetc == 1:
        strtarget = 'V1'
    elif targetc == 2:
        strtarget = 'PM'

    for area in listareas:
        if isnorm == 0:
            thisresult, thisa, thisb, thisl, thisAIC, thisAICc = fit_log_lm_noplot(area, df)
        else:
            thisresult, thisa, thisb, thisl, thisAIC , thisAICc = fit_unt_pl_noplot(area, df)

        result.append((area, thisresult))
        result_params.append((area, thisa, thisb, thisl,thisAIC, thisAICc))

    print(result[0][1])

    df_results = pd.DataFrame(result_params, columns=['area', 'intercept', 'slope', 'LL', 'AIC', 'AICc'])
    
    return df_results

def analyze_residuals(df, saveqq, savepl, listareas, isnorm, targetc):
    # isnorm = 1 for natural scale (H0 : log-normal distribution of residuals)
    # isnorm = 0 for log scale (H0 : normal distribution of residuals)
    # targetc = 0 for whole brain
    # targetc = 1 for V1
    # targetc = 2 for PM

    result = []
    result_params = []
    
    if isnorm == 0:
        strfit = 'linear'
        strscale = 'log'
    else:
        strfit = 'powerlaw'
        strscale = 'linear'

    if targetc == 0:
        strtarget = 'alltargets'
    elif targetc == 1:
        strtarget = 'V1'
    elif targetc == 2:
        strtarget = 'PM'

    for area in listareas:
        if isnorm == 0:
            thisresult, thisa, thisb, thisl, thisAIC, thisAICc = fit_log_lm(area, df)
        else:
            thisresult, thisa, thisb, thisl, thisAIC , thisAICc = fit_unt_pl(area, df)

        result.append((area, thisresult))
        result_params.append((area, thisa, thisb, thisl,thisAIC, thisAICc))

    print(result[0][1])
    
    # plot qq plots and residuals vs fitted
    
    for i, area in enumerate(listareas):

        if area == 'input':
            plotfn.qq_plot_res(result[i][1].residual, strtarget + ', all inputs, ' + strscale + ' scale, ' + strfit + ' fit' )
            plt.savefig(os.path.join(saveqq, 'qqplot_' + strscale + 'scale_' + strtarget + '_' + strfit + '_' + area + '.png'))
            plt.savefig(os.path.join(saveqq, 'qqplot_' + strscale + 'scale_' + strtarget + '_' + strfit + '_' + area + '.eps'), format = 'eps')

            plotfn.stdres_vs_fitted(result[i][1].best_fit, result[i][1].residual, strtarget +' , all inputs, ' + strscale + ' scale, ' + strfit + ' fit' )
            plt.savefig(os.path.join(savepl, 'res_vs_fitted_' + strscale + 'scale_' + strtarget + '_' + strfit + '_' + area + '.png'))
            plt.savefig(os.path.join(savepl, 'res_vs_fitted_' + strscale + 'scale_' + strtarget + '_' + strfit + '_' + area + '.eps'), format = 'eps')
        else:
            plotfn.qq_plot_res(result[i][1].residual, strtarget + ', inputs from ' + area + ', ' + strscale + ' scale, ' + strfit + ' fit' )
            plt.savefig(os.path.join(saveqq, 'qqplot_' + strscale + 'scale_' + strtarget + '_' + strfit + '_' + area + '.png'))
            plt.savefig(os.path.join(saveqq, 'qqplot_' + strscale + 'scale_' + strtarget + '_' + strfit + '_' + area + '.eps'), format = 'eps')

            plotfn.stdres_vs_fitted(result[i][1].best_fit, result[i][1].residual, strtarget + ', inputs from ' + area + ', ' + strscale + ' scale, ' + strfit + ' fit')
            plt.savefig(os.path.join(savepl, 'res_vs_fitted_' + strscale + 'scale_' + strtarget + '_' + strfit + '_' + area + '.png'))
            plt.savefig(os.path.join(savepl, 'res_vs_fitted_' + strscale + 'scale_' + strtarget + '_' + strfit + '_' + area + '.eps'), format = 'eps')



    df_results = pd.DataFrame(result_params, columns=['area', 'intercept', 'slope', 'LL', 'AIC', 'AICc'])
    
    return df_results