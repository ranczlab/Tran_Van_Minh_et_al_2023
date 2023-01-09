import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
import os


def line(x, a, b):
    return a * x + b


def median_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.median(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-2) # n - number of parameters
    #r = stats.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=stats.sem(a))
    return m, np.mean(a)-h, np.mean(a)+h


# Bootstrap raw residuals
def bootstrap_res(x, y, numB, islog,  **kwargs):
    results, pcov = curve_fit(line, x.values, y.values)
    slope = []
    intercept = []
    resid = []
    
    slope.append(results[0])
    intercept.append(results[1])
    resid.append(y.values - results[1] - x*results[0])
    ypred = intercept[0] + x.values*slope[0] # predicted values from data fit
    
    thisPars = results
    plt.figure()
    for i in range(numB-1):
        resample = np.random.choice(resid[0], size = len(resid[0])) # bootstrap initial residuals
        thisy = resample + ypred

        thisPars, pcov = curve_fit(line, x.values, thisy)
        resid.append(y - thisPars[1] - x.values*thisPars[0])
        plt.plot(x.values, thisPars[1] + x.values*thisPars[0], 'k-', linewidth = 2, alpha = 4/numB) # weight the transparency of the line by number of permutations
        slope.append(thisPars[0])
        intercept.append(thisPars[1])

    plt.plot(x.values, ypred, 'c-',label = 'ls fit') 
    plt.plot(x.values, y.values, 'ro', label = 'data')
    if islog == 1:
        plt.xlabel('Log(starters)', fontsize = 15)
        plt.ylabel('Log(inputs)' , fontsize = 15)
    else:
        plt.xlabel('starters', fontsize = 15)
        plt.ylabel('inputs' , fontsize = 15)        
    if 'savepath' in kwargs:
        savepath = kwargs['savepath']
        title = kwargs['title']
        plt.savefig(os.path.join(savepath, title+'.png'))

    return intercept, slope, resid

def bootstrap_res_and_save2(listareas, df, islog, title, savepath, savepath_plots, numB):
    boot_int = []
    boot_int_median = []
    boot_slope = []
    boot_slope_median = []
    CI95_int = []
    CI95_slope = []
    list_computed_areas = []
    for area in listareas:
        try:
            dftemp = pd.concat([df[area], df['starter']],axis=1)
            dftemp = dftemp.dropna()

            thisint, thisslope, res = bootstrap_res(dftemp['starter'], dftemp[area], numB, islog = islog, title = 'resbootstrap_'+title+'_'+area )
            [thisintmedian, thisintCImin, thisintCImax] = median_confidence_interval(thisint, confidence=0.95)
            [thisslopemedian, thisslopeCImin, thisslopeCImax] = median_confidence_interval(thisslope, confidence=0.95)
            boot_int.append((area,thisint))
            boot_slope.append((area,thisslope))
            CI95_int.append((area,np.mean(thisint), thisintmedian, pd.DataFrame(thisint).quantile(0.025, axis=0).values[0],pd.DataFrame(thisint).quantile(0.975, axis=0).values[0]))
            CI95_slope.append((area, np.mean(thisslope), thisslopemedian, pd.DataFrame(thisslope).quantile(0.025, axis=0).values[0],pd.DataFrame(thisslope).quantile(0.975, axis=0).values[0]))
            list_computed_areas.append(area)
        except:
            pass
    df_CI95_int = pd.DataFrame(CI95_int, columns = ['area','mean int','median int','95%quant-low','95%quant-high'])
    df_CI95_slope = pd.DataFrame(CI95_slope, columns = ['area','mean slope','median slope','95%quant-low','95%quant-high'])
    df_boot_int = pd.DataFrame()

    for i,area in enumerate(list_computed_areas):
        df_boot_int[area]=boot_int[i][1]
    df_boot_slope = pd.DataFrame()
    for i,area in enumerate(list_computed_areas):
        df_boot_slope[area]=boot_slope[i][1]           
    df_CI95_int.to_csv(os.path.join(savepath,'df_CI95_int_'+title+'.csv'))
    df_CI95_slope.to_csv(os.path.join(savepath,'df_CI95_slope_'+title+'.csv'))
    df_boot_int.to_csv(os.path.join(savepath,'dfboot_int_'+title+'.csv'))
    df_boot_slope.to_csv(os.path.join(savepath,'dfboot_slope_'+title+'.csv'))
    return df_boot_int, df_boot_slope, df_CI95_int, df_CI95_slope

def plot_scatter_95bs(listareas, data, islog, data_bs_int, data_bs_slope, title, savepath):
    xrange = np.linspace(min(data['starter']),max(data['starter']),100)

    for area in listareas:
        try:
            plt.figure()
            # calculate the bootstraps 
            this_fits = pd.DataFrame()
            all_fits = []
            for thisbs in range(data_bs_int.shape[0]):
                thisfit = []
                for i, thisx in enumerate(xrange):
                    thisfit.append(data_bs_int[area][thisbs] + thisx * data_bs_slope[area][thisbs])
                all_fits.append(thisfit)
            fits_pre = pd.DataFrame.from_records(all_fits) #
            this_fit_mean = fits_pre.mean(axis = 0)
            # plot sample data

            plt.scatter(data['starter'],data[area])
            plt.xlim(min(data['starter']), max(data['starter'])) 
            if islog == 1:
                plt.ylim(-3,12)            
            # plot line of best fit
            plt.plot(xrange, this_fit_mean, '-', color='C1')
            plt.title(area)
            # plot with percentiles from bootstrap fits
            plt.plot(xrange, fits_pre.quantile(0.025, axis=0), '--', color='C1')
            plt.plot(xrange, fits_pre.quantile(0.975, axis=0), '--', color='C1')
            plt.savefig(os.path.join(savepath, title+'_'+area+'.png'))
            
        except:
            pass