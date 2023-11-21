import random
import pandas as pd
import numpy as np
import os

def dir_check(thispath):
    if not os.path.exists(thispath):
        os.mkdir(thispath)

def create_subfolders(savepath):
    # create classic subfolder structure (figures, DF, csv) in given folder
    savepath_fig = os.path.join(savepath, 'figures')
    dir_check(savepath_fig)
    savepath_df = os.path.join(savepath, 'DF')
    dir_check(savepath_df)
    savepath_csv = os.path.join(savepath, 'csv')
    dir_check(savepath_csv)
    return savepath_fig, savepath_df, savepath_csv

def num_inputs_from_starters(df, ns, n_starters):
    # from a bipartite graph (edges in df with columns = ['starters', 'inputs'])
    # select randomly ns starters, count the number of unique inputs
    # n_starters : number of starters nodes 
    # INPUTS : - df :           Dataframe built from bipartite graph edges : columns = starters, inputs (list nodes for each)
    #          - ns :           number of starter to sample
    #          - all_starters : list of all starter nodes    
    # OUTPUT : - n_inputs :     number of inputs for the batch of selected starters
    
    starters_resamp = np.random.choice(int(n_starters), int(ns), replace = False)
    
    n_inputs = len(df['inputs'][df['starters'].isin(starters_resamp)].unique()) # count number of unique inputs for starters in starters_resamp
    return n_inputs


def sample_num_starters(df, nrep, n_starters):
    # for each ns in np.arange(total starter number)
    # pick ns starter, count number of inputs : repeat 10 times
    # at each iteration save ns and number of inputs
    # INPUTS : - df :           Dataframe built from bipartite graph edges : columns = starters, inputs (list nodes for each)
    #          - nrep :         number of repetitions (how many time we sample #ns starters)
    #          - n_starters :   number of starter nodes
    # OUTPUT : - df_res :       dataframe with num_starters, num_inputs as columns, for each repetition
    
    listdict = []
    for thisns in np.arange(n_starters)+1:
        for thisrep in range(nrep):
            thisni = num_inputs_from_starters(df, thisns, n_starters)
            thisdict = {'num_starters' : thisns, 'num_inputs' : thisni}
            listdict.append(thisdict)
    
    df_res = pd.DataFrame(listdict)
    
    return df_res

def make_seq(dist, target):
    dist = list(dist)
    seq2 = []
    while sum(seq2) < target:
        if target-sum(seq2)>dist[0]:
            seq2.append(dist[0])
            dist.pop(0)
        else:
            seq2[-1] += 1
    return seq2