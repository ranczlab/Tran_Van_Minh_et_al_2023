import sys
import os

if sys.platform == 'darwin': 
    import multiprocessing 
else:
    import psutil

import numpy as np
import pandas as pd
import itertools #install as more-itertools 
from pathlib import Path
import networkx as nx
from networkx.algorithms import bipartite
import pickle #already part of standard library
from scipy.stats import beta
sys.path.append(os.path.join(str(Path(os.getcwd()).parents[1]), 'functions'))
import network_functions as nxfn
import helper_functions as hfn

if sys.platform == 'darwin': 
    available_cpu_count = multiprocessing.cpu_count()-2 #this is for macOS
else:
    available_cpu_count = len(psutil.Process().cpu_affinity()) #this is for unix cluster, not extensively tested     
os.environ["MKL_NUM_THREADS"] = str(available_cpu_count)

np.random.seed(0)

input_index = int(list(sys.argv)[1]) # read from bash script

res_dir = os.path.join(os.path.join(Path(os.getcwd()).parents[1], 'results'))
if not os.path.exists(res_dir):
    try:
        os.mkdir(res_dir)
    except:
        pass

out_dir = os.path.join(res_dir, os.path.basename(os.getcwd()))
if not os.path.exists(out_dir):
    try:
        os.mkdir(out_dir)
    except:
        pass
    
params = pd.read_excel('params_grid.xlsx')

l = [tuple(x) for x in params.to_numpy()]

thisl = l[input_index]

n_starters = int(thisl[0])
nsp = int(thisl[1])
ol = int(thisl[2])
seq_s = np.random.normal(nsp, nsp*0.1, n_starters).astype(int)#(nsp * beta.rvs(10, 10, size = n_starters)).astype(int)
seq_temp = np.random.normal(ol, ol*0.2, n_starters*1000).astype(int)#(ol * beta.rvs(10, 10, size = n_starters*1000)).astype(int)
seq_i = hfn.make_seq(seq_temp, sum(seq_s))
nrep = 100

G = bipartite.configuration_model(seq_s, seq_i,  create_using = nx.MultiGraph(), seed = 0)
df_g = pd.DataFrame(G.edges(), columns = ['starters', 'inputs'])
df_res = hfn.sample_num_starters(df_g, nrep, n_starters)
df_res['starters'] = n_starters
df_res['numspines'] = nsp
df_res['overlap'] = ol

df_res.to_hdf(os.path.join(out_dir, 'df_config_ntarg_'+ str(n_starters)  + '_nspines_' + str(nsp) + '_overlap_' + str(ol)+ '_nrep_' + str(nrep) + '.h5'), key = 'df_config', mode = 'w')
 
with open(os.path.join(out_dir,'G_config_ntarg_'+ str(n_starters) + '_nspines_' + str(nsp) +'_overlap_' + str(ol)+'.pickle'), 'wb') as f:
    pickle.dump(G, f)