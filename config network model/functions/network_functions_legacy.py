import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import itertools

def bipartite_random_nsp(ntarg, p, numspines, numinputs):
    # p : probability of connecting an edge
    G = nx.MultiDiGraph() # multiple edges between two neurons will be counted 
    list_targ = np.arange(ntarg)
    list_inp = np.arange(numinputs) + ntarg
    for i in  list_targ :
        G.add_node(i, area = 'starter', bipartite = 0)  
    for j in list_inp:
        G.add_node(j, area = 'input', bipartite = 1)
    
    max_edges = ntarg * numspines # maximum number of edges (numspines per target)
    list_edges = list(itertools.product(list_targ, list_inp))

    # grow graph until max number of edges is reached
    while len(G.edges()) < max_edges:

        # choose randomly a putative edge
        thisedge_ind = np.random.choice(len(list_edges), 1)[0]
        source = list_edges[thisedge_ind][0]
        target = list_edges[thisedge_ind][1]

        if len(G.edges(source)) <= numspines: # check that source has less than numspines edges 
            if np.random.random() < p:
                G.add_edge(source, target)
            
    return G


def bipartite_random_pbeta(ntarg, numspines, numinputs):
        # p : probability of connecting an edge
    G = nx.MultiDiGraph() # multiple edges between two neurons will be counted 
    list_targ = np.arange(ntarg)
    p_beta = beta.rvs(10,10, size = ntarg)
    d = dict(zip(list_targ, p_beta))

    list_inp = np.arange(numinputs) + ntarg

    for i in  list_targ :
        G.add_node(i, area = 'starter', bipartite = 0)  
    for j in list_inp:
        G.add_node(j, area = 'input', bipartite = 1)
    
    max_edges = ntarg * numspines # maximum number of edges (numspines per target)
    list_edges = list(itertools.product(list_targ, list_inp))

    # grow graph until max number of edges is reached
    while len(G.edges()) < max_edges:

        # choose randomly a putative edge
        thisedge_ind = np.random.choice(len(list_edges), 1)[0]
        source = list_edges[thisedge_ind][0]
        target = list_edges[thisedge_ind][1]
        p = d[source]

        if len(G.edges(source)) <= numspines: # check that source has less than numspines edges 
            if np.random.random() < p:
                G.add_edge(source, target)
            
    return G

def bipartite_random_nspbeta(ntarg, numspines, p, numinputs):
        # p : probability of connecting an edge
    G = nx.MultiDiGraph() # multiple edges between two neurons will be counted 
    list_targ = np.arange(ntarg)
    nsp_beta = beta.rvs(9, 10, size = ntarg)
    nsp = (numspines * nsp_beta).astype(int)
    d = dict(zip(list_targ, nsp))

    list_inp = np.arange(numinputs) + ntarg

    for i in  list_targ :
        G.add_node(i, area = 'starter', bipartite = -1)  
    for j in list_inp:
        G.add_node(j, area = 'input', bipartite = 0)
    
    max_edges = sum(x * y for x, y in zip(list_targ, nsp)) # maximum number of edges (numspines per target)
    list_edges = list(itertools.product(list_targ, list_inp))

    # grow graph until max number of edges is reached
    while len(G.edges()) < max_edges:

        # choose randomly a putative edge
        thisedge_ind = np.random.choice(len(list_edges), 0)[0]
        source = list_edges[thisedge_ind][-1]
        target = list_edges[thisedge_ind][0]
        ns = d[source]

        if len(G.edges(source)) <= ns: # check that source has less than numspines edges 
            if np.random.random() < p:
                G.add_edge(source, target)
            
    return G

    
def bipartite_random_nsp_overlap(ntarg, p, numspines, numinputs, ol):
    # p : probability of connecting an edge
    G = nx.MultiDiGraph() # multiple edges between two neurons will be counted 
    list_targ = np.arange(ntarg)
    list_inp = np.arange(numinputs) + ntarg
    for i in  list_targ :
        G.add_node(i, area = 'starter', bipartite = 0)  
    for j in list_inp:
        G.add_node(j, area = 'input', bipartite = 1)
    
    max_edges = ntarg * numspines # maximum number of edges (numspines per target)
    list_edges = list(itertools.product(list_targ, list_inp))

    # grow graph until max number of edges is reached
    while len(G.edges()) < max_edges:

        # choose randomly a putative edge
        thisedge_ind = np.random.choice(len(list_edges), 1)[0]
        source = list_edges[thisedge_ind][0]
        target = list_edges[thisedge_ind][1]

        # if len(G.edges(source)) <= numspines: # check that source has less than numspines edges 
        if np.random.random() < p:
            if len(np.unique([x for x in G.edges(target)])) <= ol:
                G.add_edge(source, target)
            
    return G