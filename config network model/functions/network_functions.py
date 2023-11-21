import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import itertools
from functools import reduce
from networkx.utils import py_random_state

def grow_bipartite_random(ntarg, p, numspines):
    # p : probability of adding a new input edge
    G = nx.MultiDiGraph() # multiple edges between two neurons will be counted 
    G.add_nodes_from(np.arange(ntarg), bipartite = 0)
    
    max_edges = ntarg * numspines # maximum number of edges (numspines per target)
    
    # grow graph until max number of edges is reached
    while len(G.edges()) < max_edges:
        ninputs = len(G) - ntarg # we are not fixing the number of inputs

        # choose randomly a source
        source = np.random.choice(np.arange(ntarg))

        if len(G.edges(source)) <= numspines: # check that source has less than numspines edges 
            if np.random.random() < p:
                target = len(G) # add a new input node and connect
                G.add_node(target, bipartite = 1)
                G.add_edge(source, target)
            else:
                if ninputs>0: # if there are any inputs, pick one of the existing one and connect
                    target = np.random.choice(np.arange(ninputs) + ntarg)
                    G.add_edge(source, target)
            
    return G



def grow_bipartite_custom_pa(ntarg, p, numspines):
    # p : probability of adding a new input edge
    G = nx.MultiDiGraph()
    G.add_nodes_from(np.arange(ntarg), bipartite = 0)
    
    max_edges = ntarg * numspines # maximum number of edges (numspines per target)
    
    # grow graph until max number of edges is reached
    while len(G.edges()) < max_edges:
        ninputs = len(G) - ntarg

        # choose randomly a source
        source = np.random.choice(np.arange(ntarg)) 

        if len(G.edges(source)) <= numspines:  # check that source has less than numspines edges 
            if np.random.random() < p:
                target = len(G)
                G.add_node(target, bipartite = 1)
                G.add_edge(source, target)
            else:
                P = np.asarray([G.degree(b) for b in range(ntarg, len(G))])
                P  = P/float(P.sum())
                if ninputs>0:
                    target = np.random.choice(np.arange(ninputs) + ntarg, p = P)
                    G.add_edge(source, target)
    return G


def grow_bipartite_custom_inv_pa(ntarg, p, numspines):
    # p : probability of adding a new input edge
    G = nx.MultiDiGraph()
    G.add_nodes_from(np.arange(ntarg), bipartite = 0)
    
    max_edges = ntarg * numspines # maximum number of edges (numspines per target)
    
    # grow graph until max number of edges is reached
    while len(G.edges()) < max_edges:
        ninputs = len(G) - ntarg

        # choose randomly a source
        source = np.random.choice(np.arange(ntarg)) 

        if len(G.edges(source)) <= numspines:  # check that source has less than numspines edges 
            if np.random.random() < p:
                target = len(G)
                G.add_node(target, bipartite = 1)
                G.add_edge(source, target)
            else:
                P = np.asarray([1/G.degree(b) for b in range(ntarg, len(G))])
                P  = P/float(P.sum())
                if ninputs>0:
                    target = np.random.choice(np.arange(ninputs) + ntarg, p = P)
                    G.add_edge(source, target)
    return G
    
def grow_bipartite_custom_pa_exp(ntarg, p, numspines, gamma):
    # p : probability of adding a new input edge
    G = nx.MultiDiGraph()
    G.add_nodes_from(np.arange(ntarg), bipartite = 0)
    
    max_edges = ntarg * numspines # maximum number of edges (numspines per target)
    
    # grow graph until max number of edges is reached
    while len(G.edges()) < max_edges:
        ninputs = len(G) - ntarg

        # choose randomly a source
        source = np.random.choice(np.arange(ntarg)) 

        if len(G.edges(source)) <= numspines:  # check that source has less than numspines edges 
            if np.random.random() < p:
                target = len(G)
                G.add_node(target, bipartite = 1)
                G.add_edge(source, target)
            else:
                P = np.asarray([G.degree(b) for b in range(ntarg, len(G))])
                P = P**gamma
                P = P/float(P.sum())

                if ninputs>0:
                    target = np.random.choice(np.arange(ninputs) + ntarg, p = P)
                    G.add_edge(source, target)
    return G




def bipartite_random_nsp2(ntarg, p, numspines, numinputs):
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
            G.add_edge(source, target)
            
    return G





@py_random_state(4)
def pa_withol_graph(aseq, p, ol, create_using=None, seed=None):
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    if p > 1:
        raise nx.NetworkXError("probability {p} > 1")

    naseq = len(aseq)

    G = _add_nodes_with_bipartite_label(G, naseq, 0)
    vv = [[v] * aseq[v] for v in range(0, naseq)]
    while vv:
        while vv[0]:
            source = vv[0][0]
            vv[0].remove(source)
            if seed.random() < p or len(G) == naseq:
                target = len(G)
                G.add_node(target, bipartite=1)
                G.add_edge(source, target)
            else:
                bb = [([b] * G.degree(b) if G.degree(b)<ol else [b]) for b in range(naseq, len(G)) ]

                bbstubs = reduce(lambda x, y: x + y, bb)
                # choose preferentially a bottom node.
                target = seed.choice(bbstubs)
                G.add_node(target, bipartite=1)
                G.add_edge(source, target)

        vv.remove(vv[0])
    G.name = "bipartite_preferential_attachment_ol_model"
    return G

@py_random_state(4)
def rand_withol_graph(aseq, p, ol, create_using=None, seed=None):
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")

    if p > 1:
        raise nx.NetworkXError("probability {p} > 1")
    naseq = len(aseq)

    G = _add_nodes_with_bipartite_label(G, naseq, 0)
    vv = [[v] * aseq[v] for v in range(0, naseq)]
    while vv:
        while vv[0]:
            source = vv[0][0]
            vv[0].remove(source)
            if seed.random() < p or len(G) == naseq:
                target = len(G)
                G.add_node(target, bipartite=1)
                G.add_edge(source, target)
            else:

                bb = [[b] for b in range(naseq, len(G)) if G.degree(b)<ol]
                if bb:
                # flatten the list of lists into a list.
                    bbstubs = reduce(lambda x, y: x + y, bb)
                    # choose preferentially a bottom node.
                    target = seed.choice(bbstubs)
                    G.add_node(target, bipartite=1)
                    G.add_edge(source, target)
                else:
                    target = len(G)
                    G.add_node(target, bipartite=1)
                    G.add_edge(source, target)
        vv.remove(vv[0])
    G.name = "bipartite_rand_ol_model"
    return G


def _add_nodes_with_bipartite_label(G, lena, lenb):
    G.add_nodes_from(range(0, lena + lenb))
    b = dict(zip(range(0, lena), [0] * lena))
    b.update(dict(zip(range(lena, lena + lenb), [1] * lenb)))
    nx.set_node_attributes(G, b, "bipartite")
    return G

