import numpy as np
from pomegranate import *
import itertools

from .mb import *


def em_bayesnet(model, data, ind_h, max_iter = 50, criteria = 0.005):
    """Returns the data array with the hidden node filled in.
    (model is not modified.)
    
    Parameters
    ----------
    model : a BayesianNetwork object
        an already baked BayesianNetwork object with initialized parameters
        
    data : an ndarray
        each column is the data for the node in the same order as the nodes in the model
        the hidden node should be a column of NaNs
        
    ind_h : int
        index of the hidden node
        
    max_iter : int
        maximum number of iterations
        
    criteria : float between 0 and 1
        the change in probability in consecutive iterations, below this value counts as convergence 
        
    Returns
    -------
    data : an ndarray
        the same data arary with the hidden node column filled in
    """
    
    # create the Markov blanket object for the hidden node
    mb = MarkovBlanket(ind_h)
    mb.populate(model)
    mb.calculate_prob(model)
    
    # create the count table from data
    items = data[:, mb.parents + mb.children + mb.coparents]
    ct = CountTable(model, mb, items)
    
    # create expected counts
    expected_counts = ExpectedCounts(model, mb)
    expected_counts.update(model, mb)
    
    # ---- iterate over the E-M steps
    i = 0
    previous_params = np.array(mb.prob_table[mb.hidden].values())
    convergence = False
    
    while (not convergence) and (i < max_iter):
        mb.update_prob(model, expected_counts, ct)
        expected_counts.update(model, mb)
        # print 'Iteration',i,mb.prob_table
        
        # convergence criteria
        hidden_params = np.array(mb.prob_table[mb.hidden].values())
        change = abs(hidden_params - previous_params)
        convergence = max(change) < criteria
        previous_params = np.array(mb.prob_table[mb.hidden].values())
        
        i += 1
        
    if i == max_iter:
        print 'Maximum iterations reached.'
    
    # ---- fill in the hidden node data by sampling the distribution
    labels = {}
    for key, prob in expected_counts.counts.items():
        try:
            labels[key[1:]].append((key[0], prob))
        except:
            labels[key[1:]] = [(key[0], prob)]
            
    for key, counts in ct.table.items():
        label, prob = zip(*labels[key])
        prob = tuple(round(p,5) for p in prob)
        if not all(p == 0 for p in prob):
            samples = np.random.choice(label, size=counts, p=prob)
            data[ct.ind[key], ind_h] = samples
        
    return data
