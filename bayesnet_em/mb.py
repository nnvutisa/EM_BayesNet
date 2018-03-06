def search_hidden(data):
    """Returns the column index of the hidden node if only one column of NaN.
    Only works if data is numeric.
    
    Parameters
    ----------
    data : An ndarray (n_sample, n_nodes)
    
    Returns
    -------
    ind_h : the index of the hidden node column
    """
    is_col_nan = np.all(np.isnan(data), axis=0)
    ind = np.where(is_col_nan)
    if np.size(ind)==1:
        ind_h = ind[0][0]
    else:
        raise ValueError('Data contains more than one hidden nodes or no hidden node')
    return ind_h


class MarkovBlanket():
    """
    An object for storing info on nodes within the markov blanket of the hidden node
    
    Parameters
    ----------
    ind_h : int
        index of the hidden node within the model
        
    Attributes
    ----------
    hidden : int
        index of the hidden node
        
    parents : list of int
        a list of indices of the parent nodes 
        
    children : list of int
        a list of indices of the children nodes
        
    coparents : list of int
        a list of indices of the coparent nodes
        
    prob_table : dict
        a dict of probabilities table of nodes within the Markov blanket
    
    """
    
    def __init__(self, ind_h):
        self.hidden = ind_h
        self.parents = []
        self.children = []
        self.coparents = []
        self.prob_table = {}
        
    def populate(self, model):
        """populate the parents, children, and coparents nodes
        """
        state_indices = {state.name : i for i, state in enumerate(model.states)}
        
        edges_list = [(parent.name, child.name) for parent, child in model.edges]
        edges_list = [(state_indices[parent],state_indices[child]) 
                  for parent, child in edges_list]
        
        self.children = list(set([child for parent, child in edges_list if parent==self.hidden]))
        self.parents = list(set([parent for parent, child in edges_list if child==self.hidden]))
        self.coparents = list(set([parent for parent, child in edges_list if child in self.children]))
        try:
            self.coparents.remove(self.hidden)
        except ValueError:
            pass
            
    def calculate_prob(self, model):
        """Create the probability table from nodes
        """
        for ind_state in [self.hidden]+self.children:
            distribution = model.states[ind_state].distribution
            
            if isinstance(distribution, ConditionalProbabilityTable):
                table = distribution.parameters[0]
                self.prob_table[ind_state] = {
                    tuple(row[:-1]) : row[-1] for row in table}
            else:
                self.prob_table[ind_state] = distribution.parameters[0]
                
    def update_prob(self, model, expected_counts, ct):
        """Update the probability table using expected counts
        """
        ind = {x : i for i, x in enumerate([self.hidden] + self.parents + self.children + self.coparents)}
        mb_keys = expected_counts.counts.keys()
        
        for ind_state in [self.hidden] + self.children:
            distribution = model.states[ind_state].distribution
            
            if isinstance(distribution, ConditionalProbabilityTable):
                idxs = distribution.column_idxs
                table = self.prob_table[ind_state] # dict
                
                # calculate the new parameter for this key
                for key in table.keys():
                    num = 0
                    denom = 0
                    
                    # marginal counts
                    for mb_key in mb_keys:
                        # marginal counts of node + parents
                        if tuple([mb_key[ind[x]] for x in idxs]) == key:
                            num += ct.table[mb_key[1:]]*expected_counts.counts[mb_key] 
                            
                        # marginal counts of parents
                        if tuple([mb_key[ind[x]] for x in idxs[:-1]]) == key[:-1]:
                            denom += ct.table[mb_key[1:]]*expected_counts.counts[mb_key]
                            
                    try:
                        prob = num/denom
                    except ZeroDivisionError:
                        prob = 0
                        
                    # update the parameter
                    table[key] = prob
                    
            else: # DiscreteProb
                table = self.prob_table[ind_state] # dict 
                
                # calculate the new parameter for this key
                for key in table.keys():
                    prob = 0
                    for mb_key in mb_keys:
                        if mb_key[ind[ind_state]] == key:
                            prob += ct.table[mb_key[1:]]*expected_counts.counts[mb_key]
                    
                    # update the parameter
                    table[key] = prob
                    

class ExpectedCounts():
    """Calculate the expected counts using the model parameters
    
    Parameters
    ----------
    model : a BayesianNetwork object
    
    mb : a MarkovBlanket object
    
    Attributes
    ----------
    counts : dict
        a dict of expected counts for nodes in the Markov blanket
    """
    
    def __init__(self, model, mb):
        self.counts = {}
        
        self.populate(model, mb)
        
    def populate(self, model, mb):
        #create combinations of keys
        keys_list = [model.states[mb.hidden].distribution.keys()]
        for ind in mb.parents + mb.children + mb.coparents:
            keys_list.append(model.states[ind].distribution.keys())
        
        self.counts = {p:0 for p in itertools.product(*keys_list)}
        
    def update(self, model, mb):
        ind = {x : i for i, x in enumerate([mb.hidden] + mb.parents + mb.children + mb.coparents)}
    
        marginal_prob = {}
    
        # calculate joint probability and marginal probability
        for i, key in enumerate(self.counts.keys()):
            prob = 1
        
            for j, ind_state in enumerate([mb.hidden] + mb.children):
                distribution = model.states[ind_state].distribution
            
                if isinstance(distribution, ConditionalProbabilityTable):
                    idxs = distribution.column_idxs
                    state_key = tuple([key[ind[x]] for x in idxs])
                else:
                    state_key = key[ind[ind_state]]
                
                prob = prob*mb.prob_table[ind_state][state_key]         
                self.counts[key] = prob
            try:
                marginal_prob[key[1:]] += prob
            except KeyError:
                marginal_prob[key[1:]] = prob
                 
        # divide the joint prob by the marginal prob to get the conditional
        for i, key in enumerate(self.counts.keys()):
            try:
                self.counts[key] = self.counts[key]/marginal_prob[key[1:]]
            except ZeroDivisionError:
                self.counts[key] = 0

                
class CountTable():
    """Counting the data"""
    
    def __init__(self, model, mb, items):
        """
        Parameters
        ----------
        model : BayesianNetwork object
        
        mb : MarkovBlanket object
        
        items : ndarray
            columns are data for parents, children, coparents
        
        """
        self.table ={}
        self.ind = {}
        
        self.populate(model, mb, items)
        
    def populate(self, model, mb, items):
        keys_list = []
        for ind in mb.parents + mb.children + mb.coparents:
            keys_list.append(model.states[ind].distribution.keys())
        
        # init
        self.table = {p:0 for p in itertools.product(*keys_list)}
        self.ind = {p:[] for p in itertools.product(*keys_list)}
        
        # count
        for i, row in enumerate(items):
            try:
                self.table[tuple(row)] += 1
                self.ind[tuple(row)].append(i)
            except KeyError:
                print 'Items in row', i, 'does not match the set of keys.'
                raise KeyError
    


        
