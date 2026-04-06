import numpy as np
import torch


class HierarchicalPrior:
    '''
    Class for flexible hierarchical prior constructed from directed acyclic graph.
    '''
    def __init__(self, variables):
        # sort variables
        depths = np.array([var.depth for var in variables])
        self.variables = list(np.array(variables)[np.argsort(depths, kind = 'stable')])

        # collect names of variables 
        self.names = []
        for variable in self.variables:
            self.names += [variable.name + '[' + str(i) + ']' for i in range(variable.dim)]

        # for indexing
        self.index = []
        for variable in self.variables:
            self.index += [variable.name] * variable.dim
        
        # dimension (necessary?)
        self.dim = len(self.index)

        # current size of values.
        self.n_current = None
    
    def collect(self):
        theta = [variable.values() for variable in self.variables]
        return torch.cat(theta, dim = -1)
    
    def assign(self, theta):
        n = len(theta)
        self.n_current = n
        for variable in self.variables:
            variable._values = theta[:,np.array(self.index) == variable.name]
    
    def lp(self, values = None):
        if values is None:
            values = self.collect()
        else:
            self.assign(values)
        lp = torch.zeros((self.n_current, 1), device = values.device, dtype = values.dtype)
        for variable in self.variables:
            lp += variable.lp(values[:, np.array(self.index) == variable.name])
        return lp[:, 0]
    
    def sample(self, n = 1):
        for variable in self.variables:
            variable.sample(n)
        self.n_current = n
        return self.collect()

    def to_dict(self, values = None):
        if values is None:
            values = self.collect()
        dic = {}
        for variable in self.variables:
            dic[variable.name] = values[:,np.array(self.index) == variable.name]
        return dic
    
    def check_support(self, values):
        support_checks = []
        for variable in self.variables:
            check_result = variable.check_support(values[:,np.array(self.index) == variable.name])
            support_checks.append(check_result)
        support_checks = torch.cat(support_checks, dim = -1)
        support_checks = torch.prod(support_checks, dim = 1).bool()
        return support_checks