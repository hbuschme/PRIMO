#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:06:58 2016

@author: jpoeppel
"""

import numpy as np
import copy
from primo.nodes import DiscreteNode

class Factor(object):
    """
        Class representing a factor in an inference network.
        Factors can be multiplied together and variables can be summed out
        (marginalised) of factors.
    """
    
    def __init__(self):
        self.potentials = np.array([])
         # Use a dictionary that stores the variable names as 
        # keys and their corresponding dimensions as values
        self.variables = {}
        # Use a dictionary for the value lists with the variables as keys.
        self.values = {}
        self.variableOrder = []
        
        
    @classmethod
    def get_trivial(cls, potential=1.0):
        """
            Helper function to create a trivial factor with a given potential.
            A trivial factor does not represent any variables anymore.
            
            Parameter
            ---------
            potential : Float, optional
                The potential this factor should be initialized with.
                
            Returns
            -------
                Factor
                The resulting trivial factor.
        """
        res = cls()
        res.potentials = potential
        return res
        
    @classmethod
    def from_node(cls, node):
        """
            Helper function that allows to create a factor from a random node.
            Currently only DiscreteNodes are supported.
            
            Paramter
            --------
            node : DiscreteNode
                The node which is used to create the factor.
                
            Returns
            -------
                Factor
                The resulting factor representing the potential at the given node.        
        """
        if not isinstance(node, DiscreteNode):
            raise TypeError("Only DiscreteNodes are currently supported.")
        res = cls()
        res.variables[node.name] = len(res.variables)
        res.variableOrder.append(node.name)
        res.values[node.name] = copy.copy(node.values)
        res.potentials = np.copy(node.cpd)
        for p in node.parentOrder:
            res.variables[p] = len(res.variables)
            res.variableOrder.append(p)
            res.values[p] = copy.copy(node.parents[p].values)
        return res
        
    @classmethod
    def as_evidence(cls, variable, values, evidence):
        """
            Creates an "evidence factor" which is used to introduce hart and
            soft evidence into the inference algorithms.
            
            Parameters
            ---------
            variable: String
                The name of the variable that this evidence is given for.
            values: [String,]
                List of possible values/outcomes of that variable
            evidence: String or np.array
                The actual evidence that was observed. If a string is given,
                hard evidence is assumed for that given outcome. Otherwise,
                the evidence is set according to the given array. The probabilities
                in the array need to be in the same order as given in the values
                
            Returns
            -------
                Factor
                A factor for the given evidence variable containing potentials
                according to the strength of the evidence.
        """
        res = cls()
        res.variableOrder.append(variable)
        res.variables[variable] = 0
        res.values[variable] = copy.copy(values)
        if not hasattr(evidence, '__iter__'):
            if not evidence in values:
                raise ValueError("Evidence {} is not one of the possible values ({}) for this variable.".format(evidence, values))
            res.potentials = np.zeros(len(values))
            res.potentials[values.index(evidence)] = 1.0
        else:
            if len(evidence) != len(values):
                raise ValueError("The number of evidence strength ({}) does not correspont to the number of values ({})".format(len(evidence),len(values)))
            res.potentials = np.copy(evidence)
        return res
        
    def __mul__(self, other):
        """
            Allows multiplication of two factors. Will not modify the two
            factors in any way but return a new factor instead.
            
            Currently this is NOT commutative!!
            This means that the self-factor is used as base to create the resulting
            factor and any new variables in the other factor are added afterwards!
            They will however be added in the same order as they have been in
            the other factor.
            
            Also value order is currently NOT checked between the two factors,
            i.e. if one factor orders the values of some variable as "True", "False"
            and another sorts them "False", "True" the results will be wrong!
            
            Paramter
            -------
            other : Factor
                The factor that is multiplied to this factor
                
            Returns
            -------
                Factor
                The product of this factor and the other factor.
        """
        #Create copies to avoid modification of original factors
        f1 = copy.deepcopy(self)
        f2 = copy.deepcopy(other)
        
        # Shortcuts for trivial factors
        if len(f1.variables) == 0:
            f2.potentials = f1.potentials * f2.potentials
            return f2
            
        if len(f2.variables) == 0:
            f1.potentials = f1.potentials * f2.potentials
            return f1
        
        #Extend factor 1 by all the variables it is missing
        for v in other.variableOrder:
            if not v in f1.variables:
                ax = f1.potentials.ndim
                f1.variables[v] = ax
                f1.values[v] = copy.copy(other.values[v])
                f1.potentials = np.expand_dims(f1.potentials, axis = ax)
                f1.potentials = np.repeat(f1.potentials, len(other.values[v]), axis = ax)
                f1.variableOrder.append(v)
                
#        print "f1.variableOrder: ", f1.variableOrder
#        print "f1 potentials: ", f1.potentials
#        print "f2 variableOrder before: ", f2.variableOrder
        # Ensure factor2 has the same size as factor1
        for v in f1.variableOrder:
            if not v in f2.variables:
                f2.potentials = np.expand_dims(f2.potentials, f1.variables[v])
                f2.potentials = np.repeat(f2.potentials, len(f1.values[v]), axis = f1.variables[v])
#                print "inserting {} at {}".format(v, f1.variables[v])
                #Fix variable order!
                f2.variableOrder.insert(f1.variables[v], v)
                for idx, v in enumerate(f2.variableOrder):
                    f2.variables[v] = idx
            else:
                #Roll axes around so that they align. Cannot use f2.variables[v] since
                # new axes might have been inserted in the meantime
#                print "rollaxis for {} from {} to {}".format(v, f2.variableOrder.index(v), f1.variables[v])
#                print "potentials before: ", f2.potentials
                f2.potentials = np.rollaxis(f2.potentials, f2.variableOrder.index(v), f1.variables[v])
                
                f2.variableOrder.remove(v)
                f2.variableOrder.insert(f1.variables[v], v)
                for idx, v in enumerate(f2.variableOrder):
                    f2.variables[v] = idx

#        print "f2.variableOrder: ", f2.variableOrder
#        print "f2 potentials: ", f2.potentials
#        print "f1 potentials: ", f1.potentials                
                
        # Pointwise multiplication which results in a factor where all instantiations
        # are compatible to the instantiations of factor1 and factor2
        # See Definition 6.3 in "Modeling and Reasoning with Bayesian Networks" - Adnan Darwiche Chapter 6
        f1.potentials = f1.potentials * f2.potentials

        return f1
    
    
    def marginalize(self, variables):
        """
            Function to create a new factor with the given variable summed out
            of this factor.
            
            Parameter
            ---------
            variables: String or RandomNode or [String,] or [RandomNode,]
                Either a single variable or a list of variables that are to
                be removed.
                Variables can either be addressed by their names or by the
                nodes themselves.
                
            Returns
            ------
                Factor
                A new factor where the given variables has been summed out.
        """
        
        if not hasattr(variables, '__iter__'):
            variables = [variables]
            
        res = copy.deepcopy(self)
        for v in variables:
            res.potentials = np.sum(res.potentials, axis=res.variableOrder.index(v))
            del res.values[v]
            res.variableOrder.remove(v)
            
        res.variables = {}
        #Fix variable index dictionary:
        for idx, v in enumerate(res.variableOrder):
            res.variables[v] = idx
            
        return res
        
    def get_potential(self, variables={}):
        """
            Function that allows to query for specifiy potentials within this 
            factor. IMPORTANT: This potential is not necessary a probability.
            
            Even if probabilities are represented, these potentials could 
            be conditional probabilities or joint probabilities of different
            variables.
            
            If variables is not given, will simply return the full potential table.
            
            Parameter
            ---------
            variables: dict, optional.
                Dictionary containing the desired variable names as keys and
                a list of instantiations of interest as values.
                
            Returns
            -------
                np.array
                The currently stored potential for the given variables and their values.
        """
        
        
        index = []        
        for v in self.variableOrder:
            if v in variables:
                try:
                    index.append([self.values[v].index(value) for value in variables[v]])
                except ValueError:
                    raise ValueError("There is no potential for variable {} with values {} in this factor.".format(v, variables[v]))
            else:
                index.append(range(len(self.values[v])))
                    
        if len(variables) == 0:
            return self.potentials
                  
        return np.squeeze(np.copy(self.potentials[np.ix_(*index)]))
        
    def normalize(self):
        """
            Normalizes the included potential so that they add up to 1. Should
            mainly be used internally when computing posterior marginals!
        """
        
        self.potentials /= np.sum(self.potentials)