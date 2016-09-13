#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of PRIMO -- Probabilistic Inference Modules.
# Copyright (C) 2013-2015 Social Cognitive Systems Group, 
#                         Faculty of Technology, Bielefeld University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Lesser GNU General Public License as 
# published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public 
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

import itertools

import numpy as np
import copy
import numpy

import networkx as nx

import primo.nodes
import primo.networks
import primo.inference.factor
import primo.util as util
import primo.densities

class UtilityTable(object):
    '''
    self.variables -- list of the parent nodes
    self.table -- utility table which contains the utility
    '''

    def __init__(self):
        super(UtilityTable, self).__init__()
        self.table = np.array(0)
        self.variables = []

    def add_variable(self, variable):
        self.variables.append(variable)

        ax = self.table.ndim
        self.table=np.expand_dims(self.table,ax)
        self.table=np.repeat(self.table,len(variable.value_range),axis = ax)

    def get_ut_index(self, node_value_pairs):
        nodes, values = zip(*node_value_pairs)
        index = []
        for node in self.variables:
            try:
                index_in_values_list = nodes.index(node)
                value = values[index_in_values_list]
                index.append(node.value_range.index(value))
            except ValueError:
                #print("NoValue")
                index.append((-1))
        return tuple(index)

    def set_utility_table(self, table, nodes):
        if(nodes):
            if not set(nodes) == set(self.variables):
                raise Exception("The list which should define the ordering of the variables does not match"
                    " the variables that this cpt depends on (plus the node itself)")
            if not self.table.ndim == table.ndim:
                #print self.table.ndim
                #print table.ndim
                raise Exception("The provided probability table does not have the right number of dimensions")
            for d,node in enumerate(nodes):
                if len(node.value_range) != table.shape[d]:
                    raise Exception("The size of the provided probability table does not match the number of possible values of the node "+node.name+" in dimension "+str(d))

        self.table = table
        #self.variables = nodes

    def set_utility(self, value, node_value_pairs):
        index = self.get_ut_index(node_value_pairs)
        self.table[index]=value

    def get_utility_table(self):
        return self.table

    def get_variables(self):
        return self.variables

    def get_utility(self, node_value_pairs):
        index = self.get_ut_index(node_value_pairs)
        
        if(index[0] == -1):
            #print("index: " + str(index))
            return 0
        return self.table[index]
    
    def summation(self, inputFactor):
        #init a new probability tabel
        factor1 = primo.densities.ProbabilityTable()

        #all variables from both factors are needed
        factor1.variables = copy.copy(self.variables)

        for v in (inputFactor.variables):
            if not v in factor1.variables:
                factor1.variables.append(v)

            #the table from the first factor is copied
            factor1.table = copy.copy(self.table)

        #and extended by the dimensions for the left variables
        for curIdx in range(factor1.table.ndim, len(factor1.variables)):
            ax = factor1.table.ndim
            factor1.table=numpy.expand_dims(factor1.table,ax)
            factor1.table=numpy.repeat(factor1.table,len(factor1.variables[curIdx].value_range),axis = ax)

        #copy factor 2 and it's variables ...
        factor2 = primo.densities.ProbabilityTable()
        factor2.variables = copy.copy(inputFactor.variables)
        factor2.table = copy.copy(inputFactor.table)

        #extend the dimensions of factors 2 to the dimensions of factor 1
        for v in factor1.variables:
            if not v in factor2.variables:
                factor2.variables.append(v)

        for curIdx in range(factor2.table.ndim, len(factor2.variables)):
            ax = factor2.table.ndim
            factor2.table=numpy.expand_dims(factor2.table,ax)
            factor2.table=numpy.repeat(factor2.table,len(factor2.variables[curIdx].value_range),axis = ax)

        #sort the variables to the same order
        for endDim,variable in enumerate(factor1.variables):
            startDim = factor2.variables.index(variable);
            if not startDim == endDim:
                factor2.table = numpy.rollaxis(factor2.table, startDim, endDim)
                factor2.variables.insert(endDim,factor2.variables.pop(startDim))

        #pointwise addition
        #print "shape1: " +str(factor1.table.shape) +" shape2: " + str(factor2.table.shape)
        if factor1.table.shape != factor2.table.shape:
            raise Exception("Multiplication: The probability tables have the wrong dimensions for unification!")

        factor1.table = factor1.table + factor2.table;

        return factor1
    
    def __str__(self):
        return str(self.table)


class DecisionTable(object):
    '''
    self.variables -- list of the parent nodes
    self.table -- table which contains the decision rule in cpd form
    '''

    def __init__(self):
        super(DecisionTable, self).__init__()
        self.table = np.array(0.0)
        self.variables = []

    def add_variable(self, variable):
        self.variables.append(variable)

        ax = self.table.ndim
        self.table=np.expand_dims(self.table,ax)
        self.table=np.repeat(self.table,len(variable.value_range),axis = ax)

    def get_dt_index(self, node_value_pairs):
        nodes, values = zip(*node_value_pairs)
        index = []
        for node in self.variables:
            try:
                index_in_values_list = nodes.index(node)
                value = values[index_in_values_list]
                index.append(node.value_range.index(value))
            except ValueError:
                #print("NoValue")
                index.append((-1))
        return tuple(index)

    def set_decision_table(self, table, nodes):
        if(nodes):
            if not set(nodes) == set(self.variables):
                raise Exception("The list which should define the ordering of the variables does not match"
                    " the variables that this cpt depends on (plus the node itself)")
            if not self.table.ndim == table.ndim:
                #print self.table.ndim
                #print table.ndim
                raise Exception("The provided probability table does not have the right number of dimensions")
            for d,node in enumerate(nodes):
                if len(node.value_range) != table.shape[d]:
                    raise Exception("The size of the provided probability table does not match the number of possible values of the node "+node.name+" in dimension "+str(d))

        self.table = table

    def set_rule(self, value, node_value_pairs):
        index = self.get_dt_index(node_value_pairs)
        self.table[index]=value

    def get_decision_table(self):
        return self.table

    def get_variables(self):
        return self.variables

    def get_value(self, node_value_pairs):
        index = self.get_dt_index(node_value_pairs)
        
        if(index[0] == -1):
            #print("index: " + str(index))
            return 0
        return self.table[index]

    def __str__(self):
        return str(self.table)
    
    
class Strategy(object):
    '''
    A Complete assignment of decision rules to every Decisionnode in a DN is a 
    strategy (sigma) (from PGM).
    Strategy = [(D1,d_i),.....,(Dn,d_i)]
    '''
    def __init__(self):
        self.strategy = []
        self.utility = 0
    
    def init_simple(self, partialOrdering):
        
        for i in range(0,len(partialOrdering)-1,2):
            self.strategy.append([partialOrdering[i], partialOrdering[i].get_value_range()[0]])
        
        #self.strategy[1][1] = partialOrdering[0].get_value_range()[1]
        print str(self.strategy)
        
    def add_decision(self, decNode, decValue):
        self.strategy.append([decNode, decValue])
    
    def get_strategylist(self):
        return self.strategy
    
    def __repr__(self):
        return "Strategy: " + str(self.strategy) + " => ut: " + str(self.utility)
    
    
class MakeDecision(object):
    """
    Calculates a Decision on a given Bayesian Decision Network
    """

    def __init__(self, bdn = None):
        """
        Constructor

        Keyword arguments:

        bdn -- Bayesian Decision Network (default None)
        """
        super(MakeDecision, self).__init__()
        self.bdn = bdn
        self.strategy = Strategy()
        self.ddn = primo.networks.DynamicDecisionNetwork()
        
    def set_bdn(self, bdn):
        """
        Sets the Bayesian Decision Network

        Keyword arguments:

        bdn -- Bayesian Decision Network
        """
        self.bdn = bdn
        
    def set_ddn(self,ddn):
        if(isinstance(ddn,primo.networks.DynamicDecisionNetwork)):
            self.ddn = ddn
        else:
            raise Exception("network: "+ str(ddn)+ " is not a DDN!")
        
    def get_ddn(self):
        return self.ddn
    
    def get_bdn(self):
        """
        Getter for the Bayesian Decision Network
        """
        return self.bdn

    def compute_optimal_strategy(self):
        
        partialOrdering = self.bdn.get_partialOrdering()
        
        for i in range(0,len(partialOrdering)-1,2):
            best_decision = self.max_sum2(partialOrdering[i])
            partialOrdering[i].set_state(best_decision[1])
            self.strategy.add_decision(best_decision[0], best_decision[1])

        return self.strategy
    
    def setEvidence(self, evidence):
        '''
        reduces Nodes CPTs by given evidence
        @param evidence: List of(Node, Value) Pairs
        '''
        
        for item in evidence:
            tmp = self.bdn.get_node(item[0].name)
            #print(evidence[0])
            ev = tmp.set_evidence(evidence[0])
            reduced = tmp.get_cpd_reduced(evidence)
            tmp.set_cpd(reduced)
    
    def reduce_Evidence(self,evidence):
        for item in evidence:
            tmp = self.bdn.get_node(item[0].name)
            #print(item)
            cpd_ev = tmp.set_evidence(evidence[0])
        return cpd_ev

    #def calc_marginal(self):
        
####### Exact Inference ########################################################
    def max_sum(self, decisionNode):
        """
        DEPRICATED! FUNCTION does not work properly!
        
        Implementation of the max sum Algorithm to get the best Decision (according to the MEU principle).
        maximize over decisions and summing over RandomNodes.
        This function sets the state of provided DecisionNode, so later decisions can't affect that Node

        Keyword arguments:

        decisionNode -- Decision Node on which the decision should be made
        """
        if self.bdn == None:
            raise Exception("Bayesian Decision Network was not set!")

        partialOrder = self.bdn.get_partialOrdering()
        utility_nodes = self.bdn.get_all_utility_nodes()


        if not partialOrder:
            raise Exception("No partial Order was set!")

        if decisionNode not in partialOrder:
            raise Exception("Decision Node is not in the partial Order!")

        if not self.bdn.is_valid():
            raise Exception("The Bayesian Decision Network is not valid!")

        #Check if the Decision Nodes that are ordered before the provided Decision Node have a state
        for node in partialOrder:
            if isinstance(node, primo.nodes.DecisionNode):
                if not decisionNode.name == node.name:
                    if node.get_state() is None:
                        raise Exception("Decision Nodes that are ordered before the provided Decision Node must have a state!")
                else:
                    break

        '''Run through the partialOrder in reverse. Get the last two Nodes, reduce the Random Nodes with the Decision Node
        parent and with the decisions already made. Then multiply the cpts of the Random Nodes. Multiply the probability values
        with the sum of the utility values and calculate the best decision (which has the MEU).
        '''
        randomNodes = self.bdn.get_all_nodes()
        future_best_decisions = []
        future_max_value = []
        finish = False
            
        for i in range(len(partialOrder)-1, -1, -2):
            max_utility = []
            #for every decision value of the decision node
            for decValue in partialOrder[i-1].get_value_range():
                #print("decisionvalue ="+ decValue)
                #if the decision already have a value then abort. The decision has already been made.
                if not partialOrder[i-1].get_state() == None:
                    finish = True
                    break

                cpts = []
                #reduce Random Nodes with a Decision value
                for rNode in randomNodes:
                    if partialOrder[i-1] in self.bdn.get_parents(rNode):
                        cpts.append(rNode.get_cpd_reduced([(partialOrder[i-1], decValue)]))
                    else:
                        cpts.append(rNode.get_cpd())

                #reduce the cpts with the future_best_decisions
                for j in range(0,len(cpts)):
                    for node,value in future_best_decisions:
                        if node in cpts[j].get_variables():
                            cpts[j] = cpts[j].reduction([(node,value)])

                #multiply the cpts
                jointCPT = cpts[0]
                for j in range(1,len(cpts)):
                    jointCPT = jointCPT.multiplication(cpts[j])

                #print("jCPTvalues: \n"+str(jointCPT.get_variables()))
                #print("jCPT: \n"+ str(jointCPT))
                #calculate Utility
                table = jointCPT.get_table()
                value_range_list = []
                #get every variable instantiation
                for var in jointCPT.get_variables():
                    tupleList=[]
                    for value in var.get_value_range():
                        tupleList.append((var,value))
                    if tupleList:
                        value_range_list.append(tupleList)

                #get all possible assignments
                permutationList = []
                
                if len(value_range_list) >= 2:
                    permutationList = list(itertools.product(*value_range_list))
                else:
                    permutationList = value_range_list
                
                
                #save the results of each probability value and the according sum of utilities
                result = []
                if len(permutationList) > 1:
                    for perm in permutationList:
                        index = jointCPT.get_cpt_index(perm)
                        value = table[index]
                        result.append(value * self.calculate_utility(self.bdn, perm, (partialOrder[i-1],decValue), future_best_decisions))
                        #print(str(perm),": ",value," * ",str(self.calculate_utility(perm, (partialOrder[i-1],decValue), future_best_decisions)),
                            #" = ", value*self.calculate_utility(perm, (partialOrder[i-1],decValue), future_best_decisions))
                else:
                    for perm in permutationList[0]:
                        index = jointCPT.get_cpt_index([perm])
                        value = table[index]
                        result.append(value * self.calculate_utility(self.bdn, [perm], (partialOrder[i-1],decValue), future_best_decisions))
                        #print(value," * ",str(self.calculate_utility([perm], (partialOrder[i-1],decValue), future_best_decisions)))
                #print(str(permutationList))
               
                        
                #end result for this decision
                max_utility.append((decValue,sum(result)))
                #print(str(max_utility))
            #nothing more to do since the decision has already been made
            if finish:
                break

            zippedList = zip(*max_utility)
            val = max(zippedList[1])
            ind = zippedList[1].index(val)

            print str(max_utility)

            #Best Decision
            best_decision = zippedList[0][ind]
            future_best_decisions.append((partialOrder[i-1],best_decision))
            future_max_value.append((partialOrder[i-1].name,best_decision,val))
            #print(str(future_max_value))
        #the last one is the decision that we want to know about
        return future_max_value[len(future_max_value)-1]
    
    
    def calculate_utility(self, bdn, assignment, currentDecision, list_of_best_decision):
        """
        Sums up the utility values

        Keyword arguments:

        assignment -- the assignment of the variables
        currentDecision -- the current decision that we want to calculate
        list_of_best_decision -- list of the decisions that are lying in the future
        """
#        print "assign: " +str(assignment)
#        print "curDec: " +str(currentDecision)
#        print "lisbest: " +str(list_of_best_decision)
        utilityList=[]
        zippedAssignment = zip(*assignment)
        zippedDecisions = zip(*list_of_best_decision)
        utility_nodes = bdn.get_all_utility_nodes()

        for uNode in utility_nodes:
            tempList = []
            parent_nodes = bdn.get_parents(uNode)
            #print("utNode: "+ uNode.name+" parents: "+str(parent_nodes))
            for node in parent_nodes:
                #print("parentNode: "+node.name)
                if node in zippedAssignment[0]:
                    index = zippedAssignment[0].index(node)
                    tempList.append((node,zippedAssignment[1][index]))
                    #print("tmpList: "+str(tempList))
                elif zippedDecisions:
                    if node in zippedDecisions[0]:
                        index = zippedDecisions[0].index(node)
                        tempList.append((node,zippedDecisions[1][index]))
                        #print("tmpList2: "+str(tempList))
                    else:
                        tempList.append(currentDecision)
                        #print("tmpList3: "+str(tempList))
                else:
                    tempList.append(currentDecision)
                    #print("tmpList4: "+str(tempList))
                    
                
                #print("utNode: "+ uNode.name+" tmp: "+str(tempList))
            utilityList.append(uNode.get_utility(tempList))
        return sum(utilityList)
    
    
    def buildAllPerms(self, parents):
        value_range_list = []
        for parent in parents:
            tuples = []
            for val in parent.get_value_range():
                tuples.append((parent,val))
            if tuples:
                value_range_list.append(tuples)
        
        permutationList = []

        if len(value_range_list) >= 2:
            permutationList = list(itertools.product(*value_range_list))
        else:
            permutationList = list(itertools.product(*value_range_list))
            
        return permutationList
    
    
    def max_sum2(self, decisionNode, dn=None):
        """Implementation of the max sum Algorithm to get the best Decision (according to the MEU principle).
        maximize over decisions and summing over RandomNodes.
        This function sets the state of provided DecisionNode, so later decisions can't affect that Node

        Keyword arguments:

        decisionNode -- Decision Node on which the decision should be made
        """
        if dn is None:
            if self.bdn == None:
                raise Exception("Bayesian Decision Network was not set!")
            else:
                partialOrder = self.bdn.get_partialOrdering()
                utility_nodes = self.bdn.get_all_utility_nodes()
                bdn = self.bdn
        else:
            partialOrder = dn.get_partialOrdering()
            utility_nodes = dn.get_all_utility_nodes()
            bdn = dn
        
        if not partialOrder:
            raise Exception("No partial Order was set!")

        if decisionNode not in partialOrder:
            raise Exception("Decision Node is not in the partial Order! node: "+ decisionNode.name+" partialorder: "+ str(partialOrder))

        if not bdn.is_valid():
            raise Exception("The Bayesian Decision Network is not valid!")

        #Check if the Decision Nodes that are ordered before the provided Decision Node have a state
        for node in partialOrder:
            if isinstance(node, primo.nodes.DecisionNode):
                if not decisionNode.name == node.name:
                    if node.get_state() is None:
                        raise Exception("Decision Nodes that are ordered before the provided Decision Node must have a state!")
                else:
                    break

        '''Run through the partialOrder in reverse. Get the last two Nodes, reduce the Random Nodes with the Decision Node
        parent and with the decisions already made. Then multiply the cpts of the Random Nodes. Multiply the probability values
        with the sum of the utility values and calculate the best decision (which has the MEU).
        '''
           
        global finish
        finish = False
        
        randomNodes = bdn.get_all_nodes()
        utNodes = bdn.get_all_utility_nodes()
        
        future_best_decisions = []
        future_max_value = []
        
        margList = []
        for utNode in utNodes:
            for n in bdn.get_parents(utNode):
                if(isinstance(n, primo.nodes.DiscreteNode)):
                    margList.append(n) 

        #run in reverse through partialorder    
        for i in range(len(partialOrder)-1, -1, -2):
            
            #if the decision already have a value then abort. The decision has already been made.
            if partialOrder[i-1].get_state() is not None:
                finish = True
                break
                
            print("\n \tcomputing ut for: "+partialOrder[i-1].name)
            
            max_utility = []
            if(not partialOrder[i-1].get_parents()):
                #without Informationlink
                max_utility= max_utility +self.maxsumLoop(bdn,randomNodes,partialOrder[i-1],future_best_decisions,margList)
            else:
                #for every combination of values of the Parent Nodes of the Decision (Informationlink)
                #for parent in partialOrder[i-1].get_parents():
                    #for parentVal in parent.get_value_range():
                        #max_utility= max_utility + self.maxsumLoop(bdn,randomNodes,partialOrder[i-1],future_best_decisions,margList,parent,parentVal)
                parentPerms = self.buildAllPerms(partialOrder[i-1].get_parents())
                for perm in parentPerms:
                        max_utility = max_utility + self.maxsumLoop(bdn,randomNodes,partialOrder[i-1],future_best_decisions,margList,perm)
            if finish:
                break
                
            zippedList = zip(*max_utility)
            print(str(max_utility))
            #print(str(zippedList))
            val = max(zippedList[1])
            ind = zippedList[1].index(val)

            #Best Decision
            best_decision = zippedList[0][ind]
            future_best_decisions.append((partialOrder[i-1],best_decision))
            #print("future_dec: "+str(future_best_decisions))
            future_max_value.append((partialOrder[i-1],best_decision,val))
            print("futurmax: "+str(future_max_value))
                    
        #the last one is the decision that we want to know about
        return future_max_value[len(future_max_value)-1]
    
    
    def maxsumLoop(self,bdn,randomNodes,partialOrder,future_best_decisions,margList,parentPerm=None,parent=None,parentVal=None):
        
        
        max_utility = []

        #for every decision value of the decision node
        for decValue in partialOrder.get_value_range():

            #print("decision: "+partialOrder.name+" = "+ decValue)
            if parent: print("parent:   "+parent.name+" = "+ parentVal)
            
            cpts = []
            #build jCPT                                
            #reduce Random Nodes with a Decision value
            for rNode in randomNodes:
                if partialOrder in bdn.get_parents(rNode):
                    cpts.append(rNode.get_cpd_reduced([(partialOrder, decValue)]))
                else:
                    cpts.append(rNode.get_cpd())

            #reduce with Parents of Decision
            if parentPerm:
                for j in range(0,len(cpts)):
                    for tuple in parentPerm:
                        if tuple[0] in cpts[j].get_variables():
                            cpts[j] = cpts[j].reduction([tuple])

            #reduce the cpts with the future_best_decisions
            for j in range(0,len(cpts)):
                for node,value in future_best_decisions:
                    if node in cpts[j].get_variables():
                        cpts[j] = cpts[j].reduction([(node,value)])

            #multiply the cpts
            jointCPT = cpts[0]
            
            for j in range(1,len(cpts)):
                jointCPT = jointCPT.multiplication(cpts[j])
                
#            print("jCPT: \n"+ str(jointCPT))
#            print jointCPT.variables
            #print("marglist: " + str(margList))
            for v in jointCPT.get_variables():
                if v not in margList:
                    jointCPT = jointCPT.marginalization(v)

            #print("jCPT: \n"+ str(jointCPT)) 
            jointCPT = jointCPT.normalize_as_jpt()
#            print("jCPTvalues: \n"+str(jointCPT.get_variables()))
#            print("jCPT: \n"+ str(jointCPT))
#            print jointCPT.variables

            #calculate Utility
            table = jointCPT.get_table()
            value_range_list = []
            #get every variable instantiation
            for var in jointCPT.get_variables():
                tupleList=[]
                for value in var.get_value_range():
                    tupleList.append((var,value))
                if tupleList:
                    value_range_list.append(tupleList)
            
            #get all possible assignments
            permutationList = []

            if len(value_range_list) >= 2:
                permutationList = list(itertools.product(*value_range_list))
            else:
                permutationList = value_range_list


            #save the results of each probability value and the according sum of utilities
            result = []

            if len(permutationList) > 1:
                for perm in permutationList:
                    index = jointCPT.get_cpt_index(perm)
                    value = table[index]
                    result.append(value * self.calculate_utility(bdn, perm, (partialOrder,decValue), future_best_decisions))
                    #print(str(perm),": ",value," * ",str(self.calculate_utility(perm, (partialOrder[i-1],decValue), future_best_decisions)),
                        #" = ", value*self.calculate_utility(perm, (partialOrder[i-1],decValue), future_best_decisions))
            else:
                for perm in permutationList[0]:
                    index = jointCPT.get_cpt_index([perm])
                    value = table[index]
                    #print("perm: "+ str([perm]))
                    result.append(value * self.calculate_utility(bdn, [perm], (partialOrder,decValue), future_best_decisions))
                    #print(value," * ",str(self.calculate_utility([perm], (partialOrder[i-1],decValue), future_best_decisions)))
            #print(" -> "+ str(sum(result))) #"perms: "+str(permutationList)+

            #end result for this decision
            max_utility.append((decValue,sum(result)))
            #print(str(max_utility))
            #print("\n")
            #nothing more to do since the decision has already been made
            if finish:
                break

        return max_utility
    
################################ VE 4 ID #######################################

    def arg_max_mue_D(self, dn, decision, ev = None):
        
        #build the joint utilityfactor of the net an marginalize everything out but Familiy(decision)
        ef = primo.inference.factor.EasiestFactorElimination(None)
        evidence = []
        if ev:
            evidence.append(ev)
            
        if len(dn.graph.predecessors(decision)) > 0:
            pa_D = dn.graph.predecessors(decision) 
            family = pa_D + [decision]
        
            #Todo loop over parents of D and set parent states as evidence
            for state in pa_D[0].get_value_range():
                fac = ef.calculate_PosteriorMarginal_DN(dn.get_all_nodes(), dn.get_all_utility_nodes(), family ,evidence + [(pa_D[0], state)])
                print fac  
                print fac.variables
                print ""
        else:
            family = [decision]
            fac = ef.calculate_PosteriorMarginal_DN(dn.get_all_nodes(), dn.get_all_utility_nodes(), family ,evidence)
            print fac  
            print fac.variables
            
        #calc the rules for the decisionnode
#        self.calc_rule(fac, decision)
#        print decision.get_decision_table()
        
        
        
        

    def calc_rule(self, factor, decision):
        "bullshit, maybe something can be reused, clean up later"
        variables = copy.copy(factor.variables)
        index = variables.index(decision)
        variables.pop(index)
        
        value_range_list = []
        #get every variable instantiation
        for var in variables:
            tupleList=[]
            for value in var.get_value_range():
                tupleList.append((var,value))
            if tupleList:
                value_range_list.append(tupleList)

        #get all possible assignments
        permutationList = []
        if len(value_range_list) >= 2:
            permutationList = list(itertools.product(*value_range_list))
        else:
            permutationList = value_range_list
        #print permutationList

        table = factor.get_table()
        if len(permutationList) == 1:
            item = permutationList[0]
            
            for tup in item:
                tmp = []
                for dec in decision.get_value_range():
                    assign = []
                    assign.append((decision, dec))
                    assign.append(tup)
                    index = factor.get_cpt_index(assign)
                    tmp.append((table[index], assign))
                tupl = max(tmp, key=lambda item:item[0])
                decision.set_rule(1.0, tupl[1])    
        else:
            for item in permutationList:
                tmp = []
                for dec in decision.get_value_range():
                    assign = []
                    for tup in item:
                        assign.append(tup)
                    assign.append((decision, dec))
                    index = factor.get_cpt_index(assign)
                    tmp.append((table[index], assign))
                print tmp

                tupl = max(tmp, key=lambda item:item[0])
                decision.set_rule(1.0, tupl[1])
            

################################ DDN solving ###################################   
    def update_initial_Bel(self, dn, decision):
        state = {}
        state_vars = [dn.get_node_from_DN(n.name) for (n, _) in self.ddn._twoTDN.get_initial_nodes()]
        #print ("state_vars: ",state_vars)
        #print "decision: " + str(decision)
        for var in state_vars:
            if(isinstance(var,primo.nodes.DiscreteNode)):
                state[var] = self.update_node(dn, [var], dn.get_node_from_DN(decision[0]), decision[1]).table
            elif(isinstance(var,primo.nodes.DecisionNode)):
                state[var] = str(decision[1])
        return state
    
    def update_Bel(self, twoTDN, evidence=None):
        '''updates beliefs in a DN, changes CPDs'''
        
        #get decision made in t-1
        partialordering = twoTDN.get_partialOrdering()
        decision = partialordering[0]
        
        state ={}
        state_vars = [nt for (_, nt) in twoTDN.get_initial_nodes()]
        
#        print ("decision: " + str(decision))
#        print ("state_vars: ",state_vars)
        for var in state_vars:
            if(isinstance(var,primo.nodes.DiscreteNode)):
                state[var] = self.update_node(twoTDN, [var], twoTDN.get_node_from_DN(decision.name), decision.get_state(), evidence).table
                #print state[var]
        return state
    
    def calculate_decision_and_utility(self, dn, dec_node, ut_nodes, best_decisions, evidence=None):
        '''returns decision and ut value for decision'''
        #print("\t computing ut")
        max_utility = []
        future_max_value = []
        margList = []
        
        for ut_node in ut_nodes:
            for n in dn.get_parents(ut_node):
                if(isinstance(n, primo.nodes.DiscreteNode)):
                    margList.append(n)
        
        #print "Marglist: " +str(margList)
                    
        if(not dec_node.get_parents()):
            #without Informationlink
            if(evidence==None):
                max_utility = max_utility + self.calc_one_decision(dn,dec_node, margList, best_decisions)
            else:
                max_utility = max_utility + self.calc_one_decision(dn,dec_node, margList, best_decisions, evidence)
        else:
            #for every combination of values of the Parent Nodes of the Decision (Informationlink)
            for parent in dec_node.get_parents():
                for parentVal in parent.get_value_range():
                    #print "parent :" + str(parent.name) + " = " +str(parentVal)
                    if(evidence==None):
                        max_utility= max_utility + self.calc_one_decision(dn,dec_node, margList, best_decisions, [(parent,parentVal)])
                    else:
                        max_utility= max_utility + self.calc_one_decision(dn,dec_node, margList, best_decisions, evidence + [(parent,parentVal)])
        
        zippedList = zip(*max_utility)
        print(str(max_utility))
        val = max(zippedList[1])
        ind = zippedList[1].index(val)

        #Best Decision
        best_decision = zippedList[0][ind]
        #print("future_dec: "+str(future_best_decisions))
        future_max_value.append((dec_node,best_decision,val))
        print("futurmax: "+str(future_max_value))
        #the last one is the decision that we want to know about
        return future_max_value[len(future_max_value)-1]
     
        
    def calc_one_decision(self, bdn, dec_node, margList, best_decisions, evidence=None):    
        
        future_best_decisions = best_decisions
        max_ut = []
        
        
        for decValue in dec_node.get_value_range():
            #print "decision: " + str(dec_node.name) + " = " + str(decValue)
            
            if(evidence ==None):
                jCPT = self.update_node(bdn, margList, dec_node, decValue)
            else:
                jCPT = self.update_node(bdn, margList, dec_node, decValue, evidence)
            #print str(jCPT)
            permutationList = self.get_all_perms(jCPT)
            table = jCPT.get_table()
            result = []
            
            if len(permutationList) > 1:
                for perm in permutationList:
                    index = jCPT.get_cpt_index(perm)
                    value = table[index]
                    result.append(value * self.calculate_utility(bdn, perm, (dec_node,decValue), future_best_decisions))
            else:
                for perm in permutationList[0]:
                    index = jCPT.get_cpt_index([perm])
                    value = table[index]
                    result.append(value * self.calculate_utility(bdn, [perm], (dec_node,decValue), future_best_decisions))
            print(" -> "+ str(sum(result))) #"perms: "+str(permutationList)+

            #end result for this decision
            max_ut.append((decValue,sum(result)))
        
        return max_ut
    
    def get_all_perms(self, jCPT):
        value_range_list = []
        #get every variable instantiation
        for var in jCPT.get_variables():
            tupleList=[]
            for value in var.get_value_range():
                tupleList.append((var,value))
            if tupleList:
                value_range_list.append(tupleList)

        #get all possible assignments
        permutationList = []

        if len(value_range_list) >= 2:
            permutationList = list(itertools.product(*value_range_list))
        else:
            permutationList = value_range_list
        
        return permutationList
    
    #TODO: rename function
    def update_node(self, bdn, margList, decNode, decValue, evidence=None):
        cpts = []
        #reduce Random Nodes with a Decision value and/or evidence
        #print evidence
        for rNode in bdn.get_all_nodes():
            if decNode in bdn.get_parents(rNode):
                cpts.append(rNode.get_cpd_reduced([(decNode, decValue)]))
            else:
                cpts.append(rNode.get_cpd())
                    
        if(evidence != None):
            for j in range(0,len(cpts)):
                    for node,value in evidence:
                        if node in cpts[j].get_variables():
                            #cpts[j] = cpts[j].reduction([(node,value)])
                            cpts[j] = cpts[j].set_evidence((node,value))
        
        #multiply the cpts
        jointCPT = cpts[0]
        for j in range(1,len(cpts)):
            jointCPT = jointCPT.multiplication(cpts[j])

        for v in jointCPT.get_variables():
            if v not in margList:
                jointCPT = jointCPT.marginalization(v)

        jointCPT = jointCPT.normalize_as_jpt()
        #print("jCPTvalues: \n"+str(jointCPT.get_variables()))
        #print("jCPT: "+ str(jointCPT))
        return jointCPT        
   
   
    def compute_timestep_DDN(self, evidence):
        '''computes the resulting ut and decision for one timestep in twoTDN. Sets
        decision value to decision node and saves result in resultlist of ddn'''
        
        #check if evidence is for currenttimeslice and not for an old one
        nodes0 = [n for (n, _) in self.ddn._twoTDN.get_initial_nodes()]
        for item in evidence:
            if item[0] in nodes0:
                raise Exception("Evidence can not be set to past timeslices!")
        
        #compute predicted state = Bel(X_t|E_t-1) an set P(X_t) = Bel
        #add evidence for timestep -> Bel(X_t|E_t)
        state = self.update_Bel(self.ddn._twoTDN,evidence)
        #print "state:" + str(state)
        self.ddn.create_next_timeslice_twoT(state)
        
        #calc decision and utility
        #nodes must be in the right order, d0, d1, ...,dn !
        d_t0 = self.ddn._twoTDN.get_all_decision_nodes()[0] #maybe better to work with partialordering
        u_t0 = self.ddn._twoTDN.get_all_utility_nodes()[0]
        res = self.calculate_decision_and_utility(self.ddn._twoTDN, d_t0, [u_t0], [])
        res_c = copy.deepcopy(res)
        
        d_t0.set_state(res[1])
        self.ddn._twoTDN.ut_current = res[2]
        return res_c
        
    def compute_initial_DDN(self, evidence):
        '''computes the reulting ut and decision for the initial net DN0'''
        self.ddn._DN0.set_evidence(evidence)
        d0 = self.ddn._DN0.get_all_decision_nodes()[0]
        u = self.ddn._DN0.get_all_utility_nodes()[0]
        res = self.calculate_decision_and_utility(self.ddn._DN0, d0, [u], [])
        state = self.update_initial_Bel(self.ddn._DN0, copy.deepcopy(res))
        
        #print "init_state:" + str(state)
        self.ddn.create_next_timeslice_twoT(state,True)
        self.ddn._twoTDN.ut_current = res[2]
        #print("\t DN0 computed")
        return res

    def compute_DDN(self):
        #TODO: this method is example usage move it to examples
        result = []
        #compute results for DN0(initial net)
        evidence = [(self.ddn._DN0.get_node_from_DN("f0"), "growing")]
        result.append(self.compute_initial_DDN(evidence))
        
        
        #compute t=2
        evidence =  [(self.ddn._twoTDN.get_node_from_DN("f_t"), "growing")] 
        result.append(self.compute_timestep_DDN(evidence))
        
        #compute t=3
        #evidence = [(self.ddn._twoTDN.get_node_from_DN("f_t"), "growing")]
        result.append(self.compute_timestep_DDN(evidence))
        
    
        return result
        
        
        
######## Iterated Optimization #################################################

    def iteratedApprox(self, dn):
        ''' Iterated Approximation Algorithm to find best strategy for ID with 
        multiple Decisions. 
        See Probabilistic Grahical Models page 1108 ff
        current version works only for IDs where one DecNode has one corresponding UtNode
        '''
        
        sigma_0 = Strategy()
        sigma_0.init_simple(dn.get_partialOrdering())
        
        ut_nodes = dn.get_all_utility_nodes()
        self.approx_loop(dn, sigma_0,ut_nodes)
        #print sigma_0
        ut = sigma_0.utility
        sigma_0.utility = 0
        continue_optimizing = True
        
        while continue_optimizing:
            self.approx_loop(dn, sigma_0, ut_nodes)
            if ut < sigma_0.utility:
                ut = sigma_0.utility
                sigma_0.utility = 0
            else: 
                continue_optimizing = False
                
        return sigma_0  
    
    
    def iterated_Optimization(self, dn):
        '''
        Iterated Optimization for Influencdiagrams with acyclic relevance graphs.
        
        from Probabilistic Graphical Models p1116 Algorithm 23.3
        
        returns sigma_0 (global optimal Strategy)
        '''
        
        sigma_0 = Strategy()
        sigma_0.init_simple(dn.get_partialOrdering())
        
        
        self.build_relevance_graph(dn)
#        print dn.relevance_graph.nodes()
        print "relevancegraph: " + str(nx.topological_sort(dn.relevance_graph)) 
        #print ""
        #print sigma_0
        
        if nx.is_directed_acyclic_graph(dn.relevance_graph):
            
            best_decisions = []
            for dec in nx.topological_sort(dn.relevance_graph):
            #all decisions but the active one are fixed to their decisionrule in strategy
                evidence = []
                oldVal = "bla"
                for ev in sigma_0.get_strategylist():
                    if ev[0] != dec:
                        evidence.append((ev[0], ev[1]))
                    else:
                        oldVal = ev[1]
                print "evidence: " +str(evidence)
                
                res = self.calculate_decision_and_utility( dn, dec, dn.get_all_utility_nodes(), best_decisions, evidence)
                print res
                print ""
                #set decision
                best_decisions.append((res[0], res[1]))
                #dn.get_node_from_DN(res[0]).set_state(res[1])
                #update the strategy with the new value
                index = sigma_0.get_strategylist().index([dec, oldVal])
                sigma_0.get_strategylist()[index][1] = res[1]
                sigma_0.utility = sigma_0.utility + res[2]

        
        else:
            print "cyclic relevance graph!"
        
        return sigma_0
        
        
    def build_relevance_graph(self, dn):
        
        partialOrdering = dn.get_partialOrdering()
        length = len(partialOrdering)-1
        if(length < 3):
            raise Exception("Can not build a relevance Graph for a Network with less than 2 DecisionNodes!")
        
        dn.init_relevance_graph()
        
        po = dn.get_partialOrdering()
        #add edges between decisionnodes, need to be explicit for active trail test
        for i in range(0, len(po)-1,2):
            for j in range(i+2, len(po)-1, 2):
                dn.graph.add_edge(po[i], po[j])
        
        #add edges to relevance graph
        for i in range(0,length,2):
            if (i+2 <= length):
                if util.s_reachable(partialOrdering[i], partialOrdering[i+2], dn):
                    dn.relevance_graph.add_edge(partialOrdering[i], partialOrdering[i+2])
                    #print str(partialOrdering[i].name) + " -> " + str(partialOrdering[i+2].name)
                if util.s_reachable(partialOrdering[i+2], partialOrdering[i], dn):
                    dn.relevance_graph.add_edge(partialOrdering[i+2], partialOrdering[i])
                    #print str(partialOrdering[i].name) + " <- " + str(partialOrdering[i+2].name)

        #clean up explicit recalledges
        for i in range(0, len(po)-1,2):
            for j in range(i+2, len(po)-1, 2):
                dn.graph.remove_edge(po[i], po[j])

       
    def approx_loop(self, dn, sigma_0, ut_nodes): 
        
        for item in sigma_0.get_strategylist():
            #all decisions but the active one are fixed to their decisionrule in strategy
            evidence = []
            for ev in sigma_0.get_strategylist():
                if ev != item:
                    evidence.append((ev[0], ev[1]))
            #print evidence
            
            res = self.calculate_decision_and_utility( dn, item[0], ut_nodes, evidence)
            dn.get_node_from_DN(res[0]).set_state(res[1])
            item[1] = res[1]
            sigma_0.utility = sigma_0.utility + res[2]
        
        