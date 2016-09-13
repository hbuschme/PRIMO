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

import random
import primo.nodes

def weighted_random(weights):
    '''
    Implements roulette-wheel-sampling.
    @param weights: A List of float-values.
    @returns: Index of the selected entity
    '''
    counter = random.random() * sum(weights)
    for i, w in enumerate(weights):
        counter -= w
        if counter <= 0:
            return i
        
def s_reachable(D1, D2, net):
    '''
    D1 sreachable from D2
    '''
    #print "\n"+str(D1.name)+" sreachable from "+str(D2.name)
    D_dach = primo.nodes.DecisionNode("D_dach", ["x","y"])
    #add dummy decisionnode
    net.graph.add_node(D_dach)
    net.graph.add_edge(D_dach, D1)

    familiy =  net.graph.predecessors(D2) + [D2] 
    
    r = nodes_reachable_from(D_dach, familiy , net)
    
    
    #print "R: " + str(r)
    #print "U_d: " +str(net.get_utNodes_relevant_to_decNode(D2))
    
    
    for item in net.get_utNodes_relevant_to_decNode(D2):
        if item in r:
            return True
    

        
def nodes_reachable_from(X, Z, net):
    '''
    Algorithm for finding nodes reachable from X given Z via active trails Probabilistic
    Grahical Models p.75
    
    X Source Variable
    Z List of Observations (just List with Nodes)
    returns the set of all nodes reachable from X via active trails
    '''
    
    #Phase I
    L = Z
    A = []
    #print "Z: " +str(Z)
    #print "L: "+str(L)
    while L:
        y = L.pop()
        if not y in A:
            L = list(set(L).union(net.graph.predecessors(y))) 
        A = list(set(A).union([y]))
    
               
    #print "A: "+str(A)
    
    #Phase II
    L = [(X, "up")] #(Node,direction) to be visited
    V = [] #(Node,direction) marked as visited
    R = []
    
    while L:
        #print "L: " +str(L)
        y = L.pop()
        if not y in V:
            if not y[0] in Z:
                R = list(set(R).union([y[0]]))  #y is reachable
            V = list(set(V).union([y])) 
            if y[1] == "up" and not y[0] in Z:
                for z in net.graph.predecessors(y[0]):
                    L = list(set().union(L, [(z,"up")]))
                for z in net.graph.successors(y[0]):
                    L = list(set().union(L, [(z,"down")]))
            elif y[1] == "down":
                if not y[0] in Z:
                    for z in net.graph.successors(y[0]):
                       L = list(set().union(L, [(z,"down")]))
                if y[0] in A:
                    for z in net.graph.predecessors(y[0]):
                        L = list(set().union(L, [(z,"up")]))
    
    return R
                
                
