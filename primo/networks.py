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

import networkx as nx

import primo.densities
import primo.inference.factor
import primo.nodes

class BayesianNetwork(object):

    def __init__(self):
        super(BayesianNetwork, self).__init__()
        self.graph = nx.DiGraph()
        self.node_lookup = {}

    def add_node(self, node):
        if isinstance(node, primo.nodes.Node):
            if node.name in self.node_lookup.keys():
                raise Exception("Node name already exists in BayesNetwork: " + node.name)
            self.node_lookup[node.name]=node
            self.graph.add_node(node)
        else:
            raise Exception("Can only add 'Node' and its subclasses as nodes into the BayesianNetwork")

    def add_edge(self, node_from, node_to):
        if node_from in self.graph.nodes() and node_to in self.graph.nodes():
            self.graph.add_edge(node_from, node_to)
            node_to.announce_parent(node_from)
        else:
            raise Exception("Tried to add an Edge between two Nodes of which at least one was not contained in the Bayesnet")

    def remove_node(self, node):
        if node.name not in self.node_lookup.keys():
            raise Exception("Node " + node.name + "does not exists")
        else :
            try:
                self.graph.remove_node(node)
            except nx.exception.NetworkXError:
                raise Exception("Tried to remove a node which does not exist.")
            del self.node_lookup[node.name]

    def remove_edge(self, node_from, node_to):
        try:
            self.graph.remove_edge(node_from, node_to)
        except nx.exception.NetworkXError:
            raise Exception("Tried to remove an edge which does not exist in the BayesianNetwork")
        #raise Exception("Fixme: Adapt CPD of child-node")

    def get_node(self, node_name):
        try:
            return self.node_lookup[node_name]
        except KeyError:
            raise Exception("There is no node with name "+node_name+" in the BayesianNetwork")

    def get_all_nodes(self):
        return self.graph.nodes()

    def get_nodes(self, node_names=[]):
        nodes = []
        if not node_names:
            nodes = self.graph.nodes()
        else:
            for node_name in node_names:
                nodes.append(self.get_node(node_name))
        return nodes

    def get_nodes_in_topological_sort(self):
        return nx.topological_sort(self.graph)

    def get_parents(self, node):
        if node.name not in self.node_lookup.keys():
            raise Exception("Node " + node.name + "does not exists")
        else:
            return self.graph.predecessors(node)

    def get_children(self, node):
        if node.name not in self.node_lookup.keys():
            raise Exception("Node " + node.name + "does not exists")
        else:
            return self.graph.successors(node)

    def get_markov_blanket(self, node):
        raise Exception("Called unimplemented function")

    def is_dag(self):
        raise Exception("Called unimplemented function")

    def draw(self):
        import matplotlib.pyplot as plt
        nx.draw_circular(self.graph)
        plt.show()

    def draw_graphviz(self):
        import matplotlib.pyplot as plt
        nx.draw_graphviz(self.graph)
        plt.show()

    def is_valid(self):
        '''Check if graph structure is valid.
        Returns true if graph is directed and acyclic, false otherwiese'''
        if self.graph.number_of_selfloops() > 0:
            return False
        for node in self.graph.nodes():
            if self.has_loop(node):
                return False
        return True

    def has_loop(self, node, origin=None):
        '''Check if any path from node leads back to node.

        Keyword arguments:
        node -- the start node
        origin -- for internal recursive loop (default: None)

        Returns true on succes, false otherwise.'''
        if not origin:
            origin = node
        for successor in self.graph.successors(node):
            if successor == origin:
                return True
            else:
                return self.has_loop(successor, origin)

    def clear(self):
        '''Remove all nodes and edges from the graph.
        This also removes the name, and all graph, node and edge attributes.'''
        self.graph.clear()
        self.node_lookup.clear()

    def number_of_nodes(self):
        '''Return the number of nodes in the graph.'''
        return len(self)

    def __len__(self):
        '''Return the number of nodes in the graph.'''
        return len(self.graph)


class BayesianDecisionNetwork(BayesianNetwork):

    def __init__(self):
        super(BayesianDecisionNetwork, self).__init__()
        self.partialOrdering = []
        self.random_nodes = []
        self.decision_nodes = []
        self.utility_nodes = []
        self.relevance_graph = nx.DiGraph()

    def is_valid(self):
        '''Check if graph structure is valid.
        Returns true if graph is directed, acyclic and if there is a path that connects every decision node(consistency check),
        false otherwise'''
        if self.graph.number_of_selfloops() > 0:
            return False
        for node in self.graph.nodes():
            if self.has_loop(node):
                return False
        decisionNodeList = []
        for node in self.get_all_nodes():
            if isinstance(node, primo.nodes.DecisionNode):
                decisionNodeList.append(node)

        return all([nx.has_path(self.graph, x, y) == True for x in decisionNodeList for y in decisionNodeList])

    def add_node(self, node):
        if isinstance(node, primo.nodes.Node):
            if node.name in self.node_lookup.keys():
                raise Exception("Node name already exists in Bayesnet: "+node.name)
            if isinstance(node, primo.nodes.DiscreteNode):
                self.random_nodes.append(node)
            elif isinstance(node, primo.nodes.UtilityNode):
                self.utility_nodes.append(node)
            elif isinstance(node, primo.nodes.DecisionNode):
                self.decision_nodes.append(node)
            else:
                raise Exception("Tried to add a node which the Bayesian Decision Network can not work with")
            self.node_lookup[node.name]=node
            self.graph.add_node(node)
        else:
            raise Exception("Can only add 'Node' and its subclasses as nodes into the BayesianNetwork")

    def get_all_nodes(self):
        '''Returns all RandomNodes'''
        return self.random_nodes

    def get_all_decision_nodes(self):
        return self.decision_nodes

    def get_all_utility_nodes(self):
        return self.utility_nodes
    
    def get_node_from_DN(self, node_name):
        '''Returns the corresponding node to node_name from the DN, wether chance, decision or utility'''
        if isinstance(node_name, primo.nodes.Node):
            node_name = node_name.name
        try:
            return self.node_lookup[node_name]
        except KeyError:
            raise Exception("There is no node with name "+node_name+" in the DecisionNetwork")
    
    def get_all_DN_nodes(self):
        ''' Returns all Nodes (chance,decision,utility from the DN'''
        return self.random_nodes+self.decision_nodes+self.utility_nodes

    def add_edge(self, node_from, node_to):
        """
        Adds an edge between two nodes. It is impossible to create en edge between two decision nodes and between two
        utility nodes.

        keyword arguments:
        node_from -- Node from where the edge shall begin
        node_to -- Node where the edge shall end
        """
        if isinstance(node_from, primo.nodes.DecisionNode) and isinstance(node_to, primo.nodes.DecisionNode):
            raise Exception("Tried to add an edge from a DecisionNode to a DecisionNode")
        if isinstance(node_from, primo.nodes.UtilityNode) and isinstance(node_to, primo.nodes.UtilityNode):
            raise Exception("Tried to add an edge from a UtilityNode to a UtilityNode")
        if node_from in self.graph.nodes() and node_to in self.graph.nodes():
            self.graph.add_edge(node_from, node_to)
            node_to.announce_parent(node_from)
        else:
            raise Exception("Tried to add an Edge between two Nodes of which at least one was not contained in the Bayesnet")

    def reset_Decisions(self):
        """
        Resets all decisions in the Bayesian Decision Network
        """
        for node in self.decision_nodes:
            node.set_state(None)

    def get_partialOrdering(self):
        """
        Getter for the partial ordere
        """
        return self.partialOrdering

    def set_partialOrdering(self, partialOrder):
        """
        Sets the partial ordering for this Bayesian Decision Network

        partialOrder -- ordered list of RandomNodes and Decision Nodes
        example: [decisionNode1, [randomNode1,randomNode2], decisionNode2, [randomNode3]]
        """
        self.partialOrdering = partialOrder
    
    def set_evidence(self, evidence):
        for item in evidence:
            tmp = self.node_lookup[item[0].name]
            #in some cases reduction leads to problems with multiplication of cpds
#            reduced = tmp.get_cpd_reduced(evidence)
            reduced = tmp.set_evidence(item)
            tmp.set_cpd(reduced)
            
    def get_utNodes_relevant_to_decNode(self, decisionNode):
        '''
        returns all utNodes that are descendandts of decisionNode
        '''
        u_d = []
        
        all_descendants = nx.descendants(self.graph, decisionNode)
        for node in all_descendants:
            if isinstance(node, primo.nodes.UtilityNode):
                u_d.append(node)
        
        return u_d
    
    def init_relevance_graph(self):
        
        for dec in self.get_all_decision_nodes():
            self.relevance_graph.add_node(dec)
            
            

class DynamicBayesianNetwork(BayesianNetwork):
    ''' This is the implementation of a dynamic Bayesian network (also called
    temporal Bayesian network).

    Definition: DBN is a pair (B0, TwoTBN), where B0 is a BN over X(0),
    representing the initial distribution over states, and TwoTBN is a
    2-TBN for the process.
    See Koller, Friedman - "Probabilistic Graphical Models" (p. 204)

    Properties: Markov property, stationary, directed, discrete,
    acyclic (within a slice)
    '''

    def __init__(self, b0=None, two_tbn=None):
        super(DynamicBayesianNetwork, self).__init__()
        self._B0 = BayesianNetwork() if b0 is None else b0
        self._twoTBN = TwoTBN() if two_tbn is None else two_tbn
        self._t = 0

    @property
    def B0(self):
        ''' Get the Bayesian network representing the initial distribution.'''
        return self._B0

    @B0.setter
    def B0(self, value):
        ''' Set the Bayesian network representing the initial distribution.'''
        if isinstance(value, BayesianNetwork):
            if not value.is_valid():
                raise Exception("BayesianNetwork is not valid.")
            self._B0 = value
        else:
            raise Exception("Can only set 'BayesianNetwork' and its subclasses as " +
            "B0 of a DBN.")

    @property
    def twoTBN(self):
        ''' Get the 2-time-slice Bayesian network.'''
        return self._twoTBN

    @twoTBN.setter
    def twoTBN(self, value):
        ''' Set the 2-time-slice Bayesian network.'''
        if isinstance(value, TwoTBN):
            if not value.is_valid():
                raise Exception("BayesianNetwork is not valid.")
            self._twoTBN = value
        else:
            raise Exception("Can only set 'TwoTBN' and its subclasses as " +
            "twoTBN of a DBN.")

    @property
    def t(self):
        return self._t

    def t_plus_1(self, evidence=None):
        state = {}
        if self._t == 0:
            ft = primo.inference.factor.FactorTreeFactory().create_greedy_factortree(self._B0)
            state_vars = [self._B0.get_node(n.name) for (_, n) in self._twoTBN.get_initial_nodes()]
        else:
            ft = primo.inference.factor.FactorTreeFactory().create_greedy_factortree(self._twoTBN)
            state_vars = [nt for (_, nt) in self._twoTBN.get_initial_nodes()]
        ft.set_evidence(evidence)
        for var in state_vars:
            state[var] = ft.calculate_marginal([var]).table
        next_tslice = self._twoTBN.create_timeslice(state, True if self._t == 0 else False)
        self._twoTBN = next_tslice
        self._t += 1

    def inference(self, evidence=None):
        state = {}
        if self._t == 0:
            ft = primo.inference.factor.FactorTreeFactory().create_greedy_factortree(self._B0)
            state_vars = [self._B0.get_node(n.name) for (_, n) in self._twoTBN.get_initial_nodes()]
        else:
            ft = primo.inference.factor.FactorTreeFactory().create_greedy_factortree(self._twoTBN)
            state_vars = [nt for (_, nt) in self._twoTBN.get_initial_nodes()]
        ft.set_evidence(evidence)
        for var in state_vars:
            state[var] = ft.calculate_marginal([var]).table
        return state

    def is_valid(self):
        '''Check if graph structure is valid. And if there is a same-named
        inital node in towTBN for every node in BO.
        Returns true if graph is directed and acyclic, false otherwiese'''
        valid = True
        for node in self._B0.get_nodes():
            if not self._twoTBN.has_initial_node_by_name(node.name):
                print("Node with name " + str(node.name) +
                " not found in TwoTBN!")
                valid = False
        return False if not valid else super(DynamicBayesianNetwork, self).is_valid()


class TwoTBN(BayesianNetwork):
    ''' This is the implementation of a 2-time-slice Bayesian network (2-TBN).
    '''

    def __init__(self, bayesnet=None):
        BayesianNetwork.__init__(self)
        if bayesnet:
            if not isinstance(bayesnet, BayesianNetwork):
                raise Exception("Parameter 'bayesnet' is not a instance of class BayesianNetwork.")
            self.graph = bayesnet.graph
            self.node_lookup = bayesnet.node_lookup
        self._initial_nodes = []

    def create_timeslice(self, state, initial=False):
        '''
        Set all initial nodes to the value of their corresponding nodes
        in state (previous time slice).

        Keyword arguments:
        state -- Current state of the network (previous time slice).
        initial -- Set initial to true if this will be the first time slice
        and state only contains nodes of the initial distribution.

        Returns this instance with all initial nodes set to their
        new value.
        '''
        for (node, node_t) in self._initial_nodes:
            cpd = primo.densities.ProbabilityTable()
            cpd.add_variable(node)
            node.set_cpd(cpd)
            if not initial:
                if isinstance(state[node_t], basestring):
                    node.set_probability(1., [(node, state[node_t])])
                else:
                    # Set soft evidence
                    #print "from twotbn: setting " + str(node) + " cpt to " + str(state[node_t])
                    node.get_cpd().set_probability_table(state[node_t])
            else:
                for node0 in state:
                    if node0.name == node_t.name:
                        if isinstance(state[node0], basestring):
                            node.set_probability(1., [(node, state[node0])])
                        else:
                            # Set soft evidence
                            #print "from b0: setting " + str(node) + " cpt to " + str(state[node0])
                            node.get_cpd().set_probability_table(state[node0])
                        break
        return self


    def add_node(self, node, initial=False, node_t=None):
        '''
        Add a node to the TwoTBN.

        Keyword arguments:
        node -- Node to be added.
        initial -- If true node is marked as initial node.
        node_t -- If initial is true this is the corresponding node in the time slice.
        '''
        super(TwoTBN, self).add_node(node)
        if initial:
            self.set_initial_node(node.name, node_t.name)

    def set_initial_node(self, node_name, node_name_t):
        '''
        Mark a node as initial node.

        Keyword arguments:
        node_name -- Name of the initial node.
        node_name_t -- Name of the corresponding node in the time slice.
        '''
        node0 = self.get_node(node_name)
        node1 = self.get_node(node_name_t)
        self._initial_nodes.append((node0, node1))

    def get_initial_nodes(self):
        return self._initial_nodes

    def has_initial_node_by_name(self, node_name):
        '''
        Check if this instance has an inital node with name node_name.

        Returns true on success, false otherwise.
        '''
        return node_name in [node.name for (node, _) in self._initial_nodes]
    
class DynamicDecisionNetwork(BayesianDecisionNetwork):
    
    def __init__(self, dn0=None, two_tdn=None):
        super(DynamicDecisionNetwork, self).__init__()
        self._DN0 = BayesianDecisionNetwork() if dn0 is None else dn0
        self._twoTDN = TwoTDN() if two_tdn is None else two_tdn
        self.t = 0
        
    
    def set_DN0(self, value):
        ''' Set the initial Decisionnetwork.'''
        if isinstance(value, BayesianDecisionNetwork):
            if not value.is_valid():
                raise Exception("DecisionNetwork is not valid.")
            self._DN0 = value
        else:
            raise Exception("Can only set a 'DecisionNetwork' as DN0 of a DDN.")
        
    def set_TwoTDN(self, value):
        ''' Set the 2Timeslice-Decisionnetwork.'''
        if isinstance(value, TwoTDN):
            if not value.is_valid():
                raise Exception("2Timeslice-Decisionnetwork is not valid.")
            self._twoTDN = value
        else:
            raise Exception("Can only set a '2Timeslice-Decisionnetwork' as TwoTDN of a DDN.")

    def create_next_timeslice_twoT(self, state, initial=False):
        '''creates the next timeslice in the twoTDN. Sets initial nodes to their new
        values given in state.'''
        
        self.t += 1
        return self._twoTDN.create_timeslice(state, initial)
        
  
    def create_n_rollouts(self, number_rollouts):
        '''Creates a DN Instance of the DDN, from timeslice t to t+n, including 
        the current slice.'''
        dn = BayesianDecisionNetwork()
        
        #clone the exisiting 2TDN
        nodes0 = [n for (n, _) in self._twoTDN.get_initial_nodes()]
        self.add_nodes_to_network(dn,nodes0)

        nodes_t = [nt for (_, nt) in self._twoTDN.get_initial_nodes()]
        self.add_nodes_to_network(dn,nodes_t)
        
        partialorder = []
        for item in self._twoTDN.get_partialOrdering():
            if(type(item)== primo.nodes.DecisionNode):
                partialorder.append(dn.get_node(item.name)) 
            else:
                tmp = []
                for node in item:
                    tmp.append(dn.get_node(node.name))
                partialorder.append(tmp)
                
        #add slices
        for t in xrange(1,number_rollouts+1):
            
            print "add slice "+str(t)
            self.add_slice_to_network(dn,nodes_t,t)

        l = len(partialorder)
        for t in range(1, number_rollouts+1):
            for i in range(l/2,l):
                item = partialorder[i]
                if(type(item)== primo.nodes.DecisionNode):
                    partialorder.append(dn.get_node(item.name+"_"+str(t)))
                else:
                    tmp = []
                    for node in item:
                        tmp.append(dn.get_node(node.name+"_"+str(t)))
                    partialorder.append(tmp)
        
        print str(partialorder) + "\n"
        dn.set_partialOrdering(partialorder)
        #print "length: " + str(dn.__len__())
        return dn
    
    
    def add_slice_to_network(self, dn, nodes_t, t_plus, debug=False):
        '''adds a new slice to a network. node names get extended with "_tplus"'''
        
        new_nodes = {}
   
        #add nodes to network
        for node_t in nodes_t:
            new_name = node_t.name + "_" + str(t_plus)
                 
            if(isinstance(node_t, primo.nodes.DiscreteNode)):
                new_node = primo.nodes.DiscreteNode( new_name, node_t.get_value_range())
                dn.add_node(new_node)
                new_nodes[node_t.name] = new_node
            elif(isinstance(node_t, primo.nodes.DecisionNode)):
                new_node = primo.nodes.DecisionNode( new_name, node_t.get_value_range())
                dn.add_node(new_node)
                new_nodes[node_t.name] = new_node
            elif(isinstance(node_t, primo.nodes.UtilityNode)):
                new_node = primo.nodes.UtilityNode( new_name)
                dn.add_node(new_node)
                new_nodes[node_t.name] = new_node
        
        
        #add edges / set parents
        DN0 = [n for (n, _) in self._twoTDN.get_initial_nodes()]
        
        for node_t in nodes_t:
            if(debug): print "node: " + new_nodes[node_t.name].name
            #print "old_p: " , node_t.get_parents()
            for parent in (node_t.get_parents()):
                #inter slice parents
                if parent in DN0:
                    new_parent = self.get_node_tplus1(dn,parent,t_plus)
                    if(debug): print "p: " + str(new_parent.name)
                    dn.add_edge(new_parent, new_nodes[node_t.name])
                      
                #intra slice parents
                else:
                    new_parent = dn.get_node(parent.name + "_" + str(t_plus))
                    if(debug): print "p: " + str(new_parent.name)
                    dn.add_edge(new_parent, new_nodes[node_t.name])
            #set cpds
            if(isinstance(node_t, primo.nodes.DiscreteNode)):
                new_nodes[node_t.name].get_cpd().set_probability_table(node_t.get_cpd().get_table())
            elif(isinstance(node_t, primo.nodes.UtilityNode)):
                new_nodes[node_t.name].get_utility_table().set_utility_table(node_t.get_utility_table().get_utility_table(), None)
        
        #append new nodes to partialOrdering
        
      
    def get_node_tplus1(self, dn, node, t_plus):
        ''' Returns the corresponding node_t_? from timeslice_t-1, given a node from 
        timeslice_1 of the twoTDN. '''
        if(t_plus == 1):
            for (n, nt) in self._twoTDN.get_initial_nodes():
                if n.name == node.name:
                    return dn.get_node(nt.name)
        elif(t_plus > 1):
            for (n, nt) in self._twoTDN.get_initial_nodes():
                if n.name == node.name:
                    return dn.get_node(nt.name + "_" + str(t_plus-1))
            
        return None
       
        
    def add_nodes_to_network(self, dn, nodes):
        '''adds a copy of completly instanciated nodes to network. Nodes and CPTs must be
        set up correctly before!'''
        
        new_nodes = {}
        for node in nodes:
            if(isinstance(node, primo.nodes.DiscreteNode)):
                new_node = primo.nodes.DiscreteNode(node.name, node.get_value_range())
                dn.add_node(new_node)
                new_nodes[node.name] = new_node
            elif(isinstance(node, primo.nodes.DecisionNode)):
                new_node = primo.nodes.DecisionNode(node.name, node.get_value_range())
                if node.state != None:
                    new_node.state = node.state
                dn.add_node(new_node)
                new_nodes[node.name] = new_node
            elif(isinstance(node, primo.nodes.UtilityNode)):
                new_node = primo.nodes.UtilityNode(node.name)
                dn.add_node(new_node)
                new_nodes[node.name] = new_node
            
        for node in nodes:
            for parent in (node.get_parents()):
                dn.add_edge(dn.get_node(parent.name), dn.get_node(node.name))
            if(isinstance(node, primo.nodes.DiscreteNode)):
                new_nodes[node.name].get_cpd().set_probability_table(node.get_cpd().get_table())
            elif(isinstance(node, primo.nodes.UtilityNode)):
                new_nodes[node.name].get_utility_table().set_utility_table(node.get_utility_table().get_utility_table(), None)
        
   
    def create_t_plus1(self,evidence=None):
        #ef = primo.inference.factor.EasiestFactorElimination(None)
        state = {}
        if self._t == 0:
            nodes = self._twoTDN.get_all_nodes()
            #print("D0: " + str(self._DN0.get_all_DN_nodes()))
            #print("TwoTDN initial: ",self._twoTDN.get_initial_nodes())
            ft = primo.inference.factor.FactorTreeFactory().create_greedy_factortree(None, self._DN0.get_all_nodes())
            state_vars = [self._DN0.get_node_from_DN(n.name) for (n, _) in self._twoTDN.get_initial_nodes()]
        else:
            nodes = self._twoTDN.get_all_nodes
            ft = primo.inference.factor.FactorTreeFactory().create_greedy_factortree(None, self._twoTDN.get_all_nodes())
            state_vars = [nt for (_, nt) in self._twoTDN.get_initial_nodes()]
        ft.set_evidence(evidence)
        print(str(state_vars))
        for var in state_vars:
            if(isinstance(var,primo.nodes.DiscreteNode)):
                #print("var: "+var.name + "ef: "+str(ef.calculate_PosteriorMarginal([var],[],nodes).table))
                #print("var: "+var.name + " ft: "+str(ft.calculate_marginal([var]).table))
                state[var] = ft.calculate_marginal([var]).table
            elif(isinstance(var, primo.nodes.DecisionNode)):
                #print("var: "+var.name +" state: "+str(var.get_state()))
                state[var] = var.get_state()
            elif(isinstance(var,primo.nodes.UtilityNode)):
                #print("var: "+var.name +" ut: "+str(var.get_utility_table()))
                state[var] = var.get_utility_table().table
                
        t_plus_1_slice = self._twoTDN.create_timeslice(state, True if self._t == 0 else False)        
        return t_plus_1_slice
        
class TwoTDN(BayesianDecisionNetwork):
    
    def __init__(self, decisionnet=None):
        BayesianDecisionNetwork.__init__(self)
        if decisionnet:
            if not isinstance(decisionnet,BayesianDecisionNetwork):
                raise Exception("Parameter 'decisionnet' is not an instance of BaysianDecisionNetwork!")
            self.node_lookup = decisionnet.node_lookup
        self._initial_nodes = []
        self._initial_node_lookup = {}
        self._utility_sum = 0
        self.ut_current = 0
        
    def add_node(self, node, initial=False, node_t=None):
        """ add node"""
        super(TwoTDN,self).add_node(node)
        if initial:
            self.set_initial_node(node.name, node_t.name)
        
    def set_initial_node(self, node_name, node_name_t):
        """ Mark a Node as initial
            Arguments: name of nodes
        """
        
        node0 = self.get_node_from_DN(node_name)
        node1 = self.get_node_from_DN(node_name_t)
        if (type(node0) is type(node1)):
            self._initial_nodes.append((node0, node1))
            self._initial_node_lookup[(node0.name, node1.name)] = (node0, node1)
        else:
            raise Exception("cant set Nodes: "+node0.name+" and " + node1.name+" as initial! The Nodes have to be of the same class!")
        
    def get_initial_nodes(self):
        """Returns a List of the initial Nodes"""
        return self._initial_nodes
        
    def create_timeslice(self, state, initial=False):
        '''
        Set all initial nodes to the value of their corresponding nodes
        in state (previous time slice).

        Keyword arguments:
        state -- Current state of the network (previous time slice).
        initial -- Set initial to true if this will be the first time slice
        and state only contains nodes of the initial distribution.

        Returns this instance with all initial nodes set to their new value.
        '''
        #print("initial = " +str (initial))
        for (node, node_t) in self._initial_nodes:
            if isinstance(node, primo.nodes.DiscreteNode):
                cpd = primo.densities.ProbabilityTable()
                cpd.add_variable(node)
                node.set_cpd(cpd)
                if not initial:
                    if isinstance(state[node_t], basestring):
                        node.set_probability(1., [(node, state[node_t])])
                        #print str(node.get_cpd().get_table())
                    else:
                        # Set soft evidence
                        #print "from twotbn: setting " + str(node) + " cpt to " + str(state[node_t])
                        node.get_cpd().set_probability_table(state[node_t])
                else:
                    for node0 in state:
                        #print str(node0.name)
                        if node0.name == node.name:
                            if isinstance(state[node0], basestring):
                                node.set_probability(1., [(node, state[node0])])
                            else:
                                # Set soft evidence
                                #print "from dn0: setting " + str(node) + " cpt to " + str(state[node0])
                                node.get_cpd().set_probability_table(state[node0])
                            state.pop(node0)
                            break
            elif isinstance(node, primo.nodes.DecisionNode):
                if not initial:
                    pass
                else:
                    #print "state: " +str(state)
                    for node0 in state:
                        #print (node0.name +"=="+ node.name)
                        if node0.name == node.name:
                            if not state[node0] == None:
                                node.set_state(state[node0])
                                #print(node.name + " set decisionvalue to: " + state[node0])
                                state.pop(node0)
                                break            
        return True
