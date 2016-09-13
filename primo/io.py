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

import re
import xml.dom.minidom as minidom

import primo.networks
import primo.nodes


class XMLBIF(object):
    '''
    This class represents the Interchange Format for Bayesian Networks (XMLBIF).
    It helps you to convert a BayesNet to a XMLBIF and a XMLBIF to a BayesNet.

    See: http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/
    '''
    def __init__(self, network, network_name = "Unnamed network",
                 encoding = "UTF-8", ndent = "    ", newl = "\n",
                 addindent = "    "):
        '''
        Create a new XMLBIF instance.

        Keyword arguments:
        network -- is a valid BayesNet that must only contain DicreteNodes.
        network_name -- is some name that will be mentioned in the XMLBIF.
        encoding -- encoding of the XMLBIF. Encoding other than UTF-8 is
        likely incorrect, since UTF-8 is the default encoding of XML.
        ndent -- specifies the indentation string and defaults to a tabulator.
        newl -- specifies the string emitted at the end of each line.
        addindent -- is the incremental indentation to use for subnodes of the current one
        '''
        self.network = network
        self.network_name = network_name
        self.encoding = encoding
        self.ndent = ndent
        self.newl = newl
        self.addindent = addindent
        self.root = minidom.Document()
        if isinstance(network, primo.networks.BayesianNetwork):
            self.network = network
        else:
            raise Exception("Given network is not a BayesNet.")
        # Create inital XMLBIF
        self.generate_XMLBIF()

    def __str__(self):
        '''
        Returns a pretty string representation of the XMLBIF.
        '''
        return self.root.toprettyxml(self.ndent, self.newl, self.encoding);

    def write(self, filename):
        '''
        Write this XMLBIF instance to disk.

        Keyword arguments:
        filename -- is a string containing the filename.
        '''
        f = open(filename, "w")
        self.root.writexml(f, self.ndent, self.addindent, self.newl, self.encoding)



    def generate_XMLBIF(self):
        '''
        Generate the XMLBIF document.

        This method is used internally. Do not call it outside this class.
        '''
        self.calculate_positions()
        root_node = minidom.Document()
        tag_bif = root_node.createElement("BIF")
        tag_net = root_node.createElement("NETWORK")
        tag_bif.setAttribute("VERSION","0.3")
        tag_bif.setAttribute("xmlns", "http://www.cs.ubc.ca/labs/lci/fopi/ve/XMLBIFv0_3")
        tag_bif.setAttribute("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        tag_bif.setAttribute("xsi:schemaLocation", "http://www.cs.ubc.ca/labs/lci/fopi/ve/XMLBIFv0_3 http://www.cs.ubc.ca/labs/lci/fopi/ve/XMLBIFv0_3/XMLBIFv0_3.xsd")
        root_node.appendChild(tag_bif)
        tag_bif.appendChild(tag_net)

        tag_name = minidom.Element("NAME")
        text = minidom.Text()
        text.data = str(self.network_name)
        tag_name.appendChild(text)
        tag_net.appendChild(tag_name)

        for node_name in self.network.node_lookup:
            current_node = self.network.node_lookup[node_name]
            if not (isinstance(current_node, primo.nodes.DiscreteNode) 
                | isinstance(current_node, primo.nodes.DecisionNode)
                | isinstance(current_node, primo.nodes.UtilityNode)):
                raise Exception("Node " + str(current_node) + " is not a DiscreteNode.")
            node_tag = self.create_node_tag(current_node)
            tag_net.appendChild(node_tag)

        #Generate CPTs
        for node_name in self.network.node_lookup:
            current_node = self.network.node_lookup[node_name]
            tag_def = minidom.Element("DEFINITION")
            tag_for = minidom.Element("FOR")
            txt_for = minidom.Text()
            txt_for.data = node_name
            tag_for.appendChild(txt_for)
            tag_def.appendChild(tag_for)


            if(isinstance(current_node, primo.nodes.DiscreteNode)):
                # It's not guaranteed that the own node is at dimension zero in
                # the probability table.But for the function the order of the
                # variables is important                
                for parent in filter(lambda x: x.name != current_node.name, reversed(current_node.get_cpd().get_variables())):
                    tag_par = minidom.Element("GIVEN")
                    txt_par = minidom.Text()
                    txt_par.data = str(parent.name)
                    tag_par.appendChild(txt_par)
                    tag_def.appendChild(tag_par)

                tag_cpt = minidom.Element("TABLE")
                txt_cpt = minidom.Text()
                txt = ""
                for elem in current_node.get_cpd().get_table().T.flat:
                    txt += str(elem) + " "

                txt_cpt.data = txt
                tag_cpt.appendChild(txt_cpt)
                tag_def.appendChild(tag_cpt)

                tag_net.appendChild(tag_def)
                
            if(isinstance(current_node, primo.nodes.UtilityNode)):
                for parent in reversed(current_node.get_utility_table().get_variables()):
                    tag_par = minidom.Element("GIVEN")
                    txt_par = minidom.Text()
                    txt_par.data = str(parent.name)
                    tag_par.appendChild(txt_par)
                    tag_def.appendChild(tag_par)
                tag_cpt = minidom.Element("TABLE")
                txt_cpt = minidom.Text()
                txt = ""

                for elem in current_node.get_utility_table().get_utility_table().T.flat:
                    txt += str(elem) + " "

                txt_cpt.data = txt
                tag_cpt.appendChild(txt_cpt)
                tag_def.appendChild(tag_cpt)

                tag_net.appendChild(tag_def)
                
            if(isinstance(current_node, primo.nodes.DecisionNode)):   
                for parent in reversed(current_node.get_parents()):
                    tag_par = minidom.Element("GIVEN")
                    txt_par = minidom.Text()
                    txt_par.data = str(parent)
                    tag_par.appendChild(txt_par)
                    tag_def.appendChild(tag_par)
                tag_net.appendChild(tag_def)
                
        self.root = root_node
        return self



    def create_node_tag(self, node):
        '''
        Create a node tag that will look like:
        <VARIABLE TYPE="nature">
            <NAME>node_name</NAME>
            <OUTCOME>...</OUTCOME>
            <OUTCOME>...</OUTCOME>
            <PROPERTY>position = (x, y)</PROPERTY>
        </VARIABLE>

        Keyword arguments:
        node -- a Node with valid name and position

        Returns a XMLBIF conform "variable" tag
        '''
        if not isinstance(node, primo.nodes.Node):
            raise Exception("Node " + str(node) + " is not a Node.")
        tag_var = minidom.Element("VARIABLE")
        tag_own = minidom.Element("NAME")
        tag_pos = minidom.Element("PROPERTY")
        
        
        if(isinstance(node, primo.nodes.DiscreteNode)):
            tag_var.setAttribute("TYPE", "nature")
        if(isinstance(node, primo.nodes.DecisionNode)):
            tag_var.setAttribute("TYPE", "decision")
        if(isinstance(node, primo.nodes.UtilityNode)):
            tag_var.setAttribute("TYPE", "utility")
        
        # set node name
        txt_name = minidom.Text()
        txt_name.data = node.name
        tag_var.appendChild(tag_own)
        tag_own.appendChild(txt_name)

        # set outcomes
        if(not isinstance(node, primo.nodes.UtilityNode)):
            for value in node.value_range:
                tag_outcome = minidom.Element("OUTCOME")
                txt_outcome = minidom.Text()
                txt_outcome.data = value
                tag_outcome.appendChild(txt_outcome)
                tag_var.appendChild(tag_outcome)

        # set position
        txt_pos = minidom.Text()
        x, y = node.position
        txt_pos.data = "position = (" + str(x) + ", " + str(y) + ")"
        tag_pos.appendChild(txt_pos)
        tag_var.appendChild(tag_pos)

        return tag_var



    def calculate_positions(self):
        '''
        Calculate the visual position for each node.

        This method is used internally. Do not call it outside this class.
        '''
        q = []
        p = []
        already_seen = []
        x_step = 150
        y_step = 100
        x_pos = 0
        y_pos = 0
        for node_name in self.network.node_lookup:
            node = self.network.node_lookup[node_name]
            if len(self.network.graph.predecessors(node)) == 0:
                q.append(node)
                already_seen.append(node)
        while q:
            p = q
            q = []
            y_pos += y_step
            x_pos = x_step
            while p:
                node = p.pop()
                node.position = (x_pos, y_pos)
                x_pos += x_step

                for child in self.network.graph.successors(node):
                    if not child in already_seen:
                        q.append(child)
                        already_seen.append(child)

    @staticmethod
    def read(filename_or_file, is_string = False):
        '''
        Reads a XMLBIF and returns a BayesNet.

        Keyword arguments:
        filename_or_file -- may be either a file name, or a file-like object.
        is_string -- is True if filename_or_file is a XML in a string

        Returns a BayesNet.
        '''
        if is_string:
            root = minidom.parseString(filename_or_file)
        else:
            root = minidom.parse(filename_or_file)

        return XMLBIF.generate_BayesNet(root)

    @staticmethod
    def readDN(filename_or_file, is_string = False):
        '''
        Reads a XMLBIF and returns a DecisionNet
        '''
        if is_string:
            root = minidom.parseString(filename_or_file)
        else:
            root = minidom.parse(filename_or_file)

        return XMLBIF.generate_DecisionNet(root)
    
    @staticmethod
    def read_DN_xdsl(filename_or_file, is_2TDN = False, is_string = False):
        '''
        Reads a xdsl (Genie&Smile Format) and returns a DecisionNet
        '''
        if is_string:
            root = minidom.parseString(filename_or_file)
        else:
            root = minidom.parse(filename_or_file)

        return XMLBIF.generate_DecisionNetXdsl(root, is_2TDN)
    
    @staticmethod
    def generate_DecisionNet(root):
        network = primo.networks.BayesianDecisionNetwork()
        bif_nodes = root.getElementsByTagName("BIF")
        if len(bif_nodes) != 1:
            raise Exception("More than one or none <BIF>-tag in document.")
        network_nodes = bif_nodes[0].getElementsByTagName("NETWORK")
        if len(network_nodes) != 1:
            raise Exception("More than one or none <NETWORK>-tag in document.")
        
        variable_nodes = network_nodes[0].getElementsByTagName("VARIABLE")
        for variable_node in variable_nodes:
            name = "Unnamed node"
            value_range = []
            position = (0, 0)                    
            for name_node in variable_node.getElementsByTagName("NAME"):
                name = XMLBIF.get_node_text(name_node.childNodes)
                break
            for output_node in variable_node.getElementsByTagName("OUTCOME"):
                value_range.append(XMLBIF.get_node_text(output_node.childNodes))
            for position_node in variable_node.getElementsByTagName("PROPERTY"):
                position = XMLBIF.get_node_position_from_text(position_node.childNodes)
                break
            
            #print(str(variable_node.attributes["TYPE"].value))
            if("nature" == str(variable_node.attributes["TYPE"].value)):
                new_node = primo.nodes.DiscreteNode(name, value_range)
                new_node.position = position
            else:
                if("decision" == str(variable_node.attributes["TYPE"].value)):
                    new_node = primo.nodes.DecisionNode(name, value_range)
                    new_node.position = position    
                if("utility" == str(variable_node.attributes["TYPE"].value)):
                    new_node = primo.nodes.UtilityNode(name)
                    new_node.position = position
            network.add_node(new_node)
        
        definition_nodes = network_nodes[0].getElementsByTagName("DEFINITION")
        for definition_node in definition_nodes:
            node = None
            for for_node in definition_node.getElementsByTagName("FOR"):
                name = XMLBIF.get_node_text(for_node.childNodes)
                node = network.get_node(name)
                break
            if node == None:
                continue
            for given_node in reversed(definition_node.getElementsByTagName("GIVEN")):
                parent_name = XMLBIF.get_node_text(given_node.childNodes)
                parent_node = network.get_node(parent_name)
                network.add_edge(parent_node, node)
            if(not isinstance(node, primo.nodes.DecisionNode)):    
                for table_node in definition_node.getElementsByTagName("TABLE"):
                    table = XMLBIF.get_node_table_from_text(table_node.childNodes)
                    if(isinstance(node, primo.nodes.DiscreteNode)):
                        node.get_cpd().get_table().T.flat = table
                        break
                    else:
                        node.get_utility_table().get_utility_table().T.flat = table
                        break
        return network
    
    
    @staticmethod    
    def generate_DecisionNetXdsl(root, is_2TDN):
        if is_2TDN:
            network = primo.networks.TwoTDN()
        else:
            network = primo.networks.BayesianDecisionNetwork()
        
        smile_nodes = root.getElementsByTagName("smile")
        
        network_nodes = smile_nodes[0].getElementsByTagName("nodes")
        
        #add all nodes to the network
        variable_cpts = network_nodes[0].getElementsByTagName("cpt")
        for cpt_node in variable_cpts:
            name = str(cpt_node.attributes["id"].value)
            value_range = []
            for state in cpt_node.getElementsByTagName("state"):
                value_range.append(str(state.attributes["id"].value))    
            new_node = primo.nodes.DiscreteNode(name, value_range)
            network.add_node(new_node)
            
        variable_decs = network_nodes[0].getElementsByTagName("decision")
        for dec_node in variable_decs:
            name = str(dec_node.attributes["id"].value)
            value_range = []
            for state in dec_node.getElementsByTagName("state"):
                value_range.append(str(state.attributes["id"].value))
            new_node = primo.nodes.DecisionNode(name, value_range)
            network.add_node(new_node)
            
        variable_uts = network_nodes[0].getElementsByTagName("utility")
        for ut_node in variable_uts:
            name = str(ut_node.attributes["id"].value)
            new_node = primo.nodes.UtilityNode(name)
            network.add_node(new_node)
        
        
        #set parents and probability and utility tables
        for cpt_node in variable_cpts:
            xdsl_node = cpt_node.getElementsByTagName("parents")
            node = network.get_node(str(cpt_node.attributes["id"].value))
            if(xdsl_node):
                parents = XMLBIF.get_node_text(xdsl_node[0].childNodes)
            
                if parents:
                    for parent_name in reversed(parents.split(" ")):
                        network.add_edge(network.get_node(parent_name), node) 
            
            cpt = cpt_node.getElementsByTagName("probabilities")
            table = XMLBIF.get_node_table_from_text(cpt[0].childNodes)
            node.get_cpd().get_table().T.flat = table
            
        for ut_node in variable_uts:
            xdsl_node = ut_node.getElementsByTagName("parents")
            parents = XMLBIF.get_node_text(xdsl_node[0].childNodes)
            node = network.get_node(str(ut_node.attributes["id"].value))
            if parents:
                for parent_name in reversed(parents.split(" ")):    #reversed??
                    network.add_edge(network.get_node(parent_name), node)
            
            ut = ut_node.getElementsByTagName("utilities")
            table = XMLBIF.get_node_table_from_text(ut[0].childNodes)
            node.get_utility_table().get_utility_table().T.flat = table
        
        #TODO: set up partialordering automatically from decisonnodes recall edges
#        for dec_node in variable_decs:
#            xdsl_node = dec_node.getElementsByTagName("parents")
#            parents = XMLBIF.get_node_text(xdsl_node[0].childNodes)
#            node = network.get_node(str(dec_node.attributes["id"].value))
#            if parents:
#                for parent_name in parents.split(" "):
                    
            
        #set position of nodes
        network_extensions = smile_nodes[0].getElementsByTagName("extensions")
        genie = network_extensions[0].getElementsByTagName("genie")
        
        for xdsl_node in genie[0].getElementsByTagName("node"):
            node = network.get_node(str(xdsl_node.attributes["id"].value))
            position = xdsl_node.getElementsByTagName("position")
            pos = XMLBIF.get_node_text(position[0].childNodes)
            pos = pos.split(" ")
            xpos = pos[0]
            ypos = pos[1]
            node.pos = (float(xpos), float(ypos))
        
        #set initial_nodes if 2TDN
        if is_2TDN:
            network_extensions = smile_nodes[0].getElementsByTagName("twoTDN")
            interslice = network_extensions[0].getElementsByTagName("interslice")
            
            for xdsl_node in interslice[0].getElementsByTagName("node"):
                node_0 = xdsl_node.getElementsByTagName("name")
                node_t = xdsl_node.getElementsByTagName("corresponding")
                n0_name = XMLBIF.get_node_text(node_0[0].childNodes)
                nt_name = XMLBIF.get_node_text(node_t[0].childNodes)
                network.set_initial_node(n0_name, nt_name)
            
        return network
            
            
    @staticmethod
    def generate_BayesNet(root):
        '''
        Generate a BayesNet from a XMLBIF.

        This method is used internally. Do not call it outside this class.
        '''
        network = primo.networks.BayesianNetwork()
        bif_nodes = root.getElementsByTagName("BIF")
        if len(bif_nodes) != 1:
            raise Exception("More than one or none <BIF>-tag in document.")
        network_nodes = bif_nodes[0].getElementsByTagName("NETWORK")
        if len(network_nodes) != 1:
            raise Exception("More than one or none <NETWORK>-tag in document.")
        variable_nodes = network_nodes[0].getElementsByTagName("VARIABLE")
        for variable_node in variable_nodes:
            name = "Unnamed node"
            value_range = []
            position = (0, 0)
            for name_node in variable_node.getElementsByTagName("NAME"):
                name = XMLBIF.get_node_text(name_node.childNodes)
                break
            for output_node in variable_node.getElementsByTagName("OUTCOME"):
                value_range.append(XMLBIF.get_node_text(output_node.childNodes))
            for position_node in variable_node.getElementsByTagName("PROPERTY"):
                position = XMLBIF.get_node_position_from_text(position_node.childNodes)
                break
            new_node = primo.nodes.DiscreteNode(name, value_range)
            new_node.position = position
            network.add_node(new_node)
        definition_nodes = network_nodes[0].getElementsByTagName("DEFINITION")
        for definition_node in definition_nodes:
            node = None
            for for_node in definition_node.getElementsByTagName("FOR"):
                name = XMLBIF.get_node_text(for_node.childNodes)
                node = network.get_node(name)
                break
            if node == None:
                continue
            for given_node in reversed(definition_node.getElementsByTagName("GIVEN")):
                parent_name = XMLBIF.get_node_text(given_node.childNodes)
                parent_node = network.get_node(parent_name)
                network.add_edge(parent_node, node)
            for table_node in definition_node.getElementsByTagName("TABLE"):
                table = XMLBIF.get_node_table_from_text(table_node.childNodes)
                node.get_cpd().get_table().T.flat = table
                break

        return network

    @staticmethod
    def get_node_text(nodelist):
        '''
        Keyword arguments:
        nodelist -- is a list of nodes (xml.dom.minidom.Node).

        Returns the text of the given nodelist or a empty string.
        '''
        rc = []
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc.append(node.data)
        return ''.join(rc)

    @staticmethod
    def get_node_position_from_text(nodelist):
        '''
        Keyword arguments:
        nodelist -- is a list of nodes (xml.dom.minidom.Node).

        Returns the position of the given nodelist as pair (x, y).
        '''
        rc = []
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc.append(node.data)
        text = ''.join(rc)
        number_list = re.findall(r"[0-9]+.[0-9]+", text) #re.findall(r"\d+", text)
        #print("text: "+str(text))
        #print("list: "+str(number_list))
        
        if len(number_list) != 2:
            raise Exception("Ambiguous node position in XMLBIF.")
        return (number_list[0], number_list[1])

    @staticmethod
    def get_node_table_from_text(nodelist):
        '''
        Keyword arguments:
        nodelist -- is a list of nodes (xml.dom.minidom.Node).

        Returns the probability table of the given nodelist as pair numpy.array.
        '''
        rc = []
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc.append(node.data)
        text = ''.join(rc)
        number_list = re.findall(r"-?[0-9]*\.*[0-9]+", text)
        for (i, n) in enumerate(number_list):
            number_list[i] = float(n)
        return number_list


def create_DBN_from_spec(dbn_spec):
    '''
    Keyword arguments:
    dbn_spec -- is a filepath to a JSON specification of a dynamic Bayesian network

    Example:
    > {
    >     "B0": "b0_network.xbif",
    >     "TBN": "tbn_network.xbif",
    >     "transitions": [
    >         ["node_a_t0", "node_a"],
    >         ["node_b_t0", "node_b"]
    >     ]
    > }

    Returns an instantiated dynamic Bayesian network.
    '''
    import json
    with open(dbn_spec) as json_data:
        spec = json.load(json_data)
        dbn = primo.networks.DynamicBayesianNetwork()
    dbn.B0 = XMLBIF.read(spec['B0'])
    tbn = primo.networks.TwoTBN(XMLBIF.read(spec['TBN']))
    dbn.twoTBN = tbn
    for transition in spec['transitions']:
        tbn.set_initial_node(transition[0], transition[1])
    return dbn