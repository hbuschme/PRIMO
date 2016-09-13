#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "mholland"
__date__ = "$Mar 11, 2016 10:22:25 AM$"

import numpy
import time

from primo.networks import BayesianDecisionNetwork
from primo.networks import TwoTDN
from primo.networks import DynamicDecisionNetwork
from primo.nodes import DecisionNode
from primo.nodes import UtilityNode
from primo.nodes import DiscreteNode
from primo.inference.decision import MakeDecision 
from primo.io import XMLBIF

class Tests():
    def __init__(self):
        self.bla = []
    
    def sreachability(self):
        dn = BayesianDecisionNetwork()

        d = DecisionNode("d", ["bla","blub"])
        d2 = DecisionNode("d2", ["blab","bblub"])
        x = DiscreteNode("x", ["blaaaa", "bleeh"])
        v = UtilityNode("v")

        dn.add_node(d)
        dn.add_node(x)
        dn.add_node(d2)
        dn.add_node(v)

        
        dn.add_edge(d, x)
        dn.add_edge(x, d2)
        dn.add_edge(x, v)
        dn.add_edge(d2, v)

        dn.set_partialOrdering([d, [x], d2]) #
        md = MakeDecision()
        md.iterated_Optimization(dn)

class DDN_Examples():
    
    def __init__(self):
        self.fisher0 = BayesianDecisionNetwork()
        self.fisher_t = TwoTDN()
    
    def rollout(self):
        ddn = DynamicDecisionNetwork()
        
        ddn.set_DN0(self.fisher0)
        ddn.set_TwoTDN(self.fisher_t)
        
        dn = ddn.create_n_rollouts(1)

        print "\n "

        makeD = MakeDecision(dn)
        #t1 = time.clock()
        #makeD.iterated_Optimization(dn)
        print makeD.max_sum2(dn.get_node("d0"))
        #t2 = time.clock()
        #print "t2 - t1 = " + str(t2-t1)
                
#        print decision
        
        
    
    def run_example(self):
        ddn = DynamicDecisionNetwork()
        
        ddn.set_DN0(self.fisher0)
        ddn.set_TwoTDN(self.fisher_t)
        print("DDN set up")
        
        makeD = MakeDecision()
        makeD.set_ddn(ddn)
        
        res = makeD.compute_DDN()
        
        print("results: "+ str(res))
        
        
        #print("slice: "+ str(slice.get_all_DN_nodes()))
    
        #xmlbif = XMLBIF(ddn._twoTDN, "test")
        #xmlbif.write("examples/Fishing.xml")
    
    def load(self):
        ddn = DynamicDecisionNetwork()
        dn0 = XMLBIF.read_DN_xdsl("/homes/mholland/MasterArbeit/DecisionNetworks/fish0.xdsl")
        d_ = XMLBIF.read_DN_xdsl("/homes/mholland/MasterArbeit/DecisionNetworks/fish_2T.xdsl", True)
        dn0.set_partialOrdering([dn0.get_node("d0"), []])
        d_.set_partialOrdering([d_.get_node("d0"), [], d_.get_node("d_t")])
        
        ddn.set_DN0(dn0)
        ddn.set_TwoTDN(d_)
        
        makeD = MakeDecision()
        makeD.set_ddn(ddn)
        
        res = makeD.compute_DDN()
        print("results: "+ str(res))
        
    def AI_test(self):
        ddn = DynamicDecisionNetwork()
        dn0 = XMLBIF.read_DN_xdsl("/homes/mholland/MasterArbeit/DecisionNetworks/SimpleAI2.xdsl")
        d_ = XMLBIF.read_DN_xdsl("/homes/mholland/MasterArbeit/DecisionNetworks/SimpleAI2_2TDN.xdsl", True) 
        dn0.set_partialOrdering([dn0.get_node("action"), []])
        d_.set_partialOrdering([d_.get_node("action"), [], d_.get_node("action_t")])
        
        ddn.set_DN0(dn0)
        ddn.set_TwoTDN(d_)
        
        makeD = MakeDecision()
        makeD.set_ddn(ddn)
        
        result = []
        #compute results for DN0(initial net)
        evidence = [(ddn._DN0.get_node_from_DN("odd_even"), "odd")]
        result.append(makeD.compute_initial_DDN(evidence))

        #compute t=2
        evidence =  [(ddn._twoTDN.get_node_from_DN("odd_even_t"), "even")] 
        result.append(makeD.compute_timestep_DDN(evidence))
        
        print result
        
    def init_fisher_example(self):
        
        #set up DN0 for fisherexample
        d0 = DecisionNode("d0", ["goFishing","noFishing"])
        s0 = DiscreteNode("s0", ["huge","med","low"])
        f0 = DiscreteNode("f0", ["shrinking","growing"])
        u0 = UtilityNode("g0")
        
        self.fisher0.add_node(d0)
        self.fisher0.add_node(s0)
        self.fisher0.add_node(f0)
        self.fisher0.add_node(u0)
        
        
        self.fisher0.add_edge(s0, u0)
        self.fisher0.add_edge(s0, f0)
        self.fisher0.add_edge(d0, u0)
        
        u0.set_utility(100, [(d0, "goFishing"), (s0, "low")])
        u0.set_utility(500, [(d0, "goFishing"), (s0, "med")])
        u0.set_utility(1000, [(d0, "goFishing"), (s0, "huge")])
        u0.set_utility(0, [(d0, "noFishing"), (s0, "low")])
        u0.set_utility(0, [(d0, "noFishing"), (s0, "med")])
        u0.set_utility(0, [(d0, "noFishing"), (s0, "huge")])
        
        s0.set_probability(0.4, [(s0, "low")])
        s0.set_probability(0.4, [(s0, "med")])
        s0.set_probability(0.2, [(s0, "huge")])
        
        f0.set_probability(0.7, [(f0, "shrinking"),(s0, "low")])
        f0.set_probability(0.5, [(f0, "shrinking"),(s0, "med")])
        f0.set_probability(0.1, [(f0, "shrinking"),(s0, "huge")])
        f0.set_probability(0.3, [(f0, "growing"),(s0, "low")])
        f0.set_probability(0.5, [(f0, "growing"),(s0, "med")])
        f0.set_probability(0.9, [(f0, "growing"),(s0, "huge")])
        
        #self.fisher0.set_partialOrdering([d0])

        #set up TowTDN for fisherexample
        d00 = DecisionNode("d0", ["goFishing","noFishing"])
        s00 = DiscreteNode("s0", ["huge","med","low"])
        f00 = DiscreteNode("f0", ["shrinking","growing"])
        u00 = UtilityNode("g0")
        
        d_t = DecisionNode("d_t", ["goFishing","noFishing"])
        s_t = DiscreteNode("s_t", ["huge","med","low"])
        f_t = DiscreteNode("f_t", ["shrinking","growing"])
        u_t = UtilityNode("g_t")
        
        self.fisher_t.add_node(d00)
        self.fisher_t.add_node(s00)
        self.fisher_t.add_node(u00)
        self.fisher_t.add_node(f00)
        self.fisher_t.add_node(d_t)
        self.fisher_t.add_node(s_t)
        self.fisher_t.add_node(f_t)
        self.fisher_t.add_node(u_t)
        
        self.fisher_t.add_edge(s00, u00)
        self.fisher_t.add_edge(d00, u00)
        self.fisher_t.add_edge(s00, f00)
        self.fisher_t.add_edge(d00, s_t)
        self.fisher_t.add_edge(s00, s_t)
        self.fisher_t.add_edge(s_t, u_t)
        self.fisher_t.add_edge(s_t, f_t)
        self.fisher_t.add_edge(d_t, u_t)
        
        s00.set_probability(0.4, [(s00, "low")])
        s00.set_probability(0.4, [(s00, "med")])
        s00.set_probability(0.2, [(s00, "huge")])
        
        f00.set_probability(0.7, [(f00, "shrinking"),(s00, "low")])
        f00.set_probability(0.5, [(f00, "shrinking"),(s00, "med")])
        f00.set_probability(0.1, [(f00, "shrinking"),(s00, "huge")])
        f00.set_probability(0.3, [(f00, "growing"),(s00, "low")])
        f00.set_probability(0.5, [(f00, "growing"),(s00, "med")])
        f00.set_probability(0.9, [(f00, "growing"),(s00, "huge")])
        
        u00.set_utility(100, [(d00, "goFishing"), (s00, "low")])
        u00.set_utility(500, [(d00, "goFishing"), (s00, "med")])
        u00.set_utility(1000, [(d00, "goFishing"), (s00, "huge")])
        u00.set_utility(0, [(d00, "noFishing"), (s00, "low")])
        u00.set_utility(0, [(d00, "noFishing"), (s00, "med")])
        u00.set_utility(0, [(d00, "noFishing"), (s00, "huge")])
        
        s_t.set_probability(0.8, [(s_t, "low"),(d00, "goFishing"),(s00, "low")])
        s_t.set_probability(0.7, [(s_t, "low"),(d00, "goFishing"),(s00, "med")])
        s_t.set_probability(0.5, [(s_t, "low"),(d00, "goFishing"),(s00, "huge")])
        s_t.set_probability(0.4, [(s_t, "low"),(d00, "noFishing"),(s00, "low")])
        s_t.set_probability(0.1, [(s_t, "low"),(d00, "noFishing"),(s00, "med")])
        s_t.set_probability(0.1, [(s_t, "low"),(d00, "noFishing"),(s00, "huge")])        
        s_t.set_probability(0.1, [(s_t, "med"),(d00, "goFishing"),(s00, "low")])
        s_t.set_probability(0.2, [(s_t, "med"),(d00, "goFishing"),(s00, "med")])
        s_t.set_probability(0.4, [(s_t, "med"),(d00, "goFishing"),(s00, "huge")])
        s_t.set_probability(0.5, [(s_t, "med"),(d00, "noFishing"),(s00, "low")])
        s_t.set_probability(0.6, [(s_t, "med"),(d00, "noFishing"),(s00, "med")])
        s_t.set_probability(0.3, [(s_t, "med"),(d00, "noFishing"),(s00, "huge")])        
        s_t.set_probability(0.1, [(s_t, "huge"),(d00, "goFishing"),(s00, "low")])
        s_t.set_probability(0.1, [(s_t, "huge"),(d00, "goFishing"),(s00, "med")])
        s_t.set_probability(0.1, [(s_t, "huge"),(d00, "goFishing"),(s00, "huge")])
        s_t.set_probability(0.1, [(s_t, "huge"),(d00, "noFishing"),(s00, "low")])
        s_t.set_probability(0.3, [(s_t, "huge"),(d00, "noFishing"),(s00, "med")])
        s_t.set_probability(0.6, [(s_t, "huge"),(d00, "noFishing"),(s00, "huge")])
        
        f_t.set_probability(0.7, [(f_t, "shrinking"),(s_t, "low")])
        f_t.set_probability(0.5, [(f_t, "shrinking"),(s_t, "med")])
        f_t.set_probability(0.1, [(f_t, "shrinking"),(s_t, "huge")])
        f_t.set_probability(0.3, [(f_t, "growing"),(s_t, "low")])
        f_t.set_probability(0.5, [(f_t, "growing"),(s_t, "med")])
        f_t.set_probability(0.9, [(f_t, "growing"),(s_t, "huge")])
        
        u_t.set_utility(100, [(d_t, "goFishing"), (s_t, "low")])
        u_t.set_utility(500, [(d_t, "goFishing"), (s_t, "med")])
        u_t.set_utility(1000, [(d_t, "goFishing"), (s_t, "huge")])
        u_t.set_utility(0, [(d_t, "noFishing"), (s_t, "low")])
        u_t.set_utility(0, [(d_t, "noFishing"), (s_t, "med")])
        u_t.set_utility(0, [(d_t, "noFishing"), (s_t, "huge")])
        
        self.fisher_t.set_partialOrdering([d00, [s00,f00], d_t, [s_t,f_t]])
        
        print("Networks: f0 and 2TDDN f_t initialized")
        
        self.fisher_t.set_initial_node("s0","s_t")
        self.fisher_t.set_initial_node("f0","f_t")
        self.fisher_t.set_initial_node("d0","d_t")
        self.fisher_t.set_initial_node("g0","g_t")
        
        
        print("set up inital nodes")
        
    
   

class DN_Examples():
    
    def __init__(self):
        self.dn = BayesianDecisionNetwork()
    
    
    def firealarm(self):
        dn = XMLBIF.readDN("../../DecisionNetworks/firealarm.xml")
        dn.set_partialOrdering([dn.get_node("checkSmoke"), [], dn.get_node("call"), []])
        md = MakeDecision(dn)
        #dec = md.max_sum2(dn.get_node("checkSmoke"))
        #dn.get_node("checkSmoke").set_state(dec[1])
        
        dec = md.iterated_Optimization(dn)
        print dec
        
#        dec2 = md.max_sum2(dn.get_node("call"))
#        print dec2
        
        
    def load_xdsl(self):
        
        dn = XMLBIF.read_DN_xdsl("../../DecisionNetworks/LowSkillCPU.xdsl")
#        print dn.__len__()
#        print dn.get_node("Belief")
#        print dn.get_node("action")
#        print dn.get_node("intention")
        
        md = MakeDecision(dn)
        md.calculate_decision_and_utility(dn, dn.get_node("action"), dn.get_node("intention"))
    
        
    def PhD_example(self):
        
        bdn = BayesianDecisionNetwork()

        education = DecisionNode("education", ["do Phd", "no Phd"])
        cost = UtilityNode("cost")
        prize = DiscreteNode("prize", ["prize", "no prize"])
        income = DiscreteNode("income", ["low", "average", "high"])
        benefit = UtilityNode("benefit")
        startup = DecisionNode("startUp", ["do startUp", "no startUp"])
        costStartup = UtilityNode("costStartup")

        #bdn.add_node(startup)
        bdn.add_node(education)
        bdn.add_node(cost)
        bdn.add_node(prize)
        bdn.add_node(income)
        bdn.add_node(benefit)
        bdn.add_node(startup)
        bdn.add_node(costStartup)

        bdn.add_edge(education, cost)
        bdn.add_edge(education, prize)
        bdn.add_edge(prize, startup)
        bdn.add_edge(startup, income)
        bdn.add_edge(startup, costStartup)
        bdn.add_edge(prize, income)
        bdn.add_edge(income, benefit)

        costut=numpy.array([-50000, 0])
        cost.set_utility_table(costut, [education])

        benefitut=numpy.array([100000,200000,500000])
        benefit.set_utility_table(benefitut,[income])

        startuput=numpy.array([-200000,0])
        costStartup.set_utility_table(startuput,[startup])

        income.set_probability(0.1,[(income,"low"),(startup,"do startUp"), (prize,"no prize")])
        income.set_probability(0.2,[(income,"low"),(startup,"no startUp"), (prize,"no prize")])
        income.set_probability(0.005,[(income,"low"),(startup,"do startUp"), (prize,"prize")])
        income.set_probability(0.005,[(income,"low"),(startup,"no startUp"), (prize,"prize")])
        income.set_probability(0.5,[(income,"average"),(startup,"do startUp"), (prize,"no prize")])
        income.set_probability(0.6,[(income,"average"),(startup,"no startUp"), (prize,"no prize")])
        income.set_probability(0.005,[(income,"average"),(startup,"do startUp"), (prize,"prize")])
        income.set_probability(0.015,[(income,"average"),(startup,"no startUp"), (prize,"prize")])
        income.set_probability(0.4,[(income,"high"),(startup,"do startUp"), (prize,"no prize")])
        income.set_probability(0.2,[(income,"high"),(startup,"no startUp"), (prize,"no prize")])
        income.set_probability(0.99,[(income,"high"),(startup,"do startUp"), (prize,"prize")])
        income.set_probability(0.8,[(income,"high"),(startup,"no startUp"), (prize,"prize")])

        prize.set_probability(0.0000001,[(prize,"prize"),(education,"no Phd")])
        prize.set_probability(0.001,[(prize,"prize"),(education,"do Phd")])
        prize.set_probability(0.9999999,[(prize,"no prize"),(education,"no Phd")])
        prize.set_probability(0.999,[(prize,"no prize"),(education,"do Phd")])


        bdn.set_partialOrdering([education, [prize], startup, [income]])

#        xmlbif = XMLBIF(self.bdn, "PhD")
#        xmlbif.write("examples/PhD.xml")

#        print "make decision"
        md2 = MakeDecision(bdn)
        md2.max_sum2(education)
        print "\n \t iterated:"
        md2.arg_max_mue_D(bdn, education, [(startup, "no StartUp")])
        
        #t1 = time.clock()
        #md2.compute_optimal_strategy()
        #md2.max_sum2(education)
        #t2 = time.clock()
        
        #print "t2 - t1 = " + str(t2-t1)
        
        #print str(md2.compute_optimal_strategy())
        
#        decision = md2.max_sum2(education)
#        education.set_state(decision[1])
#        print ("E: "+ str(decision))
#        
#        start=md2.max_sum2(startup)
#        print ("S: "+str(start))
#        startup.set_state(start[1])
    
    def umbrella_example(self):
        bdn = BayesianDecisionNetwork()
        
        weather = DiscreteNode("weather", ["rain", "sunshine"])
        forecast = DiscreteNode("forecast", ["sunny", "cloudy", "rainy"])
        umbrella = DecisionNode("umbrella", ["take","leave"])
        satisfaction = UtilityNode("satisfaction")

        bdn.add_node(weather)
        bdn.add_node(forecast)
        bdn.add_node(umbrella)
        bdn.add_node(satisfaction)

        bdn.add_edge(weather, forecast)
        #bdn.add_edge(weather, umbrella)
        bdn.add_edge(forecast, umbrella)
        bdn.add_edge(weather, satisfaction)
        bdn.add_edge(umbrella, satisfaction)

        satisfaction.set_utility(20, [(weather, "sunshine"), (umbrella, "take")])
        satisfaction.set_utility(100, [(weather, "sunshine"), (umbrella, "leave")])
        satisfaction.set_utility(70, [(weather, "rain"), (umbrella, "take")])
        satisfaction.set_utility(0, [(weather, "rain"), (umbrella, "leave")])

        weather.set_probability(0.7, [(weather, "sunshine")])
        weather.set_probability(0.3, [(weather, "rain")])

        forecast.set_probability(0.7,[(forecast, "sunny"), (weather, "sunshine")])
        forecast.set_probability(0.2,[(forecast, "cloudy"), (weather, "sunshine")])
        forecast.set_probability(0.1,[(forecast, "rainy"), (weather, "sunshine")])
        forecast.set_probability(0.15,[(forecast, "sunny"), (weather, "rain")])
        forecast.set_probability(0.25,[(forecast, "cloudy"), (weather, "rain")])
        forecast.set_probability(0.6,[(forecast, "rainy"), (weather, "rain")])

        bdn.set_partialOrdering([umbrella])

        evidence=[(weather, "rain")]
        
        md = MakeDecision(bdn)
        #md.setEvidence(evidence)
        #decision = md.max_sum2(umbrella)
        
        #print decision
        
        print "Iteratd opt:"
        md.arg_max_mue_D(bdn,umbrella)
        
#        print("write to file")
#        xmlbif = XMLBIF(bdn, "TestUmbrella")
#        xmlbif.write("examples/UmbrellaDecision.xml")
#        
#        print("read from file")
#        dn = XMLBIF.readDN("examples/UmbrellaDecision.xml")
#        dn.set_partialOrdering([dn.get_node("umbrella")])
#        md2 = MakeDecision(dn)
#        decision2 = md2.max_sum2(dn.get_node("umbrella"))
#        
#        print decision2
        
    def football_example(self):
        
        football = XMLBIF.readDN("examples/football.xml")
        
        football.set_partialOrdering([football.get_node("AcceptBet")]) #[self.dn.get_node("forecast")]
        md = MakeDecision(football)
        
        #md.setEvidence([(football.get_node("Weather"), "wet")])
        #football.set_evidence([(football.get_node("Weather"), "wet")])
        
        #decision = md.calc_dec_opt(football, football.get_node("AcceptBet")) #md.max_sum2(football.get_node("AcceptBet"))
        decision = md.max_sum2(football.get_node("AcceptBet"))
        print(str(decision))
        
    def AI_example(self):
        
        ai = XMLBIF.read_DN_xdsl("/homes/mholland/repo.git/tetris/engine/src/resources/SimpleAI.xdsl")
        ai.set_partialOrdering([ai.get_node("action"), []])
        ai.set_evidence([(ai.get_node("odd_even"), "odd")])
        
        md = MakeDecision(ai)
        decision = md.max_sum2(ai.get_node("action"))
        print(str(decision))