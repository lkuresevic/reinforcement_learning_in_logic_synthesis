from datetime import datetime
import os

import reinforce as rf
from graph_env import graph_env

import numpy as np
import statistics


class AbcReturn:
    #returns = [number of nodes, logic levels]
    def __init__(self, returns):
        self.num_nodes = float(returns[0])
        self.level = float(returns[1])
        
    def less_than(self, other):
        if (int(self.level) == int(other.level)):
            return self.num_nodes < other.num_nodes
        else:
            return self.level < other.level

    def equal_to(self, other):
        return int(self.level) == int(other.level) and int(self.num_nodes) == int(self.num_nodes)

def test_reinforce(filename, benchmark):
    #display current time
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("Time: ", date_time)
    
    #initialize the reinforcement learning environment
    env = graph_env(filename)
        
    #vApprox = Linear(env.dimState(), env.numActions())
    val_approx = rf.PiApprox(env.state_dimensions(), env.num_actions(), 8e-4, rf.FcModelGraph)
    baseline = rf.Baseline(0)
    vbaseline = rf.BaselineVApprox(env.dimState(), 3e-3, rf.FcModel)
    reinforce = rf.Reinforce(env, 0.9, vApprox, vbaseline)

    lastfive = []

    for idx in range(200):
        returns = reinforce.episode(phaseTrain=True)
        seqLen = reinforce.lenSeq
        line = "iter " + str(idx) + " returns "+ str(returns) + " seq Length " + str(seqLen) + "\n"
        if idx >= 195:
            lastfive.append(AbcReturn(returns))
        print(line)
        #reinforce.replay()
    resultName = "./results/" + ben + ".csv"
    #lastfive.sort(key=lambda x : x.level)
    lastfive = sorted(lastfive)
    with open(resultName, 'a') as andLog:
        line = ""
        line += str(lastfive[0].numNodes)
        line += " "
        line += str(lastfive[0].level)
        line += "\n"
        andLog.write(line)
    rewards = reinforce.sumRewards
