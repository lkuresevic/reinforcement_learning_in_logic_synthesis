from datetime import datetime
import os

import reinforce as rf
from reinforcement_environments.graph_env import *

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
    env = GraphEnv(filename)
        
    #vApprox = Linear(env.dimState(), ennumActions())
    val_approx = rf.PiApprox(env.state_dimensions(), env.num_actions(), 8e-4, rf.FullyConnectedGraph)
    baseline = rf.Baseline(0)
    val_baseline = rf.BaselineVApprox(env.state_dimensions(), 3e-3, rf.FullyConnected)
    reinforce = rf.Reinforce(env, 0.9, val_approx, val_baseline)

    lastfive = []

    for idx in range(200):
        returns = reinforce.episode(in_training=True)
        seq_len = reinforce.seq_len
        line = "iter " + str(idx) + " returns "+ str(returns) + " seq Length " + str(seq_len) + "\n"
        if idx >= 195:
            lastfive.append(AbcReturn(returns))
        print(line)
        #reinforce.replay()
    result_name = "./results/" + benchmark + ".csv"
    #lastfive.sort(key=lambda x : x.level)
    lastfive = sorted(lastfive)
    with open(result_name, 'a') as and_log:
        line = ""
        line += str(lastfive[0].num_nodes)
        line += " "
        line += str(lastfive[0].level)
        line += "\n"
        and_log.write(line)
    rewards = reinforce.sum_rewards

if __name__ == "__main__":
    test_reinforce("../vtr_test/vtr_verilog_to_routing/vtr_flow/benchmarks/blif/alu4.blif", "dalu")
