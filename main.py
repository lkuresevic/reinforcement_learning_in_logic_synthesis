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
        
    def __lt__(self, other):
        if (int(self.level) == int(other.level)):
            return self.num_nodes < other.num_nodes
        else:
            return self.level < other.level

    def __eq__(self, other):
        return int(self.level) == int(other.level) and int(self.num_nodes) == int(self.num_nodes)

def test_reinforce(filename, benchmark):
    #display current time
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("Time: ", date_time)
    
    #initialize the reinforcement learning environment
    env = GraphEnv(filename)
        
    #vApprox = Linear(env.dimState(), ennumActions())
    val_approx = rf.PiApprox(env.state_dimensions(), env.num_actions(), 1e-5, rf.FullyConnectedGraph)
    baseline = rf.Baseline(0)
    val_baseline = rf.BaselineVApprox(env.state_dimensions(), 1e-5, rf.FullyConnected)
    reinforce = rf.Reinforce(env, 0.8, val_approx, val_baseline)

    lastfive = []

    for idx in range(200):
        returns = reinforce.episode(in_training=True)
        seq_len = reinforce.seq_len
        line = "Iteration: " + str(idx) + "\n[num_nodes, depth_of_graph]: "+ str(returns) + "\nSequence length: " + str(seq_len) + "\n"
        if idx >= 195:
            lastfive.append(AbcReturn(returns))
        print(line)
    reinforce.replay()
    result_name = "./results/" + benchmark + ".csv"
    #lastfive.sort(key=lambda x : x.level)
    lastfive = sorted(lastfive)
    with open(result_name, 'a') as and_log:
        line = ""
        line += str(lastfive[0].num_nodes)
        line += ","
        line += str(lastfive[0].level)
        line += "\n"
        and_log.write(line)
    rewards = reinforce.sum_rewards

if __name__ == "__main__":
    test_reinforce("./benchmarks/C17.blif", "dalu")
    test_reinforce("./benchmarks/C6288.blif", "dalu")
    test_reinforce("./benchmarks/C2670.blif", "dalu")
