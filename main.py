from datetime import datetime
import os

import reinforce as rf
import functions as func
from reinforcement_environments.graph_env import *
from models.fully_connected import FullyConnected
from models.fully_connected_graph import FullyConnectedGraph

import numpy as np
import statistics

m_policy = FullyConnectedGraph(input_features=6, hidden_features=128, output_features=5)
m_value = FullyConnected(input_features=6, hidden_features=128, output_features=1)

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
    policy_func = func.PiApprox(env.num_actions(), 1e-5, m_policy)
    value_func = func.VApprox(5e-3, m_value)
    reinforce = rf.Reinforce(env, 0.95, policy_func, value_func)

    lastfive = []
    
    #state validity debug
    

    for idx in range(200):
        returns = reinforce.episode(state_dictionary, in_training=True)
        seq_len = reinforce.seq_len
        print("Iteration: " + str(idx) + 
              "\n[num_nodes, depth_of_graph]: " + str(returns) + 
              "\nSequence length: " + str(seq_len) +
              "\n")
        #reinforce.replay()
    result = AbcReturn(reinforce.episode(state_dictionary, in_training=False))
    result_name = "./results/" + benchmark + ".csv"
    #lastfive.sort(key=lambda x : x.level)
    with open(result_name, 'a') as and_log:
        line = ""
        line += str(result.num_nodes)
        line += ","
        line += str(result.level)
        line += "\n"
        and_log.write(line)
    rewards = reinforce.sum_rewards

from tqdm import tqdm
from itertools import product

def test_var(filename, benchmark):
    env = GraphEnv(filename)
    heu_returns = []
    
    value_range = range(5)  # 0 to 4
    list_length = 5

    # Generate all variations
    all_variations = product(value_range, repeat=list_length)

    # Iterate over all variations with a progress bar
    for variation in tqdm(all_variations, total=5**list_length):
        var = list(variation)
        env.heuristic(var)
        heu_returns.append([env.returns(), var])
        env.reset()
    
    heu_returns.sort()
    counter = 0
    line = ""
    for res in heu_returns:
        if(counter == 10):
            break
        print(res)
        line += str(res[0][0])
        line += ","
        line += str(res[0][1])
        line += ","
        line += str(res[1])
        line += "\n"
    line += ".\n"
 
    with open("results/variations.csv", 'a') as and_log:
        and_log.write(line)     

def interactive_opt(filename, benchmark):
    env = GraphEnv(filename)

    for move in range(10):
        action_idx = int(input("Enter action: "))
        if action_idx == 0:
            env._abc.balance(l=False)
        elif action_idx == 1:
            env._abc.rewrite(l=False)
        elif action_idx == 2:
            env._abc.refactor(l=False)
        elif action_idx == 3:
            env._abc.rewrite(l=False, z=True)
        elif action_idx == 4:
            env._abc.refactor(l=False, z=True)
        env._prev_stats = env._curr_stats
        env._curr_stats = env._abc.aigStats()

        print(env.returns())
    env.reset()
    
if __name__ == "__main__":
    state_dictionary = {}
    test_reinforce("./benchmarks/C432.blif", "dalu")
#    test_reinforce("./benchmarks/C880.blif", "dalu")
#    test_reinforce("./benchmarks/C499.blif", "dalu")
#    test_reinforce("./benchmarks/C432.blif", "dalu")

#    interactive_opt("./benchmarks/C432.blif", "dalu")
#    interactive_opt("./benchmarks/C880.blif", "dalu")
#    interactive_opt("./benchmarks/C7552.blif", "dalu")
#    interactive_opt("./benchmarks/C3540.blif", "dalu")
#    interactive_opt("./benchmarks/C2670.blif", "dalu")
#    interactive_opt("./benchmarks/C5315.blif", "dalu")
#    interactive_opt("./benchmarks/C6288.blif", "dalu")
#    interactive_opt("./benchmarks/C6288.blif", "dalu")
#    interactive_opt("./benchmarks/C6288.blif", "dalu")
    print(sorted(state_dictionary.values()))
