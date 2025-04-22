# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:50:08 2024

@author: admin
"""

import numpy as np
import multiprocessing
import time
import logging
from collections import defaultdict
from model import Strategy, Agent, Builder, Searcher, Block, Bundle, Simulation
import csv
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import math

# 假设 results_dict 已经有数据
# results_dict = {(probability, replication): [(avg_bid, avg_rebate, avg_reward_builders, avg_reward_searchers), ...]}

# 我们先组织数据为 DataFrame，以便用 seaborn 作图
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(message)s', handlers=[logging.StreamHandler()])

def run_experiment(args):
    
    probability, (num_builders, num_searchers), replication = args
    
    # print(f"Starting experiment with Probability: {probability}, Builders: {num_builders}, Searchers: {num_searchers}, Replication: {replication}")  

    strategy_size_builders = 5  
    strategy_size_searchers = 10  
    num_strategies = 20  
    decay_rate = 0.5  
    block_size = 20  
    generations = 10000

    sim = Simulation(  
        num_builders=num_builders,   
        num_searchers=num_searchers,   
        strategy_size_builders=strategy_size_builders,   
        strategy_size_searchers=strategy_size_searchers,   
        num_strategies=num_strategies,  
        decay_rate=decay_rate,   
        probability=probability,   
        block_size=block_size,   
        generations=generations)  
    
    sim = sim.run()  
    avg_bid = np.mean([bid for _, bid in sim.avg_bid_history[-500:]])  
    avg_rebate = np.mean([rebate for _, rebate in sim.avg_rebate_history[-500:]])  
    avg_reward_builders = np.mean([reward for _, reward in sim.avg_reward_history_builders[-500:]])  
    avg_reward_searchers = np.mean([reward for _, reward in sim.avg_reward_history_searchers[-500:]])  
    avg_reward_proposers = np.mean(sim.mev[-500:])
    
    # logging.info(f"Completed experiment with Probability: {probability}, Builders: {num_builders}, Searchers: {num_searchers}, Replication: {replication}")    
    return (probability, (num_builders, num_searchers)), (avg_bid, avg_rebate, avg_reward_builders, avg_reward_searchers, avg_reward_proposers)  

if __name__ == '__main__':  
    
    probability_list = [i/10 for i in range(11)]  
    profile_list = [(i, 10 - i) for i in range(1, 10)]  
    replication_list = range(10)  
    
    params = [(probability, profile, replication)  
              for probability in probability_list  
              for profile in profile_list  
              for replication in replication_list]  
    
    start_time = time.time()  
    # Adjust the number of processes based on your system capabilities  
    pool = multiprocessing.Pool(processes=20)  
    
    # Use pool.imap for better progress tracking  
    results = list(pool.imap(run_experiment, params)) 

    pool.close()  
    pool.join()  
    end_time = time.time()  
    run_time = end_time - start_time  
    
    print(f"Total runtime: {run_time:.2f} seconds")  
    
    # Initialize a dictionary to store results  
    results_dict = defaultdict(list)  
    
    for (params, metrics) in results:  
        results_dict[params].append(metrics)  
    
    # Construct DataFrame  
    data = []  
    for (probability, (num_builders, num_searchers)), metrics_list in results_dict.items():  
        for (avg_bid, avg_rebate, avg_reward_builders, avg_reward_searchers, avg_reward_proposers) in metrics_list:  
            data.append({  
                'Probability': probability,  
                'Num Builders': num_builders,  
                'Num Searchers': num_searchers,  
                'Avg Bid': avg_bid,  
                'Avg Rebate': avg_rebate,  
                'Avg Reward Builders': avg_reward_builders,  
                'Avg Reward Searchers': avg_reward_searchers,
                'Avg Reward Proposers': avg_reward_proposers
            })  

    # Create the DataFrame  
    df = pd.DataFrame(data)  

    # Optionally, save the results to a CSV file  
    df.to_csv('experiment_results_10000_4.csv', index=False)    

    # Now calculate mean values for each group of (Probability, Num Builders, Num Searchers)  
    df_mean = df.groupby(['Probability', 'Num Builders', 'Num Searchers'], as_index=False).mean()  

    # Optionally, save the results to a CSV file  
    df_mean.to_csv('experiment_results_mean_10000_4.csv', index=False)    


  