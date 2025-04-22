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

# 假设 results_dict 已经有数据
# results_dict = {(probability, replication): [(avg_bid, avg_rebate, avg_reward_builders, avg_reward_searchers), ...]}

# 我们先组织数据为 DataFrame，以便用 seaborn 作图
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(message)s', handlers=[logging.StreamHandler()])

def run_experiment(args):
    
    probability, replication = args
    
    num_builders = 3
    num_searchers = 7
    strategy_size_builders = 5
    strategy_size_searchers = 10
    num_strategies = 20
    decay_rate = 0.5
    block_size = 20
    generations = 10000

    sim = Simulation(num_builders=num_builders, num_searchers=num_searchers, strategy_size_builders=strategy_size_builders, strategy_size_searchers=strategy_size_searchers, num_strategies=num_strategies, decay_rate=decay_rate, probability=probability, block_size=block_size, generations=generations)
    sim = sim.run()
    avg_bid = np.mean([bid for _,bid in sim.avg_bid_history[-500:]])
    avg_rebate = np.mean([rebate for _,rebate in sim.avg_rebate_history[-500:]])
    avg_reward_builders = np.mean([reward for _,reward in sim.avg_reward_history_builders[-500:]])
    avg_reward_searchers = np.mean([reward for _,reward in sim.avg_reward_history_searchers[-500:]])
    avg_reward_proposers = np.mean(sim.mev[-500:])
    
    return probability, (avg_bid, avg_rebate, avg_reward_builders, avg_reward_searchers, avg_reward_proposers)
    

if __name__ == '__main__':
    
    probability_list = [x/10 for x in range(11)]
    replication_list = range(10)
    
    params = [(probability,replication)
              for probability in probability_list
              for replication in replication_list]
    
    
    start_time = time.time()
    # Adjust the number of processes based on your system capabilities
    pool = multiprocessing.Pool(processes=5)
    # Use pool.imap for better progress tracking  
    results = list(pool.imap(run_experiment, params))  
    pool.close()
    pool.join()
    end_time = time.time()
    run_time = end_time - start_time
    
    print(f"Total runtime: {run_time:.2f} seconds")
    
    # Additional code to process and visualize results goes here
    # 初始化一个字典来存储每组参数的所有结果
    results_dict = defaultdict(list)
    
    for (params, metrics) in results:
        results_dict[params].append(metrics)
    
    # params 是 (block_size, N, opportunities)
    # metrics 是 (bribe_ratio, mev)
    


    # 构造 DataFrame  
    data = []  
    for probability, metrics_list in results_dict.items():  
        for (avg_bid, avg_rebate, avg_reward_builders, avg_reward_searchers, avg_reward_proposers) in metrics_list:  
            data.append({  
                'Probability': probability,  
                'Average Bid Ratio': avg_bid,  
                'Average Rebate Ratio': avg_rebate,  
                'Average Reward for Builders': avg_reward_builders,  
                'Average Reward for Searchers': avg_reward_searchers,  # 修复拼写错误  
                'Average Reward for Proposers': avg_reward_proposers  
            })  
    
    df = pd.DataFrame(data)  
    
    # 在保存图像前确保 figures 目录存在  
    if not os.path.exists('figures'):  
        os.makedirs('figures')  
    
    # 设置 Seaborn 风格  
    # sns.set(style="whitegrid")  
    
    # 定义要绘制的度量  
    metrics = ['Average Bid Ratio', 'Average Rebate Ratio',   
               'Average Reward for Builders', 'Average Reward for Searchers',   
               'Average Reward for Proposers']  
    
    
    sns.set(style="white", font="Arial", font_scale=1.2)  

    for metric in metrics:  
        plt.figure(figsize=(10, 8))  
    
        # 1. 抖动点图（swarm），点加透明，描边，控制大小  
        sns.swarmplot(  
            x='Probability', y=metric, data=df,   
            size=6, palette="Set3",   
            edgecolor="k", alpha=0.7, linewidth=0.8)  
    
        # 2. 叠加箱线图，细线边框  
        sns.boxplot(  
            x='Probability', y=metric, data=df,  
            width=0.25, showcaps=False, boxprops={'facecolor':'none', 'edgecolor':'gray', 'linewidth':2},  
            whiskerprops={'linewidth':2, 'color':'gray'}, medianprops={'color':'black', 'linewidth':2},  
            showfliers=False, zorder=10)  
    
        plt.xlabel('Probability of Conflicts', fontsize=24, fontweight='bold')  
        plt.ylabel(metric, fontsize=24, fontweight='bold')  
        plt.tick_params(axis='both', which='major', labelsize=20)  
        sns.despine()  # 去掉顶部和右侧框线，美观  
    
        plt.tight_layout()  
        plt.savefig(os.path.join('figures', f'{metric.replace(" ", "_").lower()}_swarm.png'),  
                    dpi=300, bbox_inches='tight')  
        plt.show()  
        plt.close()  
        

    # Step 1: 变换 DataFrame 结构，融合 reward 到同一列  
    df_melted = pd.melt(  
        df,  
        id_vars=['Probability'],  
        value_vars=[  
            'Average Reward for Builders',  
            'Average Reward for Searchers',  
            'Average Reward for Proposers'  
        ],  
        var_name='Role',  
        value_name='Average Reward'  
    )  
    
    # Step 2: 画融合图  
    plt.figure(figsize=(12, 8))  
    sns.swarmplot(  
        x='Probability', y='Average Reward', hue='Role',   
        data=df_melted, size=6, palette="Set2", edgecolor="k", alpha=0.7, linewidth=0.8, dodge=True  
    )  
    
    sns.boxplot(  
        x='Probability', y='Average Reward', hue='Role',   
        data=df_melted, width=0.25, showcaps=False,   
        boxprops={'facecolor':'none', 'edgecolor':'gray', 'linewidth':2},  
        whiskerprops={'linewidth':2, 'color':'gray'},   
        medianprops={'color':'black', 'linewidth':2},  
        showfliers=False, zorder=10, dodge=True  
    )  
    
    plt.xlabel('Probability of Conflicts', fontsize=24, fontweight='bold')  
    plt.ylabel('Average Reward', fontsize=24, fontweight='bold')  
    plt.tick_params(axis='both', which='major', labelsize=20)  
    sns.despine()  
    plt.legend(title="Role", fontsize=18, title_fontsize=20)  
    plt.tight_layout()  
    plt.savefig(os.path.join('figures', 'average_reward_roles_swarm.png'), dpi=300, bbox_inches='tight')  
    plt.show()  
    plt.close()  
    
    
    
    # # 循环处理每个度量，创建独立的图并保存为 PNG  
    # for metric in metrics:  
    #     plt.figure(figsize=(8, 6))  # 创建新图形  
    #     palette = ["#4878CF"] * len(df["Probability"].unique())  
    #     sns.boxplot(x='Probability', y=metric, data=df, palette=palette, width=0.5, showmeans=True,  
    #                 meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": "8"})  
    
    #     plt.xlabel('Probability of Conflicts')  
    #     plt.ylabel(metric)  
        
    #     # 保存为 PNG 文件  
    #     plt.savefig(os.path.join('figures', f'{metric.replace(" ", "_").lower()}.png'),dpi=300)  # 替换空格并小写文件名  
    #     plt.show()
    #     plt.close()  # 关闭当前 figure，释放内存  
    
    
    # for metric in metrics:  
    #     plt.figure(figsize=(8, 6))  
    #     sns.swarmplot(x='Probability', y=metric, data=df, size=5, palette="Set2", linewidth=0.7)  
    #     sns.boxplot(x='Probability', y=metric, data=df,   
    #                 width=0.3, showcaps=False, boxprops={'facecolor':'none'}, showfliers=False)  
    #     plt.xlabel('Probability of Conflicts', fontsize=16)  
    #     plt.ylabel(metric, fontsize=16)  
    #     plt.tight_layout()  
    #     plt.savefig(os.path.join('figures', f'{metric.replace(" ", "_").lower()}_swarm.png'), dpi=300, bbox_inches='tight')  
    #     plt.show()  
    #     plt.close()  