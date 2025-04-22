# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:35:15 2024

@author: admin
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 00:49:21 2024

@author: admin
"""

import random
import numpy as np
import math
import matplotlib.pyplot as plt  
import os  
import seaborn as sns  

# Strategy 类
class Strategy:  
    def __init__(self, chromosome, fitness): 
        self.chromosome = chromosome
        self.fitness = fitness  
        
    def crossover(self, other_strategy):  
        # 实现交叉逻辑，假设简单的单点交叉  
        crossover_point = np.random.randint(1, len(self.chromosome))  
        child_chromosome = self.chromosome[:crossover_point] + other_strategy.chromosome[crossover_point:]
        child_fitness = 0.5 * (self.fitness + other_strategy.fitness)
        return Strategy(child_chromosome, child_fitness)  

    def mutate(self, mutation_rate=0.01):  
        # 实现变异逻辑，假设简单的随机变异  
        for i in range(len(self.chromosome)):  
            if np.random.rand() < mutation_rate:  
                self.chromosome[i] = 1 - self.chromosome[i]  # 假设策略是二进制的  


# 基础Agent类
class Agent:
    def __init__(self, strategy_size, num_strategies, decay_rate, index, evolve_probability=0.01, temperature=10):
        self.index = index
        self.strategy_size = strategy_size
        self.num_strategies = num_strategies
        self.strategies = [self.random_strategy() for _ in range(self.num_strategies)]
        self.fitness_memory = np.zeros(num_strategies)
        self.decay_rate = decay_rate  # 遗忘参数
        self.value = 0
        self.reward = 0
        self.reward_history = []
        self.strategy_chosen = 0
        self.evolve_probability = evolve_probability
        self.temperature = temperature

    def random_strategy(self):
        chromosome = [random.randint(0, 1) for _ in range(self.strategy_size)]
        strategy = Strategy(chromosome, 0)
        return strategy
    
    
    def select_strategy(self, strategies): 
        temperature = self.temperature
        # 计算每个策略的指数适应度，使用温度参数调节  
        exp_fitness = [math.exp(strategy.fitness / temperature) for strategy in strategies]  
        
        # 计算总的指数适应度  
        total_exp_fitness = sum(exp_fitness)  
        
        # 如果总的指数适应度为 0，随机选择一个策略  
        if total_exp_fitness == 0:  
            strategy_index = random.choice(range(len(strategies)))  
            self.strategy_chosen = strategy_index  
            return strategies[strategy_index]  
        
        # 归一化指数适应度以得到选择概率  
        probs = [ef / total_exp_fitness for ef in exp_fitness]  
        
        # 根据概率选择策略  
        strategy_index = random.choices(range(len(strategies)), weights=probs, k=1)[0]  
        self.strategy_chosen = strategy_index  
        return strategies[strategy_index]

    def update_fitness(self):  
        # 获取当前选择的策略索引  
        index = self.strategy_chosen  
        # 更新策略的适应度  
        strategy = self.strategies[index]  
        strategy.fitness = (  
            strategy.fitness * self.decay_rate + self.reward * (1 - self.decay_rate)  
        )  
    
    def evolve_strategies(self, elimination_ratio=0.5):
        # 以一定概率决定是否进行策略演化  
        if random.random() > self.evolve_probability:  
            return  # 不进行演化  
        # Step 1: 淘汰低适应度策略  
        num_to_eliminate = int(self.num_strategies * elimination_ratio)  
        # 按适应度排序，适应度低的在前  
        self.strategies.sort(key=lambda strategy: strategy.fitness)  
        # 保留适应度高的策略  
        surviving_strategies = self.strategies[num_to_eliminate:]  

        # Step 2: 选择父代并生成子代  
        new_strategies = surviving_strategies[:]  # 保留父代  
        while len(new_strategies) < self.num_strategies:  
            parent1 = self.select_strategy(surviving_strategies)  
            parent2 = self.select_strategy(surviving_strategies)  
            child = parent1.crossover(parent2)  
            child.mutate() 
            new_strategies.append(child)  

        # 更新策略池  
        self.strategies = new_strategies
    
    def get_value(self, rate):  
        """生成一个服从指数分布的随机数"""  
        value = random.expovariate(rate)
        self.value = value
        return value

    

                
# Builder
class Builder(Agent):
    def __init__(self, strategy_size, num_strategies, decay_rate, index, temperature):
        super().__init__(strategy_size, num_strategies, decay_rate, index, temperature=temperature)
        self.rebate = 0
        self.rebate_history = []
        self.pending_pool = []
        self.block = None

    def perform_action(self, strategy):  
        assert isinstance(strategy, Strategy), "Expected a Strategy object"  
        decimal_value = int(''.join(map(str, strategy.chromosome)), 2)  
        rebate = decimal_value / (2 ** self.strategy_size - 1)  
        self.rebate = rebate  
        self.rebate_history.append(rebate)  
        # print(f"Executing Builder strategy with rebate {rebate:.3f}")  
        return rebate 
    
    def build_block(self, block_size, matrix):
        selected_bundles = []
        while len(selected_bundles) < block_size and self.pending_pool:
            
            # sort bundles in pending pool
            self.pending_pool.sort(key=lambda x: (x.value <= 0, -x.bid))
            
            # pop the top bundle 
            top_bundle = self.pending_pool.pop(0)
            
            # break if no bundle avalible
            if top_bundle.value <= 0: # the remaining bundles have no profit potential
                break
            
            # merge the top bundle
            selected_bundles.append(top_bundle)
            
            # update the potential value of rest bundles
            for bundle in self.pending_pool:
                bundle.value = bundle.value + matrix[bundle.sender, top_bundle.sender] * bundle.value
        return selected_bundles

# Searcher
class Searcher(Agent):
    def __init__(self, strategy_size, num_strategies, decay_rate, index, temperature):
        super().__init__(strategy_size, num_strategies, decay_rate, index, temperature=temperature)
        self.bid_ratio = None
        self.bid_history = []
        self.para_history = []


    def perform_action(self, strategy, rebate_list):  
        assert isinstance(strategy, Strategy), "Expected a Strategy object"  
        
        # 确保染色体长度可以被4整除  
        assert len(strategy.chromosome) % 2 == 0, "Chromosome length must be divisible by 4"  
        
        bid_ratio = np.zeros(len(rebate_list))
        quarter_length = len(strategy.chromosome) // 2  
        decimal_values = []  
        parameters = np.zeros(2)
        
        for i in range(2):  
            # 获取染色体的每个四分之一部分  
            part = strategy.chromosome[i * quarter_length: (i + 1) * quarter_length]  
            # 将二进制数转化为十进制  
            decimal_value = int(''.join(map(str, part)), 2)  
            decimal_values.append(decimal_value)
            
        parameters[0] = 1 + (decimal_values[0]/ (2 ** quarter_length - 1))*(5-1)  
        parameters[1] = 0 + (decimal_values[1]/ (2 ** quarter_length - 1))*(4-0)
        # print(f"Executing Searcher strategy with decimal values {parameters}")  
        
        # 将这些十进制数的某种处理或使用逻辑  
        # 这里用它们来计算平均折扣作为示例  
        for i in range(len(rebate_list)):
            bid_ratio[i] = ((1/(1+parameters[0] ** (-1 * rebate_list[i]))) ** parameters[1]) 
        self.bid_ratio = bid_ratio  
        self.bid_history.append(bid_ratio)  
        
        
        # print(f"Bid from strategy: {bid_ratio}")  
        return bid_ratio  

        
# Block
class Block:
    def __init__(self, builder, bundles, mev):
        self.builder = builder
        self.bundles = bundles
        self.mev = mev
        
# Bundle
class Bundle:
    def __init__(self, value, bid_ratio, sender, receiver):
        self.value = value 
        self.bid_ratio = bid_ratio
        self.sender = sender
        self.receiver = receiver
        
    @property
    def bid(self):
        bid = self.value * self.bid_ratio
        return bid


# 模拟系统
class Simulation:
    def __init__(self, num_builders=10, num_searchers=10, strategy_size_builders=5, strategy_size_searchers=10, num_strategies=20, decay_rate=0.5, probability=0.5, block_size=20, generations=500, temperature=2):
        self.generations = generations
        self.builders = [Builder(strategy_size_builders, num_strategies, decay_rate, index=i, temperature=temperature) for i in range(num_builders)]
        self.searchers = [Searcher(strategy_size_searchers, num_strategies, decay_rate, index=i+num_builders, temperature=temperature) for i in range(num_searchers)]
        self.num_builders = num_builders
        self.num_searchers = num_searchers
        self.scale = num_builders+num_searchers
        self.probability = probability
        self.block_size = block_size
        self.rebate_history = [[] for _ in range(num_builders)]  
        self.bid_history = [[] for _ in range(num_searchers)]  
        self.avg_rebate_history = []  
        self.avg_bid_history = []  
        self.reward_history_builders = [[] for _ in range(num_builders)]
        self.reward_history_searchers = [[] for _ in range(num_searchers)]
        self.avg_reward_history_builders = []
        self.avg_reward_history_searchers = []

        self.para_usage_count = [np.zeros((generations, 2 ** (strategy_size_searchers // 2))), np.zeros((generations, 2 ** (strategy_size_searchers // 2)))]
        self.rebate_usage_count = np.zeros((generations, 2 ** strategy_size_builders))
        
        self.rebate_cov = [np.zeros(generations) for _ in range(num_builders)]
        self.para_cov = [[np.zeros(generations) for _ in range(num_searchers)] for _ in range(2)]
        self.mev = np.zeros(generations)
         
    def generate_symmetric_matrix(self, n, p):  
        # 初始化 n x n 矩阵为 0  
        matrix = np.zeros((n, n), dtype=int)  
        
        # 填充上三角矩阵（不包括对角线）  
        for i in range(n):  
            for j in range(i + 1, n):  
                # 根据概率 p 生成 -1 或 0  
                matrix[i, j] = -1  if np.random.rand() < p else 0  
                # 确保矩阵对称  
                matrix[j, i] = matrix[i, j]  
                
        return matrix    

    
    def calculate_cov(self, values):  
        mean = np.mean(values)  
        std = np.std(values)  
        return std / mean if mean != 0 else 0  


    def run(self):
        
        generations = self.generations

        
        for generation in range(generations):
            # print(f"Generation {generation}:")
            if generation % 1000 == 0:
                print(generation)
            
            # 策略适应度统计
            
            for builder in self.builders:
                # 遍历builders的策略库
                rebate_list = []
                decimal_value_list = []
                for strategy in builder.strategies:
                    
                    quarter_length = len(strategy.chromosome)
                    decimal_value = int(''.join(map(str, strategy.chromosome)), 2)  
                    rebate = decimal_value / (2 ** quarter_length - 1)  
                    rebate_list.append(rebate)
                    decimal_value_list.append(decimal_value)
                    self.rebate_usage_count[generation, decimal_value] += strategy.fitness
                self.rebate_cov[builder.index][generation] = self.calculate_cov(decimal_value_list)  
            
            for searcher in self.searchers:  
                # 遍历每个搜索者的策略库  
                decimal_value_list =  [[] for _ in range(2)]
                para_list = [[] for _ in range(2)]
                for strategy in searcher.strategies:  
                    
                    quarter_length = len(strategy.chromosome) // 2  
                    decimal_values = []  
                    parameters = np.zeros(2)
                    
                    # Convert the chromosome to a decimal number  
                    for i in range(2):  
                        # 获取染色体的每个四分之一部分  
                        part = strategy.chromosome[i * quarter_length: (i + 1) * quarter_length]  
                        # 将二进制数转化为十进制  
                        decimal_value = int(''.join(map(str, part)), 2)  
                        decimal_values.append(decimal_value)
                        decimal_value_list[i].append(decimal_value)
                        
                    parameters[0] = 1 + (decimal_values[0]/ (2 ** quarter_length - 1))*(5-1)  
                    parameters[1] = 0 + (decimal_values[1]/ (2 ** quarter_length - 1))*(4-0)
                    
                                            
                    # Update usage count  
                    for i in range(2):
                        
                        para_list[i].append(parameters[i])
                        self.para_usage_count[i][generation, decimal_values[i]] += strategy.fitness 
                
                for i in range(2):
                    self.para_cov[i][searcher.index-self.num_builders][generation] = self.calculate_cov(decimal_value_list[i]) 
            

            # generate value randomly
            for agent in self.builders + self.searchers:
                agent.get_value(10)
                # print("value:",agent.index,agent.value)
            
            # generate matrix randomly
            matrix = self.generate_symmetric_matrix(self.scale, self.probability)
            
            # builders act
            rebate_list=[]
            for i,agent in enumerate(self.builders):
                # choose strategy
                strategy = agent.select_strategy(agent.strategies)
                # action
                rebate = agent.perform_action(strategy)
                rebate_list.append(rebate)
                self.rebate_history[i].append((generation, rebate)) 
                # print(strategy.chromosome,strategy.fitness)
                
        
            # searchers act
            for i, agent in enumerate(self.searchers):
                # choose strategy
                strategy = agent.select_strategy(agent.strategies)
                # action
                bid_ratio = agent.perform_action(strategy, rebate_list)

                self.bid_history[i].append((generation, bid_ratio))
                
            # Calculate average rebate and bid  
            if generation > 50:
                avg_rebate = np.mean(rebate_list)  
                avg_bid = np.mean([bid for bid_array in self.bid_history for g, bid in bid_array if g == generation]) 
                self.avg_rebate_history.append((generation, avg_rebate))  
                self.avg_bid_history.append((generation, avg_bid)) 

                
            
                
            # block build process
            
            # searchers send bundles
            for builder in self.builders:
                builder.pending_pool = []
                bundle = Bundle(builder.value, 1, builder.index, builder.index)
                builder.pending_pool.append(bundle)
                for searcher in self.searchers:
                    bundle = Bundle(searcher.value, searcher.bid_ratio[builder.index], searcher.index, builder.index)
                    builder.pending_pool.append(bundle)

            # builders build blocks
            blocks = []
            for agent in self.builders:
                block_bundles = agent.build_block(self.block_size, matrix)
     
                block_mev = sum(bundle.bid for bundle in block_bundles)

                block = Block(agent.index, block_bundles, block_mev)

                agent.block = block
   
                blocks.append(block)
            
            # block building auction
            
            # 按照 block.mev 降序排序  
            sorted_blocks = sorted(blocks, key=lambda block: block.mev, reverse=True)  
            
            # for block in sorted_blocks:  
                # 打印每个 block 的 mev 和其包含的每个 bundle 的 bid  
                # bundle_bids = [bundle.bid for bundle in block.bundles]  
                # print(f"Block MEV: {block.mev}, Builder: {block.builder}, Bundle Bids: {bundle_bids}")
            # 选出第一名和第二名  
            first_place = sorted_blocks[0] if len(sorted_blocks) > 0 else None  
            winner = first_place.builder
            if len(sorted_blocks) == 1:
                builder_surplus = first_place.mev
                self.mev[generation] = 0
            else:
                second_place = sorted_blocks[1] if len(sorted_blocks) > 1 else None 
                builder_surplus = first_place.mev - second_place.mev
                self.mev[generation] = second_place.mev
            
 
            
            # results 
            
            for agent in self.builders:
                agent.reward = 0
                
            self.builders[winner].reward = (1 - agent.rebate) * builder_surplus

            
            for agent in self.searchers:
                agent.reward = 0
                
            total_bid = sum(bundle.bid for bundle in first_place.bundles if bundle.sender != bundle.receiver)
            
            for bundle in first_place.bundles:
                # print("bundle value",bundle.value, bundle.bid_ratio, bundle.bid)
                if bundle.sender != bundle.receiver:          
                    
                    self.searchers[bundle.sender-len(self.builders)].reward = bundle.value * (1- bundle.bid_ratio) + self.builders[bundle.receiver].rebate * (bundle.bid / total_bid) * builder_surplus
            
            for i, agent in enumerate(self.builders):
                self.reward_history_builders[i].append((generation, agent.reward))
                # print("reward",agent.index,agent.reward)
            
            for i, agent in enumerate(self.searchers):
                self.reward_history_searchers[i].append((generation, agent.reward))
                # print("reward",agent.index,agent.reward)
                
            # Calculate average reward for builders and saerchers
            if generation > 50:
                avg_reward_builders = np.mean([reward for reward_array in self.reward_history_builders for g, reward in reward_array if g == generation])
                avg_reward_searchers = np.mean([reward for reward_array in self.reward_history_searchers for g, reward in reward_array if g == generation])
                self.avg_reward_history_builders.append((generation, avg_reward_builders))
                self.avg_reward_history_searchers.append((generation, avg_reward_searchers))



                
            # Calculate average reward 
            
            # avg_reward_history_builders
            #print("here",self.reward_history_builders)
            #print("here 2",self.reward_history_searchers)
                
                
            # evolve
            for agent in self.builders:
                agent.update_fitness()
                agent.evolve_strategies()
                
            for agent in self.searchers:
                agent.update_fitness()
                agent.evolve_strategies()
            
        return self            
           
                
    def plot_results(self):  
        # Plot rebate history for builders as scatter plot  
        plt.figure(figsize=(12, 6))  
        for i, history in enumerate(self.rebate_history):  
            generations, rebates = zip(*history)  
            plt.scatter(generations, rebates, label=f'Builder {i} Rebate', alpha=0.6)  
        plt.title('Rebate Evolution')  
        plt.xlabel('Generation')  
        plt.ylabel('Rebate')  
        plt.legend()  
        plt.show()  

        # Plot bid history for searchers as scatter plot  
        plt.figure(figsize=(12, 6))  
        for i, history in enumerate(self.bid_history):  
            generations, bids = zip(*[(gen, bid) for gen, bid_array in history for bid in bid_array])  
            plt.scatter(generations, bids, label=f'Searcher {i} Bid', alpha=0.6)  
        plt.title('Bid Evolution')  
        plt.xlabel('Generation')  
        plt.ylabel('Bid')  
        plt.legend()  
        plt.show()  
        
    
    def plot_results_avg_reward(self):  
        # Plot average rebate and its moving average  
        generations, avg_reward_builders = zip(*self.avg_reward_history_builders)  
        
        plt.figure(figsize=(12, 6))  
        plt.plot(generations, avg_reward_builders, label='Average Rebate', alpha=0.6)  

        plt.plot(generations[:len(avg_reward_builders) - 49], moving_average(avg_reward_builders), label='Moving Average Reward', color='red')  
        plt.title('Average Reward Evolution')  
        plt.xlabel('Generation')  
        plt.ylabel('Average Reward')  
        plt.legend()  
        plt.show()  
    
        # Plot average bid and its moving average  
        generations, avg_reward_searchers = zip(*self.avg_reward_history_searchers)  
        plt.figure(figsize=(12, 6))  
        plt.plot(generations, avg_reward_searchers, label='Average Rebate', alpha=0.6)    
        plt.plot(generations[:len(avg_reward_searchers) - 49], moving_average(avg_reward_searchers), label='Moving Average Reward', color='red')  
        plt.title('Average Reward Evolution')  
        plt.xlabel('Generation')  
        plt.ylabel('Average Reward')  
        plt.legend()  
        plt.show()  

def plot_strategy_usage_heatmap(para_usage_count, rebate_usage_count, generations):  
        # Define the ranges for the first and last 500 generations  
        ranges = [(0, generations)]  
        
        for param_index in range(2):  
            for start, end in ranges:  
                plt.figure(figsize=(14, 8))  
                sns.heatmap(para_usage_count[param_index][start:end, :].T, cmap='viridis', cbar_kws={'label': 'Usage Count'})  
                plt.title(f'Strategy Fitness for Parameter {param_index} (Generations {start} to {end})')  
                plt.xlabel('Generations')  
                plt.ylabel(f'Parameter {param_index} Values')  
                
                # Convert decimal values to parameter values for y-axis labels  
                yticks = np.arange(0, 32, 4)  # Show fewer y-ticks 
                if param_index == 0:  
                    ytick_labels = [round(1 + (i / 31) * (5 - 1), 2) for i in yticks]  
                else:  
                    ytick_labels = [round(0 + (i / 31) * (4 - 0), 2) for i in yticks]  
                plt.yticks(ticks=yticks, labels=ytick_labels)  
                
                # Adjust x-ticks  
                xticks = np.linspace(start, end, num=5, dtype=int)  # Show fewer x-ticks  
                plt.xticks(ticks=xticks, labels=xticks, rotation=0)  
                
                # Invert y-axis  
                plt.gca().invert_yaxis()  
                
                plt.show()  
        
        for strat, end in ranges:
            plt.figure(figsize=(14, 8))  
            sns.heatmap(rebate_usage_count[start:end, :].T, cmap='viridis', cbar_kws={'label': 'fitness'})  
            plt.title(f'Strategy Fitness for Rebate (Generations {start} to {end})')  
            plt.xlabel('Generations')  
            plt.ylabel(f'Rebate Values')  
            
            # Convert decimal values to parameter values for y-axis labels  
            yticks = np.arange(0, 32, 4)  # Show fewer y-ticks 
            ytick_labels = [round((i / 31), 2) for i in yticks]  

            plt.yticks(ticks=yticks, labels=ytick_labels)  
            
            # Adjust x-ticks  
            xticks = np.linspace(start, end, num=5, dtype=int)  # Show fewer x-ticks  
            plt.xticks(ticks=xticks, labels=xticks, rotation=0)  
            
            # Invert y-axis  
            plt.gca().invert_yaxis()  
            
            plt.show()  
            
def plot_combined_heatmap(para_usage_count, rebate_usage_count, generations):  
    
    if not os.path.exists('figures'):  
        os.makedirs('figures')  
    
    # 设置全局字体大小  
    plt.rcParams.update({'font.size': 30})  
        
    ranges = [(0, generations)]  
    
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))  # 创建一个包含3个子图的图形，宽度3*7=21  

    for param_index in range(2):  
        for start, end in ranges:  
            sns.heatmap(  
                para_usage_count[param_index][start:end, :].T,  
                cmap='viridis',  
                ax=ax[param_index],  
            )  
            ax[param_index].set_title(f'Strategy Fitness for Parameter $\\gamma_{{{param_index}}}$')  
            ax[param_index].set_xlabel('Generations')  
            ax[param_index].set_ylabel(f'Values of Parameter $\\gamma_{{{param_index}}}$')  
            
            # Convert decimal values to parameter values for y-axis labels  
            yticks = np.linspace(0, 31, 9, dtype=int)  # Generating 9 evenly spaced ticks from 0 to 31 
            if param_index == 0:  
                ytick_labels = [round(1 + (i / 31) * (5 - 1), 2) for i in yticks]  
            else:  
                ytick_labels = [round(0 + (i / 31) * (4 - 0), 2) for i in yticks]  
            ax[param_index].set_yticks(yticks)  
            ax[param_index].set_yticklabels(ytick_labels)  
            
            # Adjust x-ticks  
            xticks = np.linspace(start, end, num=5, dtype=int)  # Show fewer x-ticks  
            ax[param_index].set_xticks(xticks)  
            ax[param_index].set_xticklabels(xticks, rotation=0)  
            
            # Invert y-axis  
            ax[param_index].invert_yaxis()  

    for start, end in ranges:  
        sns.heatmap(  
            rebate_usage_count[start:end, :].T,  
            cmap='viridis',  
            ax=ax[2],  
        )  
        ax[2].set_title(f'Strategy Fitness for Rebate Ratio $\\alpha$')  
        ax[2].set_xlabel('Generations')  
        ax[2].set_ylabel('Values of Rebate Ratio $\\alpha$')  
        
        # Convert decimal values to parameter values for y-axis labels  
        yticks = np.linspace(0, 31, 9, dtype=int)  # Generating 9 evenly spaced ticks from 0 to 31 
        ytick_labels = [round((i / 31), 2) for i in yticks]  
        
        ax[2].set_yticks(yticks)  
        ax[2].set_yticklabels(ytick_labels)  
        
        # Adjust x-ticks  
        xticks = np.linspace(start, end, num=5, dtype=int)  # Show fewer x-ticks  
        ax[2].set_xticks(xticks)  
        ax[2].set_xticklabels(xticks, rotation=0)  
        
        # Invert y-axis  
        ax[2].invert_yaxis()  
        
    plt.tight_layout()  # 自动调整子图参数以达到更好的布局  
    plt.savefig(os.path.join('figures', 'fitness_evolution_heatmap_plot.png'))  
    plt.show()  




def plot_heatmap(para_usage_count, rebate_usage_count, generations):  
    if not os.path.exists('figures'):  
        os.makedirs('figures')  

    ranges = [(0, generations)]  
    
    

    for param_index in range(2):  
        for start, end in ranges:  
            fig, ax = plt.subplots(figsize=(8, 6))  

            sns.heatmap(  
                para_usage_count[param_index][start:end, :].T,  
                cmap='viridis',  
                ax=ax,  
            )  
            ax.set_title(f'Strategy Fitness for Parameter $\\gamma_{{{param_index+1}}}$', fontsize=16)  
            ax.set_xlabel('Iteration', fontsize=16)  
            ax.set_ylabel(f'Values of Parameter $\\gamma_{{{param_index+1}}}$', fontsize=16)  

            # Convert decimal values to parameter values for y-axis labels  
            yticks = np.arange(0, 32, 4)  # 生成从0到31的等间距  
            yticks = np.append(yticks, 31)  # Add the maximum value  
            if param_index == 0:  
                ytick_labels = [round(1 + (i / 31) * (5 - 1), 2) for i in yticks]  
            else:  
                ytick_labels = [round(0 + (i / 31) * (4 - 0), 2) for i in yticks]  
            ax.set_yticks(yticks)  
            ax.set_yticklabels(ytick_labels, rotation=0, fontsize=16)  # Set rotation for y-tick labels to 0  

            # Adjust x-ticks  
            xticks = np.linspace(start, end, num=5, dtype=int)  
            ax.set_xticks(xticks)  
            ax.set_xticklabels(xticks, rotation=0, fontsize=16)  
            ax.tick_params(axis='both', labelsize=16)
            # Invert y-axis  
            ax.invert_yaxis()  

            plt.tight_layout()  
            plt.savefig(os.path.join('figures', f'fitness_evolution_heatmap_param_{param_index}.png'),dpi=300)  
            plt.show()  
            plt.close(fig)  

    for start, end in ranges:  
        fig, ax = plt.subplots(figsize=(8, 6))  

        sns.heatmap(  
            rebate_usage_count[start:end, :].T,  
            cmap='plasma',  
            ax=ax,  
            cbar_kws={"orientation": "vertical"}  # Ensure color bar is vertical  
        )  
        ax.set_title(f'Strategy Fitness for Rebate Ratio $\\alpha$', fontsize=16)  
        ax.set_xlabel('Iteration',fontsize=16)  
        ax.set_ylabel('Values of Rebate Ratio $\\alpha$',fontsize=16)  

        # Convert decimal values to parameter values for y-axis labels  
        yticks = np.arange(0, 32, 4)  # 生成从0到31的等间距  
        yticks = np.append(yticks, 31)  # Add the maximum value  
        ytick_labels = [round((i / 31), 2) for i in yticks]  

        ax.set_yticks(yticks)  
        ax.set_yticklabels(ytick_labels, rotation=0,fontsize=16)  # Set rotation for y-tick labels to 0  

        # Adjust x-ticks  
        xticks = np.linspace(start, end, num=5, dtype=int)  
        ax.set_xticks(xticks)  
        ax.set_xticklabels(xticks, rotation=0, fontsize=16)  
        ax.tick_params(axis='both', labelsize=16)

        # Invert y-axis  
        ax.invert_yaxis()  

        plt.tight_layout()  
        plt.savefig(os.path.join('figures', 'fitness_evolution_heatmap_rebate.png'),dpi=300)  
        plt.show()  
        plt.close(fig)  
        
def calculate_and_plot_cv(para_usage_count, rebate_usage_count, generations):  
    # 初始化存储变异系数的数组  
    cv_para0 = np.zeros(generations)  
    cv_para1 = np.zeros(generations)  
    cv_rebate = np.zeros(generations)  

    # 计算每个 generation 的变异系数  
    for gen in range(generations):  
        # Para0  
        mean_para0 = np.mean(para_usage_count[0][gen, :])  
        std_para0 = np.std(para_usage_count[0][gen, :])  
        cv_para0[gen] = std_para0 / mean_para0 if mean_para0 != 0 else 0  

        # Para1  
        mean_para1 = np.mean(para_usage_count[1][gen, :])  
        std_para1 = np.std(para_usage_count[1][gen, :])  
        cv_para1[gen] = std_para1 / mean_para1 if mean_para1 != 0 else 0  

        # Rebate  
        mean_rebate = np.mean(rebate_usage_count[gen, :])  
        std_rebate = np.std(rebate_usage_count[gen, :])  
        cv_rebate[gen] = std_rebate / mean_rebate if mean_rebate != 0 else 0  

    # 可视化变异系数，从第100代开始  
    start_generation = 100  
    plt.figure(figsize=(12, 6))  
    plt.plot(range(start_generation, generations), cv_para0[start_generation:], label='CV of Para0')  
    plt.plot(range(start_generation, generations), cv_para1[start_generation:], label='CV of Para1')  
    plt.plot(range(start_generation, generations), cv_rebate[start_generation:], label='CV of Rebate')  
    plt.xlabel('Generations')  
    plt.ylabel('Coefficient of Variation')  
    plt.title('Coefficient of Variation over Generations (from 100 onwards)')  
    plt.legend()  
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()

def plot_cov_over_time(rebate_cov, para_cov, generations):  
    # Plot rebate CoV  
    plt.figure(figsize=(12, 6))  
    for i, builder_cov in enumerate(rebate_cov):  
        plt.plot(range(generations), builder_cov, label=f'Rebate CoV Builder {i}')  
    plt.xlabel('Generations')  
    plt.ylabel('Coefficient of Variation')  
    plt.title('Rebate Coefficient of Variation over Generations')  
    plt.legend()  
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()  

    # Plot para CoV for each parameter  
    for param_index in range(2):  
        plt.figure(figsize=(12, 6))  
        for i, searcher_cov in enumerate(para_cov[param_index]):  
            plt.plot(range(generations), searcher_cov, label=f'Para {param_index+1} CoV Searcher {i}')  
        plt.xlabel('Generations')  
        plt.ylabel('Coefficient of Variation')  
        plt.title(f'Parameter {param_index+1} Coefficient of Variation over Generations')  
        plt.legend()  
        plt.grid(True)  
        plt.tight_layout()  
        plt.show()  

def plot_average_cov_over_time(rebate_cov, para_cov, generations, filename='average_cov.png'):  
    # 计算每一代的平均 rebate CoV  
    avg_rebate_cov = np.mean(rebate_cov, axis=0)  

    # 计算每一代的平均 para CoV  
    avg_para_cov_0 = np.mean(para_cov[0], axis=0)  
    avg_para_cov_1 = np.mean(para_cov[1], axis=0)  

    # 绘制平均 CoV 曲线  
    plt.figure(figsize=(8, 6))  
    plt.plot(range(generations), avg_rebate_cov, label='CoV of Rebate Ratio $\\alpha$', color='red')  
    plt.plot(range(generations), avg_para_cov_0, label='CoV of Parameter $\\gamma_{1}$', color='green')  
    plt.plot(range(generations), avg_para_cov_1, label='CoV of Parameter $\\gamma_{2}$', color='blue')  

    plt.xlabel('Generations', fontsize=16)  
    plt.ylabel('Coefficient of Variation', fontsize=16)  
    # plt.title('Average Coefficient of Variation over Generations')  
    plt.legend(fontsize=16)  
    plt.grid(True)  
    
    # 设置 x 轴的范围  
    plt.xlim(0, generations)  
    plt.tight_layout()  
    
    plt.savefig(os.path.join('figures', 'cov.png'))  
    plt.show()  
    
def plot_average_cov_over_time_2(rebate_cov, para_cov, generations, filename='average_cov.png'):  

    avg_rebate_cov = np.mean(rebate_cov, axis=0)  
    avg_para_cov_0 = np.mean(para_cov[0], axis=0)  
    avg_para_cov_1 = np.mean(para_cov[1], axis=0)  

    sns.set_theme(style="whitegrid")  

    plt.figure(figsize=(8, 6))  
    plt.plot(range(generations), avg_rebate_cov, label='CoV of Rebate Ratio $\\alpha$', color='#e41a1c', linewidth=3, alpha=0.8)  
    plt.plot(range(generations), avg_para_cov_0, label='CoV of Parameter $\\gamma_{1}$', color='#377eb8', linewidth=3, alpha=0.8)  
    plt.plot(range(generations), avg_para_cov_1, label='CoV of Parameter $\\gamma_{2}$', color='#4daf4a', linewidth=3, alpha=0.8)  

    plt.xlabel('Iteration', fontsize=16)  
    plt.ylabel('Coefficient of Variation', fontsize=16)  
    # plt.title('Average Coefficient of Variation over Generations', fontsize=16)  
    plt.legend(fontsize=16, loc='upper right', frameon=True)  
    plt.grid(False)  # 如果不希望网格显示  
    plt.xlim(0, generations)  
    plt.tick_params(axis='both', labelsize=16)  # 设置主刻度字体为16  
    plt.xticks(fontsize=16)                    # 明确设置x轴刻度为16  
    plt.yticks(fontsize=16)                    # 明确设置y轴刻度为16  
    plt.tight_layout()  

    if not os.path.exists('figures'):  
        os.makedirs('figures')  

    plt.savefig(os.path.join('figures', filename),dpi=300)  
    plt.show()   
    
def moving_average(data, window_size=50):  
    """Calculate the moving average of a list of values."""  
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results_2(avg_rebate_history, avg_bid_history, generations):  
    # 设置 Seaborn 样式  
    sns.set(style="whitegrid")  

    # Plot average rebate and its moving average  
    generations_list, avg_rebates = zip(*avg_rebate_history)  
    plt.figure(figsize=(8, 6))  
    sns.lineplot(x=generations_list, y=avg_rebates, label='Average Rebate Ratio', alpha=0.6)  
    sns.lineplot(x=generations_list[:len(avg_rebates) - 49], y=moving_average(avg_rebates), label='Moving Average', color='red')  
    plt.xlabel('Iteration', fontsize=16)  
    plt.ylabel('Average Rebate Ratio', fontsize=16)  
    plt.legend(loc='lower right', fontsize=16)  # 将图例固定在右下角  
    plt.grid(False)  # 如果不希望网格显示  
    plt.xlim(0, generations)  # 设置 x 轴的范围  
    plt.tick_params(axis='both', labelsize=16)  # 刻度字体  
    plt.tight_layout()  
    plt.savefig(os.path.join('figures', 'rebate.png'),dpi=300)  
    plt.show()  

    # Plot average bid and its moving average  
    generations_list, avg_bids = zip(*avg_bid_history)  
    plt.figure(figsize=(8, 6))  
    sns.lineplot(x=generations_list, y=avg_bids, label='Average Bid Ratio', alpha=0.6)  
    sns.lineplot(x=generations_list[:len(avg_bids) - 49], y=moving_average(avg_bids), label='Moving Average', color='red')  
    plt.xlabel('Iteration', fontsize=16)  
    plt.ylabel('Average Bid Ratio', fontsize=16)  
    plt.legend(loc='lower right', fontsize=16)  # 将图例固定在右下角  
    plt.grid(False)  # 如果不希望网格显示  
    plt.xlim(0, generations)  # 设置 x 轴的范围  
    plt.tick_params(axis='both', labelsize=16)  # 刻度字体  
    plt.xticks(fontsize=16)  
    plt.yticks(fontsize=16)  
    plt.tight_layout()  
    plt.savefig(os.path.join('figures', 'bid.png'),dpi=300)  
    plt.show()  
    
    
def plot_heatmap_2(para_usage_count, rebate_usage_count, generations, sample_interval=1):  
    if not os.path.exists('figures'):  
        os.makedirs('figures')  

    ranges = [(0, generations)]  

    for param_index in range(2):  
        for start, end in ranges:  
            sample_indices = np.arange(start, end, sample_interval)  
            data_sampled = para_usage_count[param_index][sample_indices, :].T  

            fig, ax = plt.subplots(figsize=(8, 6))  
            sns.heatmap(  
                data_sampled,  
                cmap='viridis',  
                ax=ax,  
            )  
            ax.set_title(f'Strategy Fitness for Parameter $\\gamma_{{{param_index+1}}}$', fontsize=16)  
            ax.set_xlabel('Iteration', fontsize=16)  
            ax.set_ylabel(f'Values of Parameter $\\gamma_{{{param_index+1}}}$', fontsize=16)  

            yticks = np.arange(0, 32, 4)  
            yticks = np.append(yticks, 31)  
            if param_index == 0:  
                ytick_labels = [round(1 + (i / 31) * (5 - 1), 2) for i in yticks]  
            else:  
                ytick_labels = [round(0 + (i / 31) * (4 - 0), 2) for i in yticks]  
            ax.set_yticks(yticks)  
            ax.set_yticklabels(ytick_labels, rotation=0, fontsize=16)  

            # x-ticks for sampled indices  
            xticks = np.linspace(0, len(sample_indices)-1, num=5, dtype=int)  
            xtick_labels = [sample_indices[x] for x in xticks]  
            ax.set_xticks(xticks)  
            ax.set_xticklabels(xtick_labels, rotation=0, fontsize=16)  
            ax.tick_params(axis='both', labelsize=16)  
            ax.invert_yaxis()  

            plt.tight_layout()  
            plt.savefig(os.path.join('figures', f'fitness_evolution_heatmap_param_{param_index}.png'), dpi=300)  
            plt.show()  
            plt.close(fig)  

    for start, end in ranges:  
        sample_indices = np.arange(start, end, sample_interval)  
        data_sampled = rebate_usage_count[sample_indices, :].T  

        fig, ax = plt.subplots(figsize=(8, 6))  
        sns.heatmap(  
            data_sampled,  
            cmap='plasma',  
            ax=ax,  
            cbar_kws={"orientation": "vertical"}  
        )  
        ax.set_title(f'Strategy Fitness for Rebate Ratio $\\alpha$', fontsize=16)  
        ax.set_xlabel('Iteration', fontsize=16)  
        ax.set_ylabel('Values of Rebate Ratio $\\alpha$', fontsize=16)  

        yticks = np.arange(0, 32, 4)  
        yticks = np.append(yticks, 31)  
        ytick_labels = [round((i / 31), 2) for i in yticks]  

        ax.set_yticks(yticks)  
        ax.set_yticklabels(ytick_labels, rotation=0, fontsize=16)  

        # x-ticks for sampled indices  
        xticks = np.linspace(0, len(sample_indices)-1, num=5, dtype=int)  
        xtick_labels = [sample_indices[x] for x in xticks]  
        ax.set_xticks(xticks)  
        ax.set_xticklabels(xtick_labels, rotation=0, fontsize=16)  
        ax.tick_params(axis='both', labelsize=16)  
        ax.invert_yaxis()  

        plt.tight_layout()  
        plt.savefig(os.path.join('figures', 'fitness_evolution_heatmap_rebate.png'), dpi=300)  
        plt.show()  
        plt.close(fig)  

        
# 主函数
if __name__ == "__main__":
    generations = 10000
    sim = Simulation(num_builders=10, num_searchers=10, probability = 0.8, generations=generations, temperature=2)
    sim = sim.run()
    # sim.plot_results()  
    plot_results_2(sim.avg_rebate_history, sim.avg_bid_history, generations)  
    #plot_results_3(sim.avg_rebate_history, sim.avg_bid_history, generations)  
    # sim.plot_results_avg_reward()
    # plot_strategy_usage_heatmap(sim.para_usage_count, sim.rebate_usage_count, generations = 100000)  
    # plot_combined_heatmap(sim.para_usage_count, sim.rebate_usage_count, generations = 100000)  
    plot_heatmap(sim.para_usage_count, sim.rebate_usage_count, generations)
    plot_heatmap(sim.para_usage_count, sim.rebate_usage_count, generations, sample_interval=10)  
    # calculate_and_plot_cv(sim.para_usage_count, sim.rebate_usage_count, generations)  
    # plot_cov_over_time(sim.rebate_cov, sim.para_cov, generations)  
    # plot_average_cov_over_time(sim.rebate_cov, sim.para_cov, generations) 
    plot_average_cov_over_time_2(sim.rebate_cov, sim.para_cov, generations)


    