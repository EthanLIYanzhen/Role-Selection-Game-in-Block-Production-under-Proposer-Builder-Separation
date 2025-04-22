# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 00:15:13 2024

@author: admin
"""

import pandas as pd  
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import os


df = pd.read_csv('experiment_results_10000_4.csv', sep=',') 

# 打印 DataFrame 的前几行  
print("导入的 DataFrame 内容：")  
print(df.head())  

# 打印 DataFrame 的列名  
print("DataFrame 列名:", df.columns.tolist())  

df_mean = df.groupby(['Probability', 'Num Builders', 'Num Searchers'], as_index=False).mean()  
probability_list = [i/10 for i in range(11)]  

# 提取特定概率的结果  
results_dict = {}  

for prob in probability_list:  
    # 过滤出当前概率的结果  
    filtered_results = df_mean[df_mean['Probability'] == prob]  
    
    # 将结果存储在字典中  
    results_dict[prob] = filtered_results  


  
# 现在可以对每个固定概率的结果进行计算分析  

alpha_list = [0.1,1,10,100]



# 创建一个 2x2 的子图  
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  

for idx, alpha in enumerate(alpha_list):  
    steady_state_distributions = []  # 用于存储稳态分布  
    prob_values = []  # 用于存储 prob 值
    for prob, results in results_dict.items():  
        C = np.zeros((2, 2))  
        T_1 = np.zeros(10)  
        T_2 = np.zeros(10)  
        
        for k in range(1, 10):  
            T_1[k] = ((k * (10 - k)) / (10 * 9)) / (1 + math.exp(alpha * (results.values[k - 1][6] - results.values[k - 1][5])))  
            T_2[k] = ((k * (10 - k)) / (10 * 9)) / (1 + math.exp(alpha * (results.values[k - 1][5] - results.values[k - 1][6])))  

        # 计算 C[0][1] 和 C[1][0]  
        C[0][1] = 1 / (1 + sum([np.prod(T_1[1:l + 1] / T_2[1:l + 1]) for l in range(1, 10)]))  
        C[1][0] = 1 / (1 + sum([np.prod(T_2[1:l + 1] / T_1[1:l + 1]) for l in range(1, 10)]))  

        C[0][0] = 1 - C[0][1]  
        C[1][1] = 1 - C[1][0]  

        # 定义单位矩阵 I  
        I = np.eye(C.shape[0])  

        # 定义矩阵 E  
        E = np.ones((C.shape[0], C.shape[0]))  

        # 计算 [P - I + E]  
        matrix_to_inverse = C - I + E  

        # 求逆  
        inverse_matrix = np.linalg.inv(matrix_to_inverse)  

        # 定义单位向量 1  
        one_vector = np.ones((1, C.shape[0]))  

        # 计算稳态分布 π  
        pi = one_vector @ inverse_matrix  

        # 存储稳态分布和 prob 值  
        steady_state_distributions.append(pi.flatten())  # 这里可以使用 append  
        prob_values.append(prob)  

    # 将稳态分布转换为 NumPy 数组以便于绘图  
    steady_state_distributions = np.array(steady_state_distributions)  

    # 在对应的子图中绘制数据  
    ax = axs[idx // 2, idx % 2]  # 选择子图  
    ax.plot(prob_values, steady_state_distributions[:, 0], label='Block Building', marker='o')  
    ax.plot(prob_values, steady_state_distributions[:, 1], label='Bundle Sharing', marker='o')  

    # 添加图形标题和标签  
    ax.set_title(f'Stationary Distribution (alpha={alpha})')  
    ax.set_xlabel('Probability of Conflicts')  
    ax.set_ylabel('Stationary Distribution Mass')  
    ax.legend()  
    ax.grid()  
    ax.set_xlim(0, 1)  

# 调整布局  
plt.tight_layout()  

# 保存图形  
plt.savefig(os.path.join('figures', 'combined_rank.png'),dpi=300)  
plt.show()  


fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 一行四张子图  

for idx, alpha in enumerate(alpha_list):  
    steady_state_distributions = []  
    prob_values = []  
    for prob, results in results_dict.items():  
        C = np.zeros((2, 2))  
        T_1 = np.zeros(10)  
        T_2 = np.zeros(10)  
        
        for k in range(1, 10):  
            T_1[k] = ((k * (10 - k)) / (10 * 9)) / (1 + math.exp(alpha * (results.values[k - 1][6] - results.values[k - 1][5])))  
            T_2[k] = ((k * (10 - k)) / (10 * 9)) / (1 + math.exp(alpha * (results.values[k - 1][5] - results.values[k - 1][6])))  
        
        C[0][1] = 1 / (1 + sum([np.prod(T_1[1:l + 1] / T_2[1:l + 1]) for l in range(1, 10)]))  
        C[1][0] = 1 / (1 + sum([np.prod(T_2[1:l + 1] / T_1[1:l + 1]) for l in range(1, 10)]))  
        C[0][0] = 1 - C[0][1]  
        C[1][1] = 1 - C[1][0]  
        I = np.eye(C.shape[0])  
        E = np.ones((C.shape[0], C.shape[0]))  
        matrix_to_inverse = C - I + E  
        inverse_matrix = np.linalg.inv(matrix_to_inverse)  
        one_vector = np.ones((1, C.shape[0]))  
        pi = one_vector @ inverse_matrix  
        steady_state_distributions.append(pi.flatten())  
        prob_values.append(prob)  
    steady_state_distributions = np.array(steady_state_distributions)  
    ax = axs[idx]  
    ax.plot(prob_values, steady_state_distributions[:, 0], label='Block Building', marker='o')  
    ax.plot(prob_values, steady_state_distributions[:, 1], label='Bundle Sharing', marker='o')  
    ax.set_title(f'Stationary Distribution (alpha={alpha})',fontsize=16)  
    ax.set_xlabel('Probability of Conflicts',fontsize=16)  
    ax.set_ylabel('Stationary Distribution Mass',fontsize=16)  
    ax.legend(fontsize=16)  
    ax.tick_params(axis='both', labelsize=16)  
    ax.grid(False)  
    ax.set_xlim(0, 1)  

plt.tight_layout()  
plt.savefig(os.path.join('figures', 'combined_rank_1.png'), dpi=300)  
plt.show()  


fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 一行四张子图  

for idx, alpha in enumerate(alpha_list):  
    steady_state_distributions = []  
    prob_values = []  
    for prob, results in results_dict.items():  
        C = np.zeros((2, 2))  
        T_1 = np.zeros(10)  
        T_2 = np.zeros(10)  
        for k in range(1, 10):  
            T_1[k] = ((k * (10 - k)) / (10 * 9)) / (1 + math.exp(alpha * (results.values[k - 1][6] - results.values[k - 1][5])))  
            T_2[k] = ((k * (10 - k)) / (10 * 9)) / (1 + math.exp(alpha * (results.values[k - 1][5] - results.values[k - 1][6])))  
        C[0][1] = 1 / (1 + sum([np.prod(T_1[1:l + 1] / T_2[1:l + 1]) for l in range(1, 10)]))  
        C[1][0] = 1 / (1 + sum([np.prod(T_2[1:l + 1] / T_1[1:l + 1]) for l in range(1, 10)]))  
        C[0][0] = 1 - C[0][1]  
        C[1][1] = 1 - C[1][0]  
        I = np.eye(C.shape[0])  
        E = np.ones((C.shape[0], C.shape[0]))  
        matrix_to_inverse = C - I + E  
        inverse_matrix = np.linalg.inv(matrix_to_inverse)  
        one_vector = np.ones((1, C.shape[0]))  
        pi = one_vector @ inverse_matrix  
        steady_state_distributions.append(pi.flatten())  
        prob_values.append(prob)  
    steady_state_distributions = np.array(steady_state_distributions)  
    ax = axs[idx]  

    # 美化线条、颜色  
    ax.plot(prob_values, steady_state_distributions[:, 0],  
            label='Block Building', marker='o', color='#1f77b4',  
            linewidth=2, markersize=7)  
    ax.plot(prob_values, steady_state_distributions[:, 1],  
            label='Bundle Sharing', marker='s', color='#d62728',  
            linewidth=2, markersize=7)  

    # 只在最左面panel显示Y轴，其余全部隐藏  
    if idx == 0:  
        ax.set_ylabel('Stationary Distribution Mass', fontsize=16)  
    else:  
        ax.set_ylabel('')  
        ax.set_yticklabels([])  
        ax.tick_params(left=False)  

    ax.set_title(f'({chr(65+idx)}) α={alpha}', fontsize=16)  
    ax.set_xlabel('Probability of Conflicts', fontsize=16)  
    #ax.legend(fontsize=16, frameon=False, loc='best')  # 美化图例  
    ax.tick_params(axis='both', labelsize=16)  
    ax.set_xlim(0, 1)  
    ax.set_ylim(0, 1)  # 让所有panel纵轴统一  
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)  

    # 去除顶部和右侧边框  
    ax.spines['top'].set_visible(False)  
    ax.spines['right'].set_visible(False)  

# 只在这里加，总图例浮在外部（避免遮内容）  
fig.legend(  
    labels=['Block Building', 'Bundle Sharing'],  
    loc='upper center',  # 你可以选'upper center', 'lower center', 'center right'等  
    bbox_to_anchor=(0.5, 1.1),  # 往上挪一些，不遮住顶部  
    ncol=2,  # 横排排列  
    fontsize=16,  
    frameon=False  
)  

plt.tight_layout()  
plt.savefig(os.path.join('figures', 'combined_rank_2.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)  
plt.show()  