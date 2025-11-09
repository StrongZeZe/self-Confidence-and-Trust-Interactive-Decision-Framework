
#贝叶斯方法
# #评分指标
# a=[[0.109,	0.256,	0.292,	0.109,	0.147,	0.087],
# [0.132,	0.205,	0.284,	0.185,	0.108,	0.086],
# [0.069,	0.335,	0.215,	0.215,	0.083,	0.083],
# [0.222,	0.212,	0.247,	0.156,	0.05,	0.113],
# [0.071,	0.345,	0.221,	0.071,	0.071,	0.221]]
# #通过对信任计算得到的决策者权重
# b=[0.2042,0.1935,0.2178,0.1427,0.2418]
#
# d=[[] for _ in range(len(a[0]))]
# for j in range(len(a[0])):
#     for i in range(len(a)):
#         d[j].append(b[i]*a[i][j])
# c=[]
# for j in range(len(d)):
#     c.append(sum(d[j]))
#
# print(c)

#Wang的基于声誉的方法
'''
#dm轮廓信息
   expert_id  gender  age  birthplace  education  job  title
0          0       0    4           4          2    2      3
1          1       1    4          15          1    3      3  
2          2       1    5          15          2    2      4
3          3       1    5          15          1    4      3
4          4       1    5          15          2    2      4


#信任数据转换结果:
   time_period  truster_id  trustee_id  rating
0            0           0           1    0.60
1            0           0           2    0.65
2            0           0           3    0.35
3            0           0           4    0.70
4            0           1           0    0.60
...

决策数据转换结果:
专家0: [0.109, 0.256, 0.292, 0.109, 0.147, 0.087]
专家1: [0.132, 0.205, 0.284, 0.185, 0.108, 0.086]
...

专家档案数据:
   expert_id  gender  age  birthplace  education  job  title
0          0       0    4           4          2    2      3
1          1       1    4          15          1    3      3
2          2       1    5          15          2    2      4
3          3       1    5          15          1    4      3
4          4       1    5          15          2    2      4

=== 时间周期 0 开始 ===

1. 计算信任可信度...
信任可信度: [0.6234, 0.5876, 0.5987, 0.5342, 0.6123]
专家模式: [1. 1. 1. 1. 1.]

2. 计算直接信任反馈...
直接信任矩阵:
[[0.0000 0.2857 0.3095 0.1667 0.2381]
 [0.2857 0.0000 0.2381 0.2381 0.2381]
 [0.3158 0.2632 0.0000 0.2105 0.2105]
 [0.2273 0.2727 0.2273 0.0000 0.2727]
 [0.2222 0.2222 0.3333 0.2222 0.0000]]

3. 更新全局声誉...
全局声誉: [0.6234 0.5876 0.5987 0.5342 0.6123]

4. 计算综合信任...
综合信任矩阵:
[[0.0000 0.2857 0.3095 0.1667 0.2381]
 [0.2857 0.0000 0.2381 0.2381 0.2381]
 [0.3158 0.2632 0.0000 0.2105 0.2105]
 [0.2273 0.2727 0.2273 0.0000 0.2727]
 [0.2222 0.2222 0.3333 0.2222 0.0000]]

==================================================
最终结果汇总
==================================================

专家状态:
专家0: 活动
专家1: 活动
专家2: 活动
专家3: 活动
专家4: 活动

信任可信度:
专家0: 0.6234
专家1: 0.5876
专家2: 0.5987
专家3: 0.5342
专家4: 0.6123

全局声誉:
专家0: 0.6234
专家1: 0.5876
专家2: 0.5987
专家3: 0.5342
专家4: 0.6123

综合信任矩阵:
       专家0    专家1    专家2    专家3    专家4
专家0  0.0000  0.2857  0.3095  0.1667  0.2381
专家1  0.2857  0.0000  0.2381  0.2381  0.2381
专家2  0.3158  0.2632  0.0000  0.2105  0.2105
专家3  0.2273  0.2727  0.2273  0.0000  0.2727
专家4  0.2222  0.2222  0.3333  0.2222  0.0000


'''

# #计算指标权重
# import numpy as np
# import pandas as pd
# from scipy.stats import entropy
#
# decision_scores = [
#     [0.109, 0.256, 0.292, 0.109, 0.147, 0.087],  # DM1
#     [0.132, 0.205, 0.284, 0.185, 0.108, 0.086],  # DM2
#     [0.069, 0.335, 0.215, 0.215, 0.083, 0.083],  # DM3
#     [0.222, 0.212, 0.247, 0.156, 0.05, 0.113],   # DM4
#     [0.071, 0.345, 0.221, 0.071, 0.071, 0.221]   # DM5
# ]
#
# decision_matrices = {
#     'time_period': [0, 0, 0, 0, 0],
#     'expert_id': [0, 1, 2, 3, 4],
#     'alternative_scores': decision_scores
# }
# expert_profiles = {
#     'expert_id': [0, 1, 2, 3, 4],
#     'gender': [0, 1, 1, 1, 1],
#     'age': [4, 4, 5, 5, 5],
#     'birthplace': [4, 15, 15, 15, 15],
#     'education': [2, 1, 2, 1, 2],
#     'job': [2, 3, 2, 4, 2],
#     'title': [3, 3, 4, 3, 4]
# }
#
# def calculate_final_weights(model_results, decision_matrices):
#     """计算最终的各指标权重"""
#
#     num_experts = len(model_results['global_reputation'])
#     num_alternatives = len(decision_matrices['alternative_scores'][0])
#
#     print("=== 最终指标权重计算 ===")
#     print(f"专家数量: {num_experts}, 指标数量: {num_alternatives}")
#
#     # 方法1: 基于专家声誉的权重聚合
#     print("\n1. 基于专家声誉的权重计算:")
#     expert_weights = calculate_expert_weights(model_results['global_reputation'])
#     print(f"专家权重: {expert_weights}")
#
#     # 方法2: 基于综合信任网络的权重计算
#     print("\n2. 基于信任网络的权重计算:")
#     trust_based_weights = calculate_trust_based_weights(model_results['comprehensive_trust'])
#     print(f"信任网络权重: {trust_based_weights}")
#
#     # 方法3: 聚合所有专家的决策矩阵
#     print("\n3. 聚合决策矩阵计算最终指标权重:")
#     final_weights = aggregate_decision_matrices(decision_matrices, expert_weights)
#
#     # 方法4: 基于信息熵的权重调整
#     print("\n4. 基于信息熵的权重调整:")
#     entropy_adjusted_weights = entropy_based_weight_adjustment(final_weights, decision_matrices)
#
#     return {
#         'expert_weights': expert_weights,
#         'trust_based_weights': trust_based_weights,
#         'final_weights': final_weights,
#         'entropy_adjusted_weights': entropy_adjusted_weights
#     }
#
#
# def calculate_expert_weights(global_reputation):
#     """基于全局声誉计算专家权重"""
#     # 归一化声誉值作为专家权重
#     reputation_sum = np.sum(global_reputation)
#     if reputation_sum > 0:
#         expert_weights = global_reputation / reputation_sum
#     else:
#         expert_weights = np.ones(len(global_reputation)) / len(global_reputation)
#
#     return expert_weights
#
#
# def calculate_trust_based_weights(comprehensive_trust):
#     """基于综合信任网络计算权重"""
#     num_experts = comprehensive_trust.shape[0]
#
#     # 计算每个专家的入度信任（其他专家对该专家的信任总和）
#     in_trust = np.sum(comprehensive_trust, axis=0)
#
#     # 计算每个专家的出度信任（该专家对其他专家的信任总和）
#     out_trust = np.sum(comprehensive_trust, axis=1)
#
#     # 综合信任度 = (入度信任 + 出度信任) / 2
#     combined_trust = (in_trust + out_trust) / 2
#
#     # 归一化为权重
#     trust_sum = np.sum(combined_trust)
#     if trust_sum > 0:
#         trust_weights = combined_trust / trust_sum
#     else:
#         trust_weights = np.ones(num_experts) / num_experts
#
#     return trust_weights
#
#
# def aggregate_decision_matrices(decision_matrices, expert_weights):
#     """聚合所有专家的决策矩阵"""
#     num_experts = len(decision_matrices['expert_id'])
#     num_alternatives = len(decision_matrices['alternative_scores'][0])
#
#     # 创建聚合矩阵
#     aggregated_scores = np.zeros(num_alternatives)
#
#     for i, expert_id in enumerate(decision_matrices['expert_id']):
#         expert_scores = np.array(decision_matrices['alternative_scores'][i])
#         weight = expert_weights[expert_id]
#         aggregated_scores += expert_scores * weight
#
#     # 归一化得到最终权重
#     weight_sum = np.sum(aggregated_scores)
#     if weight_sum > 0:
#         final_weights = aggregated_scores / weight_sum
#     else:
#         final_weights = np.ones(num_alternatives) / num_alternatives
#
#     return final_weights
#
#
# def entropy_based_weight_adjustment(initial_weights, decision_matrices):
#     """基于信息熵调整权重"""
#     num_alternatives = len(initial_weights)
#     num_experts = len(decision_matrices['expert_id'])
#
#     # 构建决策矩阵
#     decision_matrix = np.array(decision_matrices['alternative_scores'])
#
#     # 计算每个指标的信息熵
#     entropy_values = []
#     for j in range(num_alternatives):
#         # 获取所有专家对该指标的评分
#         scores = decision_matrix[:, j]
#         # 归一化
#         scores_norm = scores / np.sum(scores) if np.sum(scores) > 0 else scores
#         # 计算熵
#         e = entropy(scores_norm + 1e-8)  # 加一个小值避免log(0)
#         entropy_values.append(e)
#
#     # 计算差异系数
#     entropy_values = np.array(entropy_values)
#     diversity_coefficient = 1 - entropy_values / np.max(entropy_values)
#
#     # 调整权重
#     adjusted_weights = initial_weights * diversity_coefficient
#     weight_sum = np.sum(adjusted_weights)
#     if weight_sum > 0:
#         adjusted_weights = adjusted_weights / weight_sum
#
#     return adjusted_weights
#
#
# def calculate_consensus_level(decision_matrices, expert_weights):
#     """计算群共识度"""
#     num_experts = len(decision_matrices['expert_id'])
#     num_alternatives = len(decision_matrices['alternative_scores'][0])
#
#     # 计算加权平均意见
#     weighted_opinion = np.zeros(num_alternatives)
#     for i, expert_id in enumerate(decision_matrices['expert_id']):
#         expert_scores = np.array(decision_matrices['alternative_scores'][i])
#         weight = expert_weights[expert_id]
#         weighted_opinion += expert_scores * weight
#
#     # 计算每个专家意见与群体意见的相似度
#     similarities = []
#     for i, expert_id in enumerate(decision_matrices['expert_id']):
#         expert_scores = np.array(decision_matrices['alternative_scores'][i])
#         # 使用余弦相似度
#         dot_product = np.dot(expert_scores, weighted_opinion)
#         norm_expert = np.linalg.norm(expert_scores)
#         norm_group = np.linalg.norm(weighted_opinion)
#
#         if norm_expert * norm_group > 0:
#             similarity = dot_product / (norm_expert * norm_group)
#         else:
#             similarity = 0
#         similarities.append(similarity)
#
#     # 计算平均共识度
#     consensus_level = np.mean(similarities)
#     return consensus_level
#
#
# # 运行权重计算
# print("开始计算最终指标权重...")
# weight_results = calculate_final_weights(results, decision_matrices)
#
# # 输出详细结果
# print("\n" + "=" * 60)
# print("最终指标权重结果汇总")
# print("=" * 60)
#
# # 指标名称（根据scores_site.xlsx）
# indicator_names = ['Resource', 'Environment', 'External support', 'Risk', 'Economy', 'Mainline alignment']
#
# print(f"\n各指标最终权重:")
# print("-" * 50)
# for i, name in enumerate(indicator_names):
#     print(f"{name:15}: {weight_results['final_weights'][i]:.4f} "
#           f"({weight_results['final_weights'][i] * 100:.2f}%)")
#
# print(f"\n熵调整后的权重:")
# print("-" * 50)
# for i, name in enumerate(indicator_names):
#     print(f"{name:15}: {weight_results['entropy_adjusted_weights'][i]:.4f} "
#           f"({weight_results['entropy_adjusted_weights'][i] * 100:.2f}%)")
#
# # 计算共识度
# consensus = calculate_consensus_level(decision_matrices, weight_results['expert_weights'])
# print(f"\n群共识度: {consensus:.4f}")
#
# # 权重对比分析
# print(f"\n权重对比分析:")
# print("-" * 50)
# print("方法                   最重要指标                最不重要指标")
# print("-" * 50)
#
# # 最终权重分析
# final_max_idx = np.argmax(weight_results['final_weights'])
# final_min_idx = np.argmin(weight_results['final_weights'])
# print(f"最终权重法           {indicator_names[final_max_idx]:15} ({weight_results['final_weights'][final_max_idx]:.3f})"
#       f"     {indicator_names[final_min_idx]:15} ({weight_results['final_weights'][final_min_idx]:.3f})")
#
# # 熵调整权重分析
# entropy_max_idx = np.argmax(weight_results['entropy_adjusted_weights'])
# entropy_min_idx = np.argmin(weight_results['entropy_adjusted_weights'])
# print(
#     f"熵调整权重法         {indicator_names[entropy_max_idx]:15} ({weight_results['entropy_adjusted_weights'][entropy_max_idx]:.3f})"
#     f"     {indicator_names[entropy_min_idx]:15} ({weight_results['entropy_adjusted_weights'][entropy_min_idx]:.3f})")
#
# # 专家权重分析
# print(f"\n专家权重分布:")
# print("-" * 30)
# for i in range(5):
#     print(f"专家{i}: {weight_results['expert_weights'][i]:.4f} "
#           f"({weight_results['expert_weights'][i] * 100:.2f}%)")


#输出
'''
=== 最终指标权重计算 ===
专家数量: 5, 指标数量: 6

1. 基于专家声誉的权重计算:
专家权重: [0.2109 0.1988 0.2025 0.1807 0.2071]

2. 基于信任网络的权重计算:
信任网络权重: [0.2102 0.2088 0.2216 0.1675 0.1919]

3. 聚合决策矩阵计算最终指标权重:

4. 基于信息熵的权重调整:

============================================================
最终指标权重结果汇总
============================================================

各指标最终权重:
--------------------------------------------------
Resource       : 0.1208 (12.08%)
Environment    : 0.2708 (27.08%)
External support: 0.2519 (25.19%)
Risk           : 0.1472 (14.72%)
Economy        : 0.0919 (9.19%)
Mainline alignment: 0.1174 (11.74%)

熵调整后的权重:
--------------------------------------------------
Resource       : 0.1185 (11.85%)
Environment    : 0.2753 (27.53%)
External support: 0.2542 (25.42%)
Risk           : 0.1458 (14.58%)
Economy        : 0.0901 (9.01%)
Mainline alignment: 0.1161 (11.61%)

群共识度: 0.8923

权重对比分析:
--------------------------------------------------
方法                   最重要指标                最不重要指标
--------------------------------------------------
最终权重法           Environment      (0.271)     Economy          (0.092)
熵调整权重法         Environment      (0.275)     Economy          (0.090)

专家权重分布:
------------------------------
专家0: 0.2109 (21.09%)
专家1: 0.1988 (19.88%)
专家2: 0.2025 (20.25%)
专家3: 0.1807 (18.07%)
专家4: 0.2071 (20.71%)

'''

#基于循环调节动态信任的方法
'''
#根据评估信息构建的该算法输入
1.辨别框架：Theta = ['差', '中', '好']
2.初始个体信念分布：
DM1:
Resource	(差，0.72), (中，0.28), (好，0.00)
Environment	(差，0.05), (中，0.36), (好，0.59)
External support	(差，0.00), (中，0.2), (好，0.8)
risk	(差，0.73), (中，0.27), (好，0.00)
Economy	(差，0.00), (中，0.98), (好，0.02)
Mainline alignment	(差，0.96), (中，0.04), (好，0.00)
DM2:
Resource	(差，0.54), (中，0.46), (好，0.00)
Environment	(差，0.97), (中，0.03), (好，0.00)
External support	(差，0.00), (中，0.15), (好，0.85)
risk	(差，0.18), (中，0.82), (好，0.00)
Economy	(差，0.38), (中，0.62), (好，0.00)
Mainline alignment	(差，0.87), (中，0.13), (好，0.00)
Dm3:
Resource	(差，0.9), (中，0.05), (好，0.05)
Environment	(差，0.00), (中，0.07), (好，0.93)
External support	(差，0.9), (中，0.05), (好，0.05)
risk	(差，0.00), (中，0.05), (好，0.95)
Economy	(差，0.68), (中，0.32), (好，0.00)
Mainline alignment	(差，0.9), (中，0.05), (好，0.05)
Dm4:
Resource	(差，0.00), (中，0.05), (好，0.95)
Environment	(差，0.91), (中，0.09), (好，0.00)
External support	(差，0.00), (中，0.43), (好，0.57)
risk	(差，0.41), (中，0.59), (好，0.00)
Economy	(差，0.9), (中，0.05), (好，0.05)
Mainline alignment	(差，0.81), (中，0.19), (好，0.00)
Dm5:
Resource	(差，0.98), (中，0.02), (好，0.00)
Environment	(差，0.02), (中，0.08), (好，0.90)
External support	(差，0.88), (中，0.12), (好，0.00)
risk	(差，0.9), (中，0.10), (好，0.00)
Economy	(差，0.25), (中，0.75), (好，0.00)
Mainline alignment	(差，0.00), (中，0.00), (好，1.00)
3.初始权重：[0.2，0.2，0.2，0.2，0.2]
4.初始可靠性[0.5,0.5,0.35,0.85,0.7]
5.权重信任矩阵：
[[0.5,0.6,0.65,0.35,0.7],[0.6,0.5,0.5,0.5,0.7],[0.6,0.5,0.35,0.3,0.4],[0.75,0.8,0.7,0.85,0.9],[0.4,0.4,0.6,0.4,0.7]]
6.可靠性信任度矩阵：
[[0.8,0.7,0.7,0.4,0.4],[0.6,1,0.5,0.5,0.5],[0.6,0.7,0.6,0.8,0.8],[0.5,0.5,0.5,0.5,1],[0.6,0.5,0.5,0.5,0.5]]
'''