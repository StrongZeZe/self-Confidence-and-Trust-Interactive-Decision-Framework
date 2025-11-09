#柱状图，折线图可删
'''
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据定义
shape_params = [0.05, 0.5, 5]
indices = ['Resource', 'Environment', 'External Support', 'Risk', 'Economy', 'Geology']

# 三种信任度量方式的数据
distance_trust_data = {0.05: [0.19435918101035612, 0.21617573274452587, 0.20093410577280194, 0.16132607294783766, 0.12383395045730676, 0.1033709570671718],
                       0.5: [0.1943754630641115, 0.21616710528303165, 0.20092693236772924, 0.16132439616369118, 0.1238349083927575, 0.10337119472867909],
                       5: [0.19450480559225614, 0.2161071810138735, 0.20088160563985388, 0.16130519360660486, 0.12383013897002805, 0.10337107517738361]}
fuzzy_distance_trust_data = {0.05: [0.198930686334785, 0.21546736081580625, 0.20231463610557587, 0.1590456063313716, 0.12129877126720091, 0.10294293914526033],
                             0.5: [0.1993989856119705, 0.21387287759834359, 0.20100835111365242, 0.15982578372576903, 0.1227028089313758, 0.10319119301888857],
                             5: [0.19893368505068273, 0.21548581490950008, 0.2023395825622047, 0.15903026945902307, 0.12127236688571998, 0.10293828113286946]}
nonlinear_trust_data = {0.05: [0.19458228553766896, 0.21617993743770908, 0.20101880399947628, 0.1611994071744093, 0.12367428332043824, 0.10334528253029826],
                        0.5: [0.19460091831879867, 0.2161719021931727, 0.20101301646807418, 0.1611961724659559, 0.12367285408688929, 0.10334513646710931],
                        5: [0.19473456515110113, 0.21612043849638687, 0.2009790320011629, 0.1611694587007168, 0.12365382150195187, 0.10334268414868038]}

# 计算每个数据在其权重集中的排序
def get_rank_labels(data_dict):
    rank_dict = {}
    for shape_param, values in data_dict.items():
        # 对值从大到小排序，获取每个值的排名
        sorted_values = sorted(values, reverse=True)
        ranks = [sorted_values.index(v) + 1 for v in values]
        rank_dict[shape_param] = ranks
    return rank_dict


distance_ranks = get_rank_labels(distance_trust_data)
fuzzy_ranks = get_rank_labels(fuzzy_distance_trust_data)
nonlinear_ranks = get_rank_labels(nonlinear_trust_data)

# 美观的颜色方案 - 同一类信任度量方式使用相近颜色
colors = {
    'distance': ['#1f77b4', '#4c72b0', '#6a6eb5'],  # 蓝色系
    'fuzzy': ['#ff7f0e', '#ff9e4a', '#ffbb78'],  # 橙色系
    'nonlinear': ['#2ca02c', '#59a14f', '#86c67c']  # 绿色系
}

# 坐标轴位置设置
x_pos = np.arange(len(indices))
y_pos = np.arange(len(shape_params))

# 创建第一幅图：柱状图
fig1 = plt.figure(figsize=(16, 10))
ax1 = fig1.add_subplot(111, projection='3d')

bar_width = 0.25
bar_depth = 0.25

# 绘制柱状图
for i, shape_param in enumerate(shape_params):
    y = np.full(len(x_pos), i)

    # 距离信任度量
    distances = distance_trust_data[shape_param]
    ax1.bar3d(x_pos - bar_width, y, np.zeros(len(distances)),
              bar_width, bar_depth, distances,
              color=colors['distance'][i], alpha=0.8,
              label=f'Distance Trust (α={shape_param})' if i == 0 else "",
              edgecolor='white', linewidth=0.5)

    # 模糊信息距离信任度量
    fuzzy_distances = fuzzy_distance_trust_data[shape_param]
    ax1.bar3d(x_pos, y, np.zeros(len(fuzzy_distances)),
              bar_width, bar_depth, fuzzy_distances,
              color=colors['fuzzy'][i], alpha=0.8,
              label=f'Fuzzy Distance Trust (α={shape_param})' if i == 0 else "",
              edgecolor='white', linewidth=0.5)

    # 非线性度量信任
    nonlinear_distances = nonlinear_trust_data[shape_param]
    ax1.bar3d(x_pos + bar_width, y, np.zeros(len(nonlinear_distances)),
              bar_width, bar_depth, nonlinear_distances,
              color=colors['nonlinear'][i], alpha=0.8,
              label=f'Nonlinear Trust (α={shape_param})' if i == 0 else "",
              edgecolor='white', linewidth=0.5)

# 设置坐标轴标签
ax1.set_xlabel('Index', labelpad=20, fontsize=12, fontweight='bold')
ax1.set_ylabel('Shape Parameter (α)', labelpad=20, fontsize=12, fontweight='bold')
ax1.set_zlabel('Weight', labelpad=20, fontsize=12, fontweight='bold')

# 设置x轴和y轴刻度
ax1.set_xticks(x_pos)
ax1.set_xticklabels(indices, rotation=45, ha='right')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(shape_params)

# 设置标题
ax1.set_title('Sensitivity Analysis of the max GCD Trust Metric - Bar Chart', pad=25, fontsize=14, fontweight='bold')

# 调整视角
ax1.view_init(elev=50, azim=70)

# 设置z轴范围
ax1.set_zlim(0, 0.25)

# 添加图例
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10, framealpha=0.9)

# 调整布局
plt.tight_layout()

# 保存柱状图
save_path1 = r"E:\newManucript\manuscript2\image\信任度量方式敏感性分析_柱状图_GCD.png"
plt.savefig(save_path1, dpi=300, bbox_inches='tight', facecolor='white')

print("柱状图已保存至:", save_path1)

# # 创建第二幅图：折线图（带排序数字）
# fig2 = plt.figure(figsize=(16, 10))
# ax2 = fig2.add_subplot(111, projection='3d')
#
# # 绘制折线 - 每个权重集构成一条折线
# # 距离信任度量
# for i, shape_param in enumerate(shape_params):
#     y = np.full(len(x_pos), i)
#     distances = distance_trust_data[shape_param]
#     ax2.plot(x_pos, y, distances,
#              color=colors['distance'][i], linewidth=3, linestyle='-',
#              marker='o', markersize=8, markerfacecolor='white', markeredgecolor=colors['distance'][i],
#              alpha=0.9, label=f'Distance Trust (α={shape_param})')
#
# # 模糊信息距离信任度量
# for i, shape_param in enumerate(shape_params):
#     y = np.full(len(x_pos), i)
#     fuzzy_distances = fuzzy_distance_trust_data[shape_param]
#     ax2.plot(x_pos, y, fuzzy_distances,
#              color=colors['fuzzy'][i], linewidth=3, linestyle='-',
#              marker='o', markersize=8, markerfacecolor='white', markeredgecolor=colors['fuzzy'][i],
#              alpha=0.9, label=f'Fuzzy Distance Trust (α={shape_param})')
#
# # 非线性度量信任
# for i, shape_param in enumerate(shape_params):
#     y = np.full(len(x_pos), i)
#     nonlinear_distances = nonlinear_trust_data[shape_param]
#     ax2.plot(x_pos, y, nonlinear_distances,
#              color=colors['nonlinear'][i], linewidth=3, linestyle='-',
#              marker='o', markersize=8, markerfacecolor='white', markeredgecolor=colors['nonlinear'][i],
#              alpha=0.9, label=f'Nonlinear Trust (α={shape_param})')
#
# # 添加排序数字标签
# for i, shape_param in enumerate(shape_params):
#     y = np.full(len(x_pos), i)
#
#     # 添加距离信任度量的排序数字
#     distances = distance_trust_data[shape_param]
#     distance_rank = distance_ranks[shape_param]
#     for j, (x, rank) in enumerate(zip(x_pos, distance_rank)):
#         ax2.text(x, y[j], distances[j] + 0.005,
#                  str(rank), ha='center', va='bottom', fontsize=10, fontweight='bold',
#                  color='white', bbox=dict(boxstyle="circle,pad=0.3", facecolor='black', alpha=0.8))
#
#     # 添加模糊信息距离信任度量的排序数字
#     fuzzy_distances = fuzzy_distance_trust_data[shape_param]
#     fuzzy_rank = fuzzy_ranks[shape_param]
#     for j, (x, rank) in enumerate(zip(x_pos, fuzzy_rank)):
#         ax2.text(x, y[j], fuzzy_distances[j] + 0.005,
#                  str(rank), ha='center', va='bottom', fontsize=10, fontweight='bold',
#                  color='white', bbox=dict(boxstyle="circle,pad=0.3", facecolor='black', alpha=0.8))
#
#     # 添加非线性度量信任的排序数字
#     nonlinear_distances = nonlinear_trust_data[shape_param]
#     nonlinear_rank = nonlinear_ranks[shape_param]
#     for j, (x, rank) in enumerate(zip(x_pos, nonlinear_rank)):
#         ax2.text(x, y[j], nonlinear_distances[j] + 0.005,
#                  str(rank), ha='center', va='bottom', fontsize=10, fontweight='bold',
#                  color='white', bbox=dict(boxstyle="circle,pad=0.3", facecolor='black', alpha=0.8))
#
# # 设置坐标轴标签
# ax2.set_xlabel('Index', labelpad=20, fontsize=12, fontweight='bold')
# ax2.set_ylabel('Shape Parameter (α)', labelpad=20, fontsize=12, fontweight='bold')
# ax2.set_zlabel('Weight', labelpad=20, fontsize=12, fontweight='bold')
#
# # 设置x轴和y轴刻度
# ax2.set_xticks(x_pos)
# ax2.set_xticklabels(indices, rotation=70, ha='right')
# ax2.set_yticks(y_pos)
# ax2.set_yticklabels(shape_params)
#
# # 设置标题
# ax2.set_title('Sensitivity Analysis of the min UL Trust Metric - Line Chart', pad=25, fontsize=14, fontweight='bold')
#
# # 调整视角
# ax2.view_init(elev=30, azim=70)
#
# # 设置z轴范围
# ax2.set_zlim(0, 0.25)
#
# # 添加图例
# ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10, framealpha=0.9)
#
# # 调整布局
# plt.tight_layout()
#
# # 保存折线图
# save_path2 = r"E:\newManucript\manuscript2\image\信任度量方式敏感性分析_折线图.png"
# plt.savefig(save_path2, dpi=300, bbox_inches='tight', facecolor='white')
#
# print("折线图已保存至:", save_path2)
#
# # 显示图片
# plt.show()
#
# print("两幅图已分别保存:")
# print(f"1. 柱状图: {save_path1}")
# print(f"2. 折线图(带排序数字): {save_path2}")
'''
#'''
#权重折线图
'''
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 数据定义
shape_params = [0.05, 0.5, 5]
indices = ['Resource', 'Environment', 'External Support', 'Risk', 'Economy', 'Geology']

# 三种信任度量方式的数据
distance_trust_data = {0.05: [0.19450659158313333, 0.21875243856944146, 0.20865709948645275, 0.15738478162504785, 0.11834179377010223, 0.10235729496582249],0.5: [0.19452204605946288, 0.21874437075004283, 0.2086502815493562, 0.1573834285778727, 0.11834234152106239, 0.10235753154220293],5: [0.19464505472235244, 0.21868823670622953, 0.20860750065519018, 0.15736685922477164, 0.11833513535068436, 0.10235721334077186]}
fuzzy_distance_trust_data = {0.05: [0.19892026760327902, 0.21800410627331704, 0.21005888214468885, 0.15522672958043224, 0.11590435607999774, 0.10188565831828521],0.5: [0.19892078133550947, 0.21800663224684713, 0.21006248628514562, 0.1552244939843013, 0.11590069933243208, 0.10188490681576445],5: [0.19892371757371405, 0.21802114005744644, 0.21008320036494674, 0.155211659339982, 0.11587969369580535, 0.1018805889681054]}
nonlinear_trust_data = {0.05: [0.19472244499462799, 0.21875354464615612, 0.20874198034402308, 0.15726423206074414, 0.1181888925795259, 0.10232890537492276],0.5: [0.19474018453108033, 0.21874600885140663, 0.20873653690851182, 0.15726138354398958, 0.1181871668985682, 0.10232871926644328],5: [0.19486757268293667, 0.21869771690117668, 0.20870484657389124, 0.15723745689923663, 0.11816656065653885, 0.1023258462862199]}


# 计算每个数据在其权重集中的排序
def get_rank_labels(data_dict):
    rank_dict = {}
    for shape_param, values in data_dict.items():
        # 对值从大到小排序，获取每个值的排名
        sorted_values = sorted(values, reverse=True)
        ranks = [sorted_values.index(v) + 1 for v in values]
        rank_dict[shape_param] = ranks
    return rank_dict


distance_ranks = get_rank_labels(distance_trust_data)
fuzzy_ranks = get_rank_labels(fuzzy_distance_trust_data)
nonlinear_ranks = get_rank_labels(nonlinear_trust_data)

# 美观的颜色方案 - 同一类信任度量方式使用相近颜色
colors = {
    'distance': ['#1f77b4', '#4c72b0', '#6a6eb5'],  # 蓝色系
    'fuzzy': ['#ff7f0e', '#ff9e4a', '#ffbb78'],  # 橙色系
    'nonlinear': ['#2ca02c', '#59a14f', '#86c67c']  # 绿色系
}

# 坐标轴位置设置
x_pos = np.arange(len(indices))
y_base = np.arange(len(shape_params))

# 创建折线图
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# 设置偏移量，使同一形状参数下的不同方案在y轴方向稍微分开
offset = 0.15

# 绘制折线 - 每个权重集构成一条折线
# 距离信任度量
for i, shape_param in enumerate(shape_params):
    y = np.full(len(x_pos), y_base[i] - offset)  # 在y轴方向偏移
    distances = distance_trust_data[shape_param]
    ax.plot(x_pos, y, distances,
            color=colors['distance'][i], linewidth=3, linestyle='-',
            marker='o', markersize=8, markerfacecolor='white', markeredgecolor=colors['distance'][i],
            alpha=0.9, label=f'Distance Trust (α={shape_param})')

# 模糊信息距离信任度量
for i, shape_param in enumerate(shape_params):
    y = np.full(len(x_pos), y_base[i])  # 不偏移
    fuzzy_distances = fuzzy_distance_trust_data[shape_param]
    ax.plot(x_pos, y, fuzzy_distances,
            color=colors['fuzzy'][i], linewidth=3, linestyle='-',
            marker='o', markersize=8, markerfacecolor='white', markeredgecolor=colors['fuzzy'][i],
            alpha=0.9, label=f'Fuzzy Distance Trust (α={shape_param})')

# 非线性度量信任
for i, shape_param in enumerate(shape_params):
    y = np.full(len(x_pos), y_base[i] + offset)  # 在y轴方向偏移
    nonlinear_distances = nonlinear_trust_data[shape_param]
    ax.plot(x_pos, y, nonlinear_distances,
            color=colors['nonlinear'][i], linewidth=3, linestyle='-',
            marker='o', markersize=8, markerfacecolor='white', markeredgecolor=colors['nonlinear'][i],
            alpha=0.9, label=f'Nonlinear Trust (α={shape_param})')

# 添加排序数字标签
for i, shape_param in enumerate(shape_params):
    # 距离信任度量的排序数字
    distances = distance_trust_data[shape_param]
    distance_rank = distance_ranks[shape_param]
    y = np.full(len(x_pos), y_base[i] - offset)  # 与折线相同的偏移
    for j, (x, rank) in enumerate(zip(x_pos, distance_rank)):
        ax.text(x, y[j], distances[j] + 0.005,
                str(rank), ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='white', bbox=dict(boxstyle="circle,pad=0.3", facecolor='black', alpha=0.8))

    # 模糊信息距离信任度量的排序数字
    fuzzy_distances = fuzzy_distance_trust_data[shape_param]
    fuzzy_rank = fuzzy_ranks[shape_param]
    y = np.full(len(x_pos), y_base[i])  # 不偏移
    for j, (x, rank) in enumerate(zip(x_pos, fuzzy_rank)):
        ax.text(x, y[j], fuzzy_distances[j] + 0.005,
                str(rank), ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='white', bbox=dict(boxstyle="circle,pad=0.3", facecolor='black', alpha=0.8))

    # 非线性度量信任的排序数字
    nonlinear_distances = nonlinear_trust_data[shape_param]
    nonlinear_rank = nonlinear_ranks[shape_param]
    y = np.full(len(x_pos), y_base[i] + offset)  # 与折线相同的偏移
    for j, (x, rank) in enumerate(zip(x_pos, nonlinear_rank)):
        ax.text(x, y[j], nonlinear_distances[j] + 0.005,
                str(rank), ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='white', bbox=dict(boxstyle="circle,pad=0.3", facecolor='black', alpha=0.8))

# 设置坐标轴标签
ax.set_xlabel('Index', labelpad=20, fontsize=12, fontweight='bold')
ax.set_ylabel('Shape Parameter (α)', labelpad=20, fontsize=12, fontweight='bold')
ax.set_zlabel('Weight', labelpad=20, fontsize=12, fontweight='bold')

# 设置x轴和y轴刻度
ax.set_xticks(x_pos)
ax.set_xticklabels(indices, rotation=45, ha='right')
ax.set_yticks(y_base)
ax.set_yticklabels(shape_params)

# 设置标题
ax.set_title('Sensitivity Analysis of the max GCD Trust Metric - Line Chart', pad=25, fontsize=14, fontweight='bold')

# 调整视角
ax.view_init(elev=30, azim=45)

# 设置z轴范围
ax.set_zlim(0, 0.25)

# 添加图例
ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10, framealpha=0.9)

# 调整布局
plt.tight_layout()

# 保存折线图
save_path = r"E:\newManucript\manuscript2\image\信任度量方式敏感性分析_折线图_GCD.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

print("折线图已保存至:", save_path)

# 显示图片
plt.show()

'''
'''
#信任样本度量方式的敏感性分析 min UL效用最优：

import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2

# 数据准备
schemes = ['DTM', 'FIDM', 'NLM']
schemes_full = ['Distance Trust\nMetric', 'Fuzzy Infor\nDistance ', 'Nonlinear\nMetric']
shape_params = [0.05, 0.5, 2.0, 5.0, 10.0]

weights = [
    [0.4855209052215778, 0.24132389503348847, 0.4775060778148357],
    [0.4849660934055529, 0.2414105586548511, 0.47705985727258926],
    [0.4869445394715142, 0.24110017972371642, 0.47861569908967033],
    [0.48863496255104266, 0.24082600049721328, 0.4798471883085196],
    [0.4899679879638681, 0.2405891006941343, 0.480722189320498]
]

# 创建图形
fig = plt.figure(figsize=(18, 14))
ax = fig.add_subplot(111, projection='3d')

# 创建网格
X, Y = np.meshgrid(range(len(schemes)), range(len(shape_params)))
Z = np.array(weights)

# 绘制精细的表面图
surf = ax.plot_surface(X, Y, Z, alpha=0.15, color='steelblue',
                       linewidth=0.25, antialiased=True, shade=True)

# 绘制数据点
for i in range(len(shape_params)):
    for j in range(len(schemes)):
        # 绘制数据点
        ax.scatter(j, i, weights[i][j], color='crimson', s=80, marker='o',
                   edgecolors='darkred', linewidth=2, zorder=5)

        # 添加数据标签
        ax.text(j, i, weights[i][j] + 0.004,
                f'{weights[i][j]:.5f}',
                fontsize=10, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor='white', alpha=0.85,
                          edgecolor='darkgray', linewidth=1))

# 横向对比线（同一形状参数）
horizontal_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i in range(len(shape_params)):
    x_line = list(range(len(schemes)))
    y_line = [i] * len(schemes)
    z_line = weights[i]
    ax.plot(x_line, y_line, z_line, color=horizontal_colors[i],
            linewidth=3, marker='D', markersize=8, markerfacecolor=horizontal_colors[i],
            label=f'Shape={shape_params[i]}', zorder=4)

# 纵向对比线（同一度量方案）
vertical_colors = ['#e377c2', '#7f7f7f', '#bcbd22']
line_styles = ['-', '--', '-.']
for j in range(len(schemes)):
    x_line = [j] * len(shape_params)
    y_line = list(range(len(shape_params)))
    z_line = [weights[i][j] for i in range(len(shape_params))]
    ax.plot(x_line, y_line, z_line, color=vertical_colors[j],
            linewidth=3, linestyle=line_styles[j],
            marker='^', markersize=8, markerfacecolor=vertical_colors[j],
            label=f'{schemes[j]} Trend', zorder=4)

# 坐标轴设置
ax.set_xlabel('度量方案\nMetric Scheme', fontsize=16, labelpad=25, fontweight='bold')
ax.set_ylabel('Shape Parameter', fontsize=16, labelpad=25, fontweight='bold')
ax.set_zlabel('Loss UL', fontsize=16, labelpad=25, fontweight='bold')

# 刻度设置
ax.set_xticks(range(len(schemes)))
ax.set_xticklabels(schemes_full, fontsize=12, rotation=0, ha='center')

ax.set_yticks(range(len(shape_params)))
ax.set_yticklabels(shape_params, fontsize=12)

ax.set_zlim(0.2, 0.5)
ax.tick_params(axis='z', labelsize=12)

# 标题
plt.title('Sensitivity Analysis of the min UL-Loss Trust Metric (UL-Loss)',
          fontsize=18, pad=30, fontweight='bold')

# 视角
ax.view_init(elev=50, azim=60)

# 图例（优化布局）
legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                   fontsize=11, ncol=2, framealpha=0.95,
                   edgecolor='black', facecolor='white')
legend.get_frame().set_linewidth(1.5)

# 网格
ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.8)

# 背景颜色
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

# 调整布局
plt.tight_layout()

# 保存
save_path = r"E:\newManucript\manuscript2\image\信任度量方式敏感性分析（min_UL的效用降低对比）.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"图片已保存至: {save_path}")
plt.show()
'''
''' #绘制fig 16b  上面是绘制fig 16 a
#信任样本度量方式的敏感性分析（群共识最优）效用
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2

# 数据准备
schemes = ['DTM', 'FIDM', 'NLM']
schemes_full = ['Distance Trust\nMetric', 'Fuzzy Infor\nDistance', 'Nonlinear\nMetric']
shape_params = [0.05, 0.5, 2.0, 5.0, 10.0]

# 使用原始数据的精确值
gc_values = [
    [1.288478963504691, 1.2876859156788238, 1.288368799291259],
    [1.2884655332685588, 1.2876831142504768, 1.288353183954277],
    [1.288425387874079, 1.2876758956046905, 1.2883076633192318],
    [1.2883614190829666, 1.287667041016407, 1.2882392646838552],
    [1.2882866123479653, 1.287659371302241, 1.2881669007969347]
]

# 创建图形
fig = plt.figure(figsize=(18, 14))
ax = fig.add_subplot(111, projection='3d')

# 创建网格
X, Y = np.meshgrid(range(len(schemes)), range(len(shape_params)))
Z = np.array(gc_values)

# 绘制精细的表面图
surf = ax.plot_surface(X, Y, Z, alpha=0.1, color='steelblue',
                       linewidth=0.25, antialiased=True, shade=True)

# 绘制数据点
for i in range(len(shape_params)):
    for j in range(len(schemes)):
        # 绘制数据点
        ax.scatter(j, i, gc_values[i][j], color='crimson', s=80, marker='o',
                   edgecolors='darkred', linewidth=2, zorder=5)

        # 添加数据标签
        ax.text(j, i, gc_values[i][j] + 0.0003,
                f'{gc_values[i][j]:.5f}',
                fontsize=10, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor='white', alpha=0.85,
                          edgecolor='darkgray', linewidth=1))

# 横向对比线（同一形状参数）
horizontal_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i in range(len(shape_params)):
    x_line = list(range(len(schemes)))
    y_line = [i] * len(schemes)
    z_line = gc_values[i]
    ax.plot(x_line, y_line, z_line, color=horizontal_colors[i],
            linewidth=3, marker='D', markersize=8, markerfacecolor=horizontal_colors[i],
            label=f'Shape={shape_params[i]}', zorder=4)

# 纵向对比线（同一度量方案）
vertical_colors = ['#e377c2', '#7f7f7f', '#bcbd22']
line_styles = ['-', '--', '-.']
for j in range(len(schemes)):
    x_line = [j] * len(shape_params)
    y_line = list(range(len(shape_params)))
    z_line = [gc_values[i][j] for i in range(len(shape_params))]
    ax.plot(x_line, y_line, z_line, color=vertical_colors[j],
            linewidth=3, linestyle=line_styles[j],
            marker='^', markersize=8, markerfacecolor=vertical_colors[j],
            label=f'{schemes[j]} Trend', zorder=4)

# 坐标轴设置
ax.set_xlabel('trust\nMetric Scheme', fontsize=16, labelpad=25, fontweight='bold')
ax.set_ylabel('Shape Parameter', fontsize=16, labelpad=25, fontweight='bold')
ax.set_zlabel('GC', fontsize=16, labelpad=25, fontweight='bold')

# 刻度设置
ax.set_xticks(range(len(schemes)))
ax.set_xticklabels(schemes_full, fontsize=12, rotation=0, ha='center')

ax.set_yticks(range(len(shape_params)))
ax.set_yticklabels(shape_params, fontsize=12)

# 由于GC值非常接近，我们设置一个合适的z轴范围来突出差异
z_min = min([min(row) for row in gc_values]) - 0.00001
z_max = max([max(row) for row in gc_values]) + 0.00001
ax.set_zlim(z_min, z_max)

# 格式化z轴刻度显示更多小数位
from matplotlib.ticker import FormatStrFormatter

ax.zaxis.set_major_formatter(FormatStrFormatter('%.5f'))
ax.tick_params(axis='z', labelsize=11)

# 标题
plt.title('Sensitivity Analysis of the max GC Trust Metric (UL-loss)',
          fontsize=18, pad=30, fontweight='bold')

# 视角
ax.view_init(elev=50, azim=60)
#ax.view_init(elev=65, azim=65)   更好的视野

# 图例（优化布局）
legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                   fontsize=11, ncol=2, framealpha=0.95,
                   edgecolor='black', facecolor='white')
legend.get_frame().set_linewidth(1.5)

# 网格
ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.8)

# 背景颜色
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

# 调整布局
plt.tight_layout()

# 保存
save_path = r"E:\newManucript\manuscript2\image\信任度量方式敏感性分析（max_GC的群共识对比）.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"图片已保存至: {save_path}")
plt.show()
'''

        #       群共识度对比
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2

# 数据准备
schemes = ['DTM', 'FIDM', 'NLM']
schemes_full = ['Distance Trust\nMetric', 'Fuzzy Information\nDistance Metric', 'Nonlinear\nMetric']
shape_params = [0.05, 0.5, 2.0, 5.0, 10.0]

# 使用原始数据的精确值
ul_loss_values = [
    [0.942680578565609, 0.9416541789761585, 0.942424452649152],
    [0.9426828428893832, 0.941652205696721, 0.9424257910584898],
    [0.9426885240539494, 0.941647112717567, 0.9424288136469118],
    [0.9426948023684483, 0.9416408683849182, 0.9424311914080914],
    [0.942698766097316, 0.9416354691359612, 0.9424311535851775]
]

# 创建图形
fig = plt.figure(figsize=(18, 14))
ax = fig.add_subplot(111, projection='3d')

# 创建网格
X, Y = np.meshgrid(range(len(schemes)), range(len(shape_params)))
Z = np.array(ul_loss_values)

# 绘制精细的表面图
surf = ax.plot_surface(X, Y, Z, alpha=0.1, color='lightcoral',
                       linewidth=0.25, antialiased=True, shade=True)

# 绘制数据点
for i in range(len(shape_params)):
    for j in range(len(schemes)):
        # 绘制数据点
        ax.scatter(j, i, ul_loss_values[i][j], color='blue', s=80, marker='o',
                   edgecolors='darkblue', linewidth=2, zorder=5)

        # 添加数据标签（使用90%不透明度的白色背景）
        ax.text(j, i, ul_loss_values[i][j] + 0.0003,
                f'{ul_loss_values[i][j]:.5f}',
                fontsize=10, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor='white', alpha=0.9,
                          edgecolor='darkgray', linewidth=1))

# 横向对比线（同一形状参数）
horizontal_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i in range(len(shape_params)):
    x_line = list(range(len(schemes)))
    y_line = [i] * len(schemes)
    z_line = ul_loss_values[i]
    ax.plot(x_line, y_line, z_line, color=horizontal_colors[i],
            linewidth=3, marker='D', markersize=8, markerfacecolor=horizontal_colors[i],
            label=f'Shape={shape_params[i]}', zorder=4)

# 纵向对比线（同一度量方案）
vertical_colors = ['#e377c2', '#7f7f7f', '#bcbd22']
line_styles = ['-', '--', '-.']
for j in range(len(schemes)):
    x_line = [j] * len(shape_params)
    y_line = list(range(len(shape_params)))
    z_line = [ul_loss_values[i][j] for i in range(len(shape_params))]
    ax.plot(x_line, y_line, z_line, color=vertical_colors[j],
            linewidth=3, linestyle=line_styles[j],
            marker='^', markersize=8, markerfacecolor=vertical_colors[j],
            label=f'{schemes[j]} Trend', zorder=4)

# 坐标轴设置
ax.set_xlabel('\nMetric Scheme', fontsize=16, labelpad=25, fontweight='bold')
ax.set_ylabel('Shape Parameter', fontsize=16, labelpad=25, fontweight='bold')
ax.set_zlabel('UL-Loss', fontsize=16, labelpad=25, fontweight='bold')

# 刻度设置
ax.set_xticks(range(len(schemes)))
ax.set_xticklabels(schemes_full, fontsize=12, rotation=0, ha='center')

ax.set_yticks(range(len(shape_params)))
ax.set_yticklabels(shape_params, fontsize=12)

# 设置z轴范围以突出差异
z_min = min([min(row) for row in ul_loss_values]) - 0.002
z_max = max([max(row) for row in ul_loss_values]) + 0.002
ax.set_zlim(z_min, z_max)

# 格式化z轴刻度显示更多小数位
from matplotlib.ticker import FormatStrFormatter

ax.zaxis.set_major_formatter(FormatStrFormatter('%.5f'))
ax.tick_params(axis='z', labelsize=11)

# 标题
plt.title('Sensitivity Analysis of the max GC Trust Metric (GCD)',
          fontsize=18, pad=30, fontweight='bold')

# 视角
ax.view_init(elev=28, azim=135)

# 图例（优化布局）
legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                   fontsize=11, ncol=2, framealpha=0.95,
                   edgecolor='black', facecolor='white')
legend.get_frame().set_linewidth(1.5)

# 网格
ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.8)

# 背景颜色
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

# 调整布局
plt.tight_layout()

# 保存
save_path = r"E:\newManucript\manuscript2\image\信任度量方式敏感性分析（max_GC的UL-Loss对比）.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"图片已保存至: {save_path}")
plt.show()

'''
        #       群共识度对比
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据准备
schemes = ['DTM', 'FIDM', 'NLM']
schemes_full = ['Distance Trust\nMetric', 'Fuzzy Information\nDistance Metric', 'Nonlinear\nMetric']
shape_params = [0.05, 0.5, 2.0, 5.0, 10.0]

# 使用原始数据的精确值
ul_loss_values = [
    [0.9331491845085617, 0.925372567058102, 0.9315602091978158],
    [0.9332556104583778, 0.9253547995564898, 0.9316448853758093],
    [0.9335278005404432, 0.9253089317818347, 0.9318545084098539],
    [0.9338459560354132, 0.9252527154841272, 0.9320848769927177],
    [0.9340919196574352, 0.9252041420287003, 0.9322459335855665]
]

# 创建图形
fig = plt.figure(figsize=(18, 14))
ax = fig.add_subplot(111, projection='3d')

# 创建网格
X, Y = np.meshgrid(range(len(schemes)), range(len(shape_params)))
Z = np.array(ul_loss_values)

# 绘制精细的表面图
surf = ax.plot_surface(X, Y, Z, alpha=0.1, color='lightcoral',
                       linewidth=0.25, antialiased=True, shade=True)


# 横向对比线（同一形状参数）
horizontal_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i in range(len(shape_params)):
    x_line = list(range(len(schemes)))
    y_line = [i] * len(schemes)
    z_line = ul_loss_values[i]
    ax.plot(x_line, y_line, z_line, color=horizontal_colors[i],
            linewidth=3, marker='D', markersize=8, markerfacecolor=horizontal_colors[i],
            label=f'Shape={shape_params[i]}', zorder=4)

# 纵向对比线（同一度量方案）
vertical_colors = ['#e377c2', '#7f7f7f', '#bcbd22']
line_styles = ['-', '--', '-.']
for j in range(len(schemes)):
    x_line = [j] * len(shape_params)
    y_line = list(range(len(shape_params)))
    z_line = [ul_loss_values[i][j] for i in range(len(shape_params))]
    ax.plot(x_line, y_line, z_line, color=vertical_colors[j],
            linewidth=3, linestyle=line_styles[j],
            marker='^', markersize=8, markerfacecolor=vertical_colors[j],
            label=f'{schemes[j]} Trend', zorder=4)

# 绘制数据点
for i in range(len(shape_params)):
    for j in range(len(schemes)):
        # 绘制数据点
        ax.scatter(j, i, ul_loss_values[i][j], color='blue', s=80, marker='o',
                   edgecolors='darkblue', linewidth=2, zorder=5)

        # 添加数据标签（使用90%不透明度的白色背景）
        ax.text(j, i, ul_loss_values[i][j] + 0.0003,
                f'{ul_loss_values[i][j]:.5f}',
                fontsize=10, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor='white', alpha=0.9,
                          edgecolor='darkgray', linewidth=1))


# 坐标轴设置
ax.set_xlabel('\nMetric Scheme', fontsize=16, labelpad=25, fontweight='bold')
ax.set_ylabel('Shape Parameter', fontsize=16, labelpad=25, fontweight='bold')
ax.set_zlabel('UL-Loss', fontsize=16, labelpad=25, fontweight='bold')

# 刻度设置
ax.set_xticks(range(len(schemes)))
ax.set_xticklabels(schemes_full, fontsize=12, rotation=0, ha='center')

ax.set_yticks(range(len(shape_params)))
ax.set_yticklabels(shape_params, fontsize=12)

# 设置z轴范围以突出差异
z_min = min([min(row) for row in ul_loss_values]) - 0.002
z_max = max([max(row) for row in ul_loss_values]) + 0.002
ax.set_zlim(z_min, z_max)

# 格式化z轴刻度显示更多小数位
from matplotlib.ticker import FormatStrFormatter

ax.zaxis.set_major_formatter(FormatStrFormatter('%.5f'))
ax.tick_params(axis='z', labelsize=11)

# 标题
plt.title('Sensitivity Analysis of the min UL-loss Trust Metric (GCD)',
          fontsize=18, pad=30, fontweight='bold')

# 视角
ax.view_init(elev=28, azim=-80)

# 图例（优化布局）
legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                   fontsize=11, ncol=2, framealpha=0.95,
                   edgecolor='black', facecolor='white')
legend.get_frame().set_linewidth(1.5)

# 网格
ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.8)

# 背景颜色
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

# 调整布局
plt.tight_layout()

# 保存
save_path = r"E:\newManucript\manuscript2\image\fig_19(a).png"
plt.savefig(save_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"图片已保存至: {save_path}")
plt.show()
'''