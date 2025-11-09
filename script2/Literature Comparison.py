#指标排序
'''
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# 设置后端和字体
matplotlib.use('Qt5Agg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# Index数据
index_data = {
    'Wang et al': [0.165, 0.221, 0.192, 0.148, 0.134, 0.140],
    'Dai et al': [0.112, 0.279, 0.250, 0.144, 0.093, 0.122],
    'You et al': [0.121, 0.271, 0.252, 0.147, 0.092, 0.117],
    'This Paper': [0.199, 0.218, 0.210, 0.155, 0.116, 0.102]
}

index_categories = ['Resource', 'Environment', 'External\nsupport', 'Risk', 'Economy', 'Geology']

# 颜色配置
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# 创建图形
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 反转数据顺序，使先读取的数据绘制在y轴最后
reversed_methods = list(index_data.keys())[::-1]
reversed_colors = colors[::-1]

# 为每个方法创建一条折线（按反转后的顺序）
for i, method in enumerate(reversed_methods):
    values = index_data[method]
    x = np.arange(len(index_categories))
    y = np.full(len(index_categories), len(index_data) - 1 - i)  # 反转y轴位置
    z = np.array(values)

    # 绘制折线
    ax.plot(x, y, z, 'o-', color=reversed_colors[i], linewidth=2.5,
            markersize=10, markerfacecolor=reversed_colors[i], markeredgecolor='white',
            markeredgewidth=1.5, label=method)

    # 计算每个值在其权重集中的排序（从大到小）
    sorted_indices = np.argsort(values)[::-1]
    ranks = np.zeros_like(sorted_indices)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank + 1

    # 添加排名数字
    for j, (x_val, y_val, z_val, rank) in enumerate(zip(x, y, z, ranks)):
        ax.text(x_val, y_val, z_val + 0.008, str(rank),
                fontsize=10, ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor=reversed_colors[i], alpha=0.9))

# 设置坐标轴标签
ax.set_xlabel('Index', fontsize=14, fontweight='bold', labelpad=20)
ax.set_ylabel('Research Methods', fontsize=14, fontweight='bold', labelpad=15)
ax.set_zlabel('Score', fontsize=14, fontweight='bold', labelpad=15)

# 设置刻度
ax.set_xticks(np.arange(len(index_categories)))
ax.set_xticklabels(index_categories, rotation=45, ha='right')
ax.set_yticks(np.arange(len(index_data)))
ax.set_yticklabels(reversed_methods)  # 使用反转后的方法名称

# 调整x轴刻度标签位置
ax.tick_params(axis='x', pad=-4)  # 按照要求设置为-4

# 设置网格和背景
ax.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# 设置标题和视角
ax.set_title('Literature Comparison - Index', fontsize=16, fontweight='bold', pad=20)
ax.view_init(elev=25, azim=-45)

# 添加图例（保持原始顺序）
ax.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=True,
          fancybox=True, shadow=True, fontsize=12)

# 调整布局
plt.tight_layout()

# 保存图片
save_path = "E:\\newManucript\\manuscript2\\image\\论文对比-Index.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Index对比图已保存至: {save_path}")

plt.show()


'''
#方案排序
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# 设置后端和字体
matplotlib.use('Qt5Agg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# Scheme数据
scheme_data = {
    'Wang et al': [0.1678, 0.1645, 0.1637, 0.1683, 0.1671],
    'Dai et al': [0.1698, 0.1623, 0.1596, 0.1701, 0.1702],
    'You et al': [0.1697, 0.1626, 0.1598, 0.1704, 0.1697],
    'This Paper': [0.1683, 0.1644, 0.1631, 0.1714, 0.1652]
}

scheme_categories = ['L1', 'L2', 'L3', 'L4', 'L5']

# 颜色配置
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# 创建图形
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 反转数据顺序，使先读取的数据绘制在y轴最后
reversed_methods = list(scheme_data.keys())[::-1]
reversed_colors = colors[::-1]

# 为每个方法创建一条折线（按反转后的顺序）
for i, method in enumerate(reversed_methods):
    values = scheme_data[method]
    x = np.arange(len(scheme_categories))
    y = np.full(len(scheme_categories), len(scheme_data) - 1 - i)  # 反转y轴位置
    z = np.array(values)

    # 绘制折线
    ax.plot(x, y, z, 'o-', color=reversed_colors[i], linewidth=2.5,
            markersize=10, markerfacecolor=reversed_colors[i], markeredgecolor='white',
            markeredgewidth=1.5, label=method)

    # 计算每个值在其权重集中的排序（从大到小）
    sorted_indices = np.argsort(values)[::-1]
    ranks = np.zeros_like(sorted_indices)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank + 1

    # 添加排名数字
    for j, (x_val, y_val, z_val, rank) in enumerate(zip(x, y, z, ranks)):
        ax.text(x_val, y_val, z_val + 0.0005, str(rank),
                fontsize=10, ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor=reversed_colors[i], alpha=0.9))

# 设置坐标轴标签
ax.set_xlabel('Scheme', fontsize=14, fontweight='bold', labelpad=20)
ax.set_ylabel('Research Methods', fontsize=14, fontweight='bold', labelpad=15)
ax.set_zlabel('Score', fontsize=14, fontweight='bold', labelpad=15)

# 设置刻度
ax.set_xticks(np.arange(len(scheme_categories)))
ax.set_xticklabels(scheme_categories)
ax.set_yticks(np.arange(len(scheme_data)))
ax.set_yticklabels(reversed_methods)  # 使用反转后的方法名称

# 调整x轴刻度标签位置
ax.tick_params(axis='x', pad=-4)  # 按照要求设置为-4

# 设置网格和背景
ax.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# 设置标题和视角
ax.set_title('Literature Comparison - Scheme', fontsize=16, fontweight='bold', pad=20)
ax.view_init(elev=25, azim=-45)

# 添加图例（保持原始顺序）
ax.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=True,
          fancybox=True, shadow=True, fontsize=12)

# 调整布局
plt.tight_layout()

# 保存图片
save_path = "E:\\newManucript\\manuscript2\\image\\论文对比-Scheme.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Scheme对比图已保存至: {save_path}")

plt.show()