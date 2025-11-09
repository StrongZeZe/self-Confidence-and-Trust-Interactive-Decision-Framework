import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 数据
min_UL_loss = [0.1683, 0.1644, 0.1631, 0.1714, 0.1652]
max_GCD = [0.1682, 0.1646, 0.1633, 0.1710, 0.1652]
schemes = ['L1', 'L2', 'L3', 'L4', 'L5']

# 计算排名 - 两个指标都按照从大到小排序
min_UL_loss_rank = [sorted(min_UL_loss, reverse=True).index(x) + 1 for x in min_UL_loss]
max_GCD_rank = [sorted(max_GCD, reverse=True).index(x) + 1 for x in max_GCD]

# 创建雷达图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, polar=True)

# 设置角度 - 每个方案一个角度
angles = np.linspace(0, 2 * np.pi, len(schemes), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形


# 准备数据 - 将数据归一化到0-1范围以便在雷达图上显示
def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


min_UL_loss_norm = normalize_data(min_UL_loss)
max_GCD_norm = normalize_data(max_GCD)

# 闭合数据
min_UL_loss_norm += min_UL_loss_norm[:1]
max_GCD_norm += max_GCD_norm[:1]

# 绘制雷达图
ax.plot(angles, min_UL_loss_norm, 'o-', linewidth=2, label='min UL-loss', color='#1f77b4')
ax.fill(angles, min_UL_loss_norm, alpha=0.25, color='#1f77b4')
ax.plot(angles, max_GCD_norm, 'o-', linewidth=2, label='max GCD', color='#ff7f0e')
ax.fill(angles, max_GCD_norm, alpha=0.25, color='#ff7f0e')

# 设置角度标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(schemes)

# 设置径向标签
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.164", "0.166", "0.168", "0.170", "0.172"], color="grey", size=10)
plt.ylim(0, 1)

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# 添加标题
plt.title('Performance Comparison of Hydrogen Refueling Locations', size=15, y=1.05)

# 在每个数据点上添加实际值和排名
for i, (angle, ul_val, gcd_val, ul_rank, gcd_rank) in enumerate(zip(
        angles[:-1], min_UL_loss, max_GCD, min_UL_loss_rank, max_GCD_rank)):
    # 对于min UL-loss
    ax.text(angle, min_UL_loss_norm[i] + 0.05,
            f'{ul_val}\n({ul_rank})',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                      edgecolor='#1f77b4', linewidth=1))

    # 对于max GCD
    ax.text(angle, max_GCD_norm[i] - 0.05,
            f'{gcd_val}\n({gcd_rank})',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                      edgecolor='#ff7f0e', linewidth=1))

# 保存图片
save_path = r"E:\newManucript\manuscript2\image\选之地排序雷达图.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='black')
plt.close()

print(f"雷达图已保存至: {save_path}")