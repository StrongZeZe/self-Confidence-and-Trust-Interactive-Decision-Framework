
#绘制第一幅CI折线图
'''
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 准备数据
gamma_params = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
min_ul_loss = [0.9463308452489689, 0.9463307746325287, 0.9463306365019414,
               0.9463302453653132, 0.946329661857571, 0.9463286978362072,
               0.9463268012310063, 0.9463251617594679]
max_gcd = [0.9484289845939062, 0.9484289040433225, 0.9484287464804758,
           0.9484283003157418, 0.9484276347077074, 0.9484265350267306,
           0.9484243714543438, 0.948422501134685]

# 创建图形和轴
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制min UL-loss折线（使用指定样式）
line1, = ax.plot(
    gamma_params, min_ul_loss,
    color='#2E86AB', linestyle='-', linewidth=2,
    marker='o', markersize=6, markerfacecolor='white',
    markeredgecolor='#2E86AB', markeredgewidth=1.5,
    label='min UL-loss'
)

# 绘制max GCD折线（使用指定样式）
line2, = ax.plot(
    gamma_params, max_gcd,
    color='#A23B72', linestyle='--', linewidth=2,
    marker='s', markersize=6, markerfacecolor='white',
    markeredgecolor='#A23B72', markeredgewidth=1.5,
    label='max GCD'
)

# 设置坐标轴标签
ax.set_xlabel('Gamma Shape Param. (a)', fontsize=14, fontweight='bold')
ax.set_ylabel('consistency index (CI)', fontsize=14, fontweight='bold')

# 设置x轴刻度，不显示0.1和0.2
xticks = [0.05, 0.5, 1.0, 2.0, 5.0, 10.0]
ax.set_xticks(xticks)
ax.set_xticklabels([str(tick) for tick in xticks], fontsize=12)

# 设置y轴区间，增加底部留白
ax.set_ylim(0.9445, 0.9495)
ax.set_yticklabels([f'{y:.4f}' for y in ax.get_yticks()], fontsize=12)

# 仅保留x轴和y轴，去除上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置图例位置在图内右上角，不遮挡图形
ax.legend(loc='upper right', fontsize=12, frameon=False, bbox_to_anchor=(0.95, 0.95))

# 添加y轴网格线
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 计算偏移量
y_offset_small = 0.00025
y_offset_large = 0.0005
x_offset = 0.04

# 定义箭头样式（匹配线条风格）
arrowprops_min = dict(arrowstyle='->', facecolor='#2E86AB', linewidth=1.5, shrinkA=0, shrinkB=3)
arrowprops_max = dict(arrowstyle='->', facecolor='#A23B72', linewidth=1.5, shrinkA=0, shrinkB=3)


# 添加数据点标签和箭头
def add_labels_and_arrows(x_data, y_data, color, arrowprops, ax):
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        # 格式化标签
        label = f'a={x:.2f}\nCI={y:.6f}'

        # 标签背景框
        bbox = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9)

        # 根据规则设置标签位置
        if i == 0:  # 第一个点
            xytext = (x - x_offset, y + y_offset_small)
            ax.annotate(label, xy=(x, y), xytext=xytext,
                        arrowprops=arrowprops,
                        fontsize=10, color=color, ha='right', bbox=bbox)
        elif i == 1:  # 第二个点
            xytext = (x, y - y_offset_small)
            ax.annotate(label, xy=(x, y), xytext=xytext,
                        arrowprops=arrowprops,
                        fontsize=10, color=color, ha='center', bbox=bbox)
        elif i == 2:  # 第三个点
            xytext = (x, y + y_offset_large)
            ax.annotate(label, xy=(x, y), xytext=xytext,
                        arrowprops=arrowprops,
                        fontsize=10, color=color, ha='center', bbox=bbox)
        elif i == 3:  # 第四个点
            xytext = (x, y - y_offset_large)
            ax.annotate(label, xy=(x, y), xytext=xytext,
                        arrowprops=arrowprops,
                        fontsize=10, color=color, ha='center', bbox=bbox)
        else:  # 其余点
            xytext = (x, y + y_offset_small)
            ax.annotate(label, xy=(x, y), xytext=xytext,
                        fontsize=10, color=color, ha='center', bbox=bbox)


# 为两条线添加标签和箭头（箭头颜色匹配线条）
add_labels_and_arrows(gamma_params, min_ul_loss, '#2E86AB', arrowprops_min, ax)
add_labels_and_arrows(gamma_params, max_gcd, '#A23B72', arrowprops_max, ax)

# 调整布局
plt.tight_layout()

# 保存图片
save_path = r"E:\newManucript\manuscript2\image\逆伽马形状参数敏感性分析（min_UL与max GCD的CI对比）.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# 不显示图片
plt.close()
'''

#绘制第二幅CI折线图
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 准备新数据
# min UL-loss数据
gamma_min_ul = [0.001,  0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
min_ul_loss = [0.9464042450779506,  0.9464011584737456,
               0.946372985638959, 0.9463545563178432, 0.9463432841709223,
               0.9463356995839509, 0.9463302453653132, 0.9463261322346301,
               0.9463229183771101, 0.9463203371701274, 0.9463182181008344,
               0.9463166109106567]

# max GCD数据
gamma_max_gcd = [0.0001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
max_gcd = [0.9485126519156863, 0.9485088814305989, 0.9484762252368601,
           0.9484553944215504, 0.9484427822277336, 0.9484343447528077,
           0.9484283003157418, 0.9484237545373111, 0.9484202099405009,
           0.9484173676880789, 0.94841503733473, 0.9484132717017788]

# 创建图形和轴
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制min UL-loss折线（使用指定样式）
line1, = ax.plot(
    gamma_min_ul, min_ul_loss,
    color='#2E86AB', linestyle='-', linewidth=2,
    marker='o', markersize=6, markerfacecolor='white',
    markeredgecolor='#2E86AB', markeredgewidth=1.5,
    label='min UL-loss'
)

# 绘制max GCD折线（使用指定样式）
line2, = ax.plot(
    gamma_max_gcd, max_gcd,
    color='#A23B72', linestyle='--', linewidth=2,
    marker='s', markersize=6, markerfacecolor='white',
    markeredgecolor='#A23B72', markeredgewidth=1.5,
    label='max GCD'
)

# 设置坐标轴标签
ax.set_xlabel('Gamma Shape Param. (a)', fontsize=14, fontweight='bold')
ax.set_ylabel('consistency index (CI)', fontsize=14, fontweight='bold')

# 设置x轴刻度（选择关键值显示，避免过于密集）
xticks = [0.0001,  0.1, 0.2, 0.5, 0.99]
ax.set_xticks(xticks)
ax.set_xticklabels([str(tick) for tick in xticks], fontsize=12)

# 设置y轴范围（根据新数据调整，增加底部留白）
ax.set_ylim(0.946, 0.949)
ax.set_yticklabels([f'{y:.4f}' for y in ax.get_yticks()], fontsize=12)

# 仅保留x轴和y轴，去除上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置图例位置在图内右上角，不遮挡图形
ax.legend(loc='upper right', fontsize=12, frameon=False, bbox_to_anchor=(0.95, 0.95))

# 添加y轴网格线
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 计算偏移量（根据新数据范围调整）
y_offset_small = 0.00015
y_offset_large = 0.0003
x_offset = 0.02  # x方向偏移减小，适应更小的x值范围

# 定义箭头样式（匹配线条风格）
arrowprops_min = dict(arrowstyle='->', facecolor='#2E86AB', linewidth=1.5, shrinkA=0, shrinkB=3)
arrowprops_max = dict(arrowstyle='->', facecolor='#A23B72', linewidth=1.5, shrinkA=0, shrinkB=3)


# 添加数据点标签和箭头
def add_labels_and_arrows(x_data, y_data, color, arrowprops, ax):
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        # 格式化标签（根据x值大小调整显示格式）
        if x < 0.01:
            label = f'a={x:.4f}\nCI={y:.6f}'
        else:
            label = f'a={x:.2f}\nCI={y:.6f}'

        # 标签背景框
        bbox = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9)

        # 根据规则设置标签位置
        if i == 0:  # 第一个点
            xytext = (x + 0.38*x_offset * 5, y + y_offset_small)  # 增加x偏移适应小值
            ax.annotate(label, xy=(x, y), xytext=xytext,
                        arrowprops=arrowprops,
                        fontsize=10, color=color, ha='right', bbox=bbox)
        elif i == 1:  # 第二个点
            xytext = (x, y - 2*y_offset_small)
            ax.annotate(label, xy=(x, y), xytext=xytext,
                        arrowprops=arrowprops,
                        fontsize=10, color=color, ha='center', bbox=bbox)
        elif i == 2:  # 第三个点
            xytext = (x, y + y_offset_large)
            ax.annotate(label, xy=(x, y), xytext=xytext,
                        arrowprops=arrowprops,
                        fontsize=10, color=color, ha='center', bbox=bbox)
        elif i == 3:  # 第四个点
            xytext = (x, y - y_offset_large)
            ax.annotate(label, xy=(x, y), xytext=xytext,
                        arrowprops=arrowprops,
                        fontsize=10, color=color, ha='center', bbox=bbox)
        else:  # 其余点
            xytext = (x, y + y_offset_small)
            ax.annotate(label, xy=(x, y), xytext=xytext,
                        fontsize=10, color=color, ha='center', bbox=bbox)


# 为两条线添加标签和箭头（箭头颜色匹配线条）
add_labels_and_arrows(gamma_min_ul, min_ul_loss, '#2E86AB', arrowprops_min, ax)
add_labels_and_arrows(gamma_max_gcd, max_gcd, '#A23B72', arrowprops_max, ax)

# 图例：放在图中右边（不遮挡数据）
ax.legend(
    loc='center left',
    bbox_to_anchor=(0.93, 0.5),  # 右移至图中右侧区域
    frameon=True,
    framealpha=1,
    edgecolor='black',
    fancybox=False,  # 方形图例（SCI简洁风格）
    shadow=False
)

# 调整布局
plt.tight_layout()



# 保存图片
save_path = r"E:\newManucript\manuscript2\image\平衡参数敏感性分析（min_UL与max GCD的CI对比）.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# 不显示图片
plt.close()