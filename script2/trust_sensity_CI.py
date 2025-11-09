import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2

# 数据准备
schemes = ['DTM', 'FIDM', 'NLM']
schemes_full = ['Distance Trust\nMetric', 'Fuzzy Infor\nDistance', 'Nonlinear\nMetric']
shape_params = [0.05, 0.5, 2.0, 5.0, 10.0]

# 使用原始数据的精确值
gc_values = [
    [0.9469897183349849, 0.9463302453653132, 0.9469602211146539],
    [0.9469892432109935, 0.9463308452489689, 0.9469600273776169],
    [0.9469908543486678, 0.9463286978362072, 0.9469606541789344],
    [0.9469920863581534, 0.9463268012310063, 0.9469611597781216],
    [0.9469931969906574, 0.9463251617594679, 0.9469618431150382]
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
        ax.text(j, i, gc_values[i][j] + 0.0003,                 #调整标签位置
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
z_min = min([min(row) for row in gc_values]) - 0.001                  #调整z轴的坐标区域
z_max = max([max(row) for row in gc_values]) + 0.001
ax.set_zlim(z_min, z_max)

# 格式化z轴刻度显示更多小数位
from matplotlib.ticker import FormatStrFormatter

ax.zaxis.set_major_formatter(FormatStrFormatter('%.5f'))
ax.tick_params(axis='z', labelsize=11)

# 标题
plt.title('Sensitivity Analysis of the min UL-loss Trust Metric (CI)',
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
save_path = r"E:\newManucript\manuscript2\image\信任度量方式敏感性分析（UL-LOSS-CI）.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"图片已保存至: {save_path}")
plt.show()