#min UL FIG9a

import matplotlib

matplotlib.use('Qt5Agg')
'''
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------- 1. 数据准备 --------------------------
gamma_params = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]  # x轴伽马参数
min_ul_data = [0.2414105586548511, 0.241400360025921, 0.24138040811644684,
               0.24132389503348847, 0.241239555537937, 0.24110017972371642,
               0.24082600049721328,0.2405891006941343]  # min UL数据
max_gc_data = [1.2816859156788238, 1.2816855856887106, 1.2816849403932895,
               1.2816831142504768, 1.281680391909416, 1.2816758956046905,
               1.281667041016407, 1.281659371302241]  # max GC数据

# x轴刻度标签：隐藏0.1和0.2
xtick_labels = [f'{g:.2f}' if g not in [0.1, 0.2] else '' for g in gamma_params]

# -------------------------- 2. 绘图参数设置 --------------------------
plt.rcParams['font.sans-serif'] = ['Arial']  # SCI英文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10  # 基础字体大小
plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签大小
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
plt.rcParams['legend.fontsize'] = 10  # 图例字体大小
plt.rcParams['figure.figsize'] = (12, 7)  # 加宽加高图幅，适配偏移标注

# -------------------------- 3. 创建画布与绘制折线 --------------------------
fig, ax = plt.subplots()

# 绘制min UL折线（蓝色实线+圆形标记）
line1, = ax.plot(
    gamma_params, min_ul_data,
    color='#2E86AB', linestyle='-', linewidth=2,
    marker='o', markersize=6, markerfacecolor='white',
    markeredgecolor='#2E86AB', markeredgewidth=1.5,
    label='min UL-loss'
)

# 绘制max GC折线（红色虚线+方形标记）
line2, = ax.plot(
    gamma_params, max_gc_data,
    color='#A23B72', linestyle='--', linewidth=2,
    marker='s', markersize=6, markerfacecolor='white',
    markeredgecolor='#A23B72', markeredgewidth=1.5,
    label='max GCD'
)


# -------------------------- 4. 按规则标注（含第1点左移+箭头指向） --------------------------
def get_offsets(point_idx):
    """根据点索引计算x/y轴偏移：1字符≈x轴0.02、y轴0.0012（校准后）"""
    char_h = 0.0012  # y轴1字符高度
    char_w = 0.02  # x轴1字符宽度（适配向左偏移40字符）

    # y轴偏移（按原规则）
    if point_idx == 0:
        y_off = 60 * char_h  # 第1点：上50字符
    elif point_idx == 1:
        y_off = -60 * char_h  # 第2点：下50字符
    elif point_idx == 2:
        y_off = 120 * char_h  # 第3点：上100字符
    elif point_idx == 3:
        y_off = -100 * char_h  # 第4点：下100字符
    else:
        y_off = 60 * char_h  # 第5-8点：上50字符

    # x轴偏移（仅第1点左移40字符，其余0）
    x_off = -4 * char_w if point_idx == 0 else 0

    return x_off, y_off


def annotate_with_offset_arrow(x, y, gamma, color, point_idx):
    """标注文本（含x/y偏移）+ 第3/4点箭头"""
    x_off, y_off = get_offsets(point_idx)
    text = f'a={gamma:.2f}\nUL={y:.4f}'

    # 绘制标注文本（带偏移）
    text_obj = ax.text(
        x + x_off, y + y_off,
        text,
        ha='center', va='center',
        fontsize=7,
        color=color,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none')
    )

    # 第3/4点添加同色箭头（从文本指向数据点）
    if point_idx == 2:  # 第3点：文本在上，箭头向下
        ax.annotate(
            '', xy=(x, y), xytext=(x, y + y_off - 0.01),
            arrowprops=dict(arrowstyle='->', color=color, linewidth=1)
        )
    elif point_idx == 3:  # 第4点：文本在下，箭头向上
        ax.annotate(
            '', xy=(x, y), xytext=(x, y + y_off + 0.01),
            arrowprops=dict(arrowstyle='->', color=color, linewidth=1)
        )
    # 第1点：左移后添加水平箭头（从文本指向数据点）
    elif point_idx == 0:
        ax.annotate(
            '', xy=(x, y), xytext=(x + x_off + 0.05, y + y_off),  # 箭头右向，避开文本
            arrowprops=dict(arrowstyle='->', color=color, linewidth=1)
        )
    elif point_idx == 1:  # 第2点：正下方标注→垂直向上箭头
        ax.annotate(
            '', xy=(x, y), xytext=(x, y + y_off + 0.01),  # 从文本上方指向点
            arrowprops=dict(arrowstyle='->', color=color, linewidth=1)
        )

# 批量标注所有点（分min UL和max GC）
for idx in range(len(gamma_params)):
    # 标注min UL
    annotate_with_offset_arrow(
        gamma_params[idx], min_ul_data[idx], gamma_params[idx], '#2E86AB', idx
    )
    # 标注max GC
    annotate_with_offset_arrow(
        gamma_params[idx], max_gc_data[idx], gamma_params[idx], '#A23B72', idx
    )

# -------------------------- 5. 坐标轴与图例设置 --------------------------
# X轴：隐藏0.1和0.2刻度标签，保留刻度线；扩大x范围适配第1点左移
ax.set_xlabel('Gamma Shape Param. (a)', fontsize=12, fontweight='bold')
ax.set_xticks(gamma_params)
ax.set_xticklabels(xtick_labels)
ax.set_xlim(-0.15, 10.5)  # 左扩0.15，容纳第1点左移标注

# Y轴：扩大范围适配最大y偏移（±0.12）
y_min = min(min(min_ul_data), min(max_gc_data)) - 0.15
y_max = max(max(min_ul_data), max(max_gc_data)) + 0.20
ax.set_ylabel('Utility Loss (UL-loss)', fontsize=12, fontweight='bold')
ax.set_ylim(y_min, y_max)

# 图例：放在图中右侧（不遮挡标注）
ax.legend(
    loc='center left',
    bbox_to_anchor=(0.88, 0.5),
    frameon=True,
    framealpha=1,
    edgecolor='black',
    fancybox=False,
    shadow=False
)

# 网格线与边框优化
ax.yaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# -------------------------- 6. 保存图片 --------------------------
save_dir = r"E:\newManucript\manuscript2\image"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "逆伽马形状参数敏感性分析（min_UL与max GCD损失效用对比）.png")

plt.tight_layout()
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

plt.close(fig)

#min UL FIG9b

'''
#max GCD
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------- 1. 数据准备 --------------------------
gamma_params = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
min_ul_data = [0.925372567058102,0.9253704762060854,0.9253663857724379,0.9253547995564898,0.9253375079547818,0.9253089317818347,0.9252527154841272,0.9252041420287003]
max_gc_data = [ 0.9416541789761585, 0.9416539467281362, 0.94165349240177,
               0.941652205696721, 0.9416502856782176, 0.941647112717567,
               0.9416408683849182, 0.9416354691359612]

# x轴刻度标签：隐藏0.1和0.2
xtick_labels = [f'{g:.2f}' if g not in [0.1, 0.2] else '' for g in gamma_params]

# -------------------------- 2. 绘图参数设置 --------------------------
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (12, 7)

# -------------------------- 3. 创建画布与绘制折线 --------------------------
fig, ax = plt.subplots()

# 绘制min UL折线
line1, = ax.plot(
    gamma_params, min_ul_data,
    color='#2E86AB', linestyle='-', linewidth=2,
    marker='o', markersize=6, markerfacecolor='white',
    markeredgecolor='#2E86AB', markeredgewidth=1.5,
    label='min UL-loss'
)

# 绘制max GC折线
line2, = ax.plot(
    gamma_params, max_gc_data,
    color='#A23B72', linestyle='--', linewidth=2,
    marker='s', markersize=6, markerfacecolor='white',
    markeredgecolor='#A23B72', markeredgewidth=1.5,
    label='max GCD'
)


# -------------------------- 4. 标注与箭头设置 --------------------------
def get_offsets(point_idx):
    """校准偏移量以适应新y轴范围"""
    char_h = 0.0002  # y轴字符高度（适配0.75-0.92范围）
    char_w = 0.03  # x轴字符宽度
    y_off = {0: 10 * char_h, 1: -10 * char_h, 2: 30 * char_h, 3: -30 * char_h}.get(point_idx, 10 * char_h)
    x_off = -4 * char_w if point_idx == 0 else 0
    return x_off, y_off


def annotate_with_arrows(x, y, gamma, color, point_idx):
    x_off, y_off = get_offsets(point_idx)
    text = f'a={gamma:.2f}\nGCD={y:.6f}'

    ax.text(
        x + x_off, y + y_off,
        text,
        ha='center', va='center',
        fontsize=7,
        color=color,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none')
    )

    # 1-4点箭头设置
    if point_idx == 0:
        ax.annotate('', xy=(x, y), xytext=(x + x_off + 0.02, y + y_off),
                    arrowprops=dict(arrowstyle='->', color=color, linewidth=1))
    elif point_idx == 1:
        ax.annotate('', xy=(x, y), xytext=(x, y + y_off + 0.0005),
                    arrowprops=dict(arrowstyle='->', color=color, linewidth=1))
    elif point_idx == 2:
        ax.annotate('', xy=(x, y), xytext=(x, y + y_off - 0.0005),
                    arrowprops=dict(arrowstyle='->', color=color, linewidth=1))
    elif point_idx == 3:
        ax.annotate('', xy=(x, y), xytext=(x, y + y_off + 0.001),
                    arrowprops=dict(arrowstyle='->', color=color, linewidth=1))


# 批量标注
for idx in range(len(gamma_params)):
    annotate_with_arrows(gamma_params[idx], min_ul_data[idx], gamma_params[idx], '#2E86AB', idx)
    annotate_with_arrows(gamma_params[idx], max_gc_data[idx], gamma_params[idx], '#A23B72', idx)

# -------------------------- 5. 坐标轴设置（核心调整） --------------------------
# X轴设置
ax.set_xlabel('Gamma Shape Param. (a)', fontsize=12, fontweight='bold')
ax.set_xticks(gamma_params)
ax.set_xticklabels(xtick_labels)
ax.set_xlim(-0.05, 10.5)

# Y轴核心调整：范围[0.75,0.92]，但仅显示0.8~0.9刻度
ax.set_ylim(0.915, 0.95)  # 实际绘图范围
ax.set_yticks(np.arange(0.92, 0.95, 0.005))  # 仅显示0.8,0.82,...,0.9
ax.set_yticklabels([f'{y:.1f}' if y in [0.8, 0.9] else f'{y:.3f}' for y in np.arange(0.92, 0.95, 0.005)])
ax.set_ylabel('Group Consensus Degree (GCD)', fontsize=12, fontweight='bold')

# 图例设置（图中右侧）
ax.legend(
    loc='center left',
    bbox_to_anchor=(0.88, 0.5),
    frameon=True,
    framealpha=1,
    edgecolor='black',
    fancybox=False
)

# 网格线与边框
ax.yaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# -------------------------- 6. 保存图片 --------------------------
save_dir = r"E:\newManucript\manuscript2\image"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "逆伽马形状参数敏感性分析（min_UL与max GCD群共识对比）.png")

plt.tight_layout()
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

plt.close(fig)
