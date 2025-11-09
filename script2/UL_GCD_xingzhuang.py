#min UL，max GC 最优效用
#fig 11 a
import matplotlib
matplotlib.use('Qt5Agg')
'''
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------- 1. 数据准备 --------------------------
# 伽马形状参数（x轴）
gamma_params = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
# 最小损失效用数据（min UL 和 max GC）
min_ul_data = [0.2502768531165053,0.24580421384767437,0.24370070898268664,0.24255331718740356, 0.24182663535754828,0.24132389503348847,0.24095494289152328,0.24067244757679185,0.2404491114538232,0.24026806245266225,0.24013210305131139]
max_gc_data = [1.2821062945800852,1.2818902531580392,1.2817887163350479 ,1.2817360676225606 ,1.2817042631519342 ,1.2816831142504768 ,1.281668094596827 ,1.2816569052476985 ,1.2816482612349516 ,1.2816413907296924 ,1.281636314578647]

# -------------------------- 2. 绘图参数设置（SCI格式+旧版本兼容） --------------------------
plt.rcParams['font.sans-serif'] = ['Arial']  # SCI标准英文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10  # 基础字体大小
plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签大小
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
plt.rcParams['legend.fontsize'] = 10  # 图例字体大小
plt.rcParams['figure.figsize'] = (14, 8)  # 加宽加高图幅，适配11个点的标注

# -------------------------- 3. 创建画布与绘制折线 --------------------------
fig, ax = plt.subplots()

# 绘制min UL折线（蓝色实线，圆形标记）
line1, = ax.plot(
    gamma_params, min_ul_data,
    color='#2E86AB',  # 科学配色（蓝色）
    linestyle='-',  # 实线
    linewidth=2,  # 线宽
    marker='o',  # 圆形标记
    markersize=6,  # 标记大小
    markerfacecolor='white',  # 标记填充白
    markeredgecolor='#2E86AB',  # 标记边缘色
    markeredgewidth=1.5,  # 标记边缘宽
    label='min UL-loss'
)

# 绘制max GC折线（红色虚线，方形标记）
line2, = ax.plot(
    gamma_params, max_gc_data,
    color='#A23B72',  # 科学配色（紫红色）
    linestyle='--',  # 虚线
    linewidth=2,  # 线宽
    marker='s',  # 方形标记
    markersize=6,  # 标记大小
    markerfacecolor='white',  # 标记填充白
    markeredgecolor='#A23B72',  # 标记边缘色
    markeredgewidth=1.5,  # 标记边缘宽
    label='max GCD'
)


# -------------------------- 4. 标注所有数据点（点上40字符，无重叠） --------------------------
def add_value_annotation(x_data, y_data, color, label_prefix):
    """
    标注数据点：点上40字符（1字符≈y轴0.0005，40字符≈0.02）
    label_prefix: 标注前缀（如'α='表示平衡参数）
    """
    char_height = 0.0005  # y轴1字符高度（校准后，40字符≈0.02）
    y_offset = 40 * char_height  # 点上40字符的y偏移

    for i in range(len(x_data)):
        x = x_data[i]
        y = y_data[i]
        # 格式化标注文本（参数保留2位小数，UL保留4位小数）
        text = f'{label_prefix}={x:.2f}\nUL={y:.4f}'

        # 横向错开标注（偶数索引右移，奇数索引左移），避免11个点横向重叠
        x_offset = 0.02 if i % 2 == 0 else -0.02

        ax.text(
            x + x_offset, y + y_offset,
            text,
            ha='center', va='bottom',
            fontsize=7,  # 小字体减少占用空间
            color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none')
        )


# 标注min UL（前缀'α'表示平衡参数）
add_value_annotation(gamma_params, min_ul_data, '#2E86AB', 'l')
# 标注max GC（前缀'α'，y偏移稍大，避免与min UL标注重叠）
add_value_annotation(gamma_params, max_gc_data, '#A23B72', 'l')

# -------------------------- 5. 坐标轴与图例设置 --------------------------
# X轴（伽马形状参数，英文缩写）
ax.set_xlabel('Balancing Param. (l)', fontsize=12, fontweight='bold')
ax.set_xticks(gamma_params)
# 刻度标签保留2位小数（适配0.01、0.99等）
ax.set_xticklabels([f'{g:.2f}' for g in gamma_params], rotation=45)

# Y轴（损失效用UL）：扩大范围容纳标注（上下各留0.03）
y_min = min(min(min_ul_data), min(max_gc_data)) - 0.03
y_max = max(max(min_ul_data), max(max_gc_data)) + 0.05  # 上扩更多，容纳点上标注
ax.set_ylabel('Utility Loss (UL-loss)', fontsize=12, fontweight='bold')
ax.set_ylim(y_min, y_max)

# 图例：放在图中右边（不贴边，不遮挡数据）
ax.legend(
    loc='center left',
    bbox_to_anchor=(0.92, 0.5),  # 右移至图中右侧区域
    frameon=True,
    framealpha=1,
    edgecolor='black',
    fancybox=False,  # 方形图例（SCI简洁风格）
    shadow=False
)

# 网格线（仅Y轴，虚线，增强可读性）
ax.yaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
ax.set_axisbelow(True)  # 网格线在折线下方

# 去除顶部和右侧边框（SCI简洁风格）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# -------------------------- 6. 保存图片 --------------------------
save_dir = r"E:\newManucript\manuscript2\image"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 路径不存在则创建
save_path = os.path.join(save_dir, "平衡参数敏感性分析（min_UL与max GCD损失效用对比）.png")

# 高分辨率保存（SCI投稿标准）
plt.tight_layout()  # 紧凑布局，避免标注/图例被截断
plt.savefig(
    save_path,
    dpi=300,  # 300dpi（投稿最低标准）
    bbox_inches='tight',  # 防止标注/图例被截断
    facecolor='white',  # 白色背景（避免打印问题）
    edgecolor='none'
)

plt.close(fig)  # 不显示图片

'''

##fig 11 b
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------- 1. 数据准备 --------------------------
# 伽马形状参数（x轴）
gamma_params = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
# 损失效用数据（min UL恒为0.8，max GC为给定值）
min_ul_data = [0.9271840474932456,0.9262693404512222,0.925839699911692,0.9256055307920087,0.9254573049525167,0.9253547995564898,0.9252795962979582,0.9252220298621912,0.9251765280552563,0.9251396479304453,0.9251119564492924]  # 11个参数对应相同UL值
max_gc_data = [0.9418625922412626 ,0.9417602708735675 , 0.941709578799974 , 0.9416818070946431 ,0.9416642776125039 ,0.941652205696721 , 0.9416433854259497 ,0.9416366586673519 ,0.9416313589782756 ,0.9416270756899022 , 0.9416238672393619 ]

# -------------------------- 2. 绘图参数设置（SCI格式+旧版本兼容） --------------------------
plt.rcParams['font.sans-serif'] = ['Arial']  # SCI标准英文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10  # 基础字体大小
plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签大小
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
plt.rcParams['legend.fontsize'] = 10  # 图例字体大小
plt.rcParams['figure.figsize'] = (14, 8)  # 加宽加高图幅，适配11个点标注

# -------------------------- 3. 创建画布与绘制折线 --------------------------
fig, ax = plt.subplots()

# 绘制min UL折线（蓝色实线+圆形标记，恒值线）
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


# -------------------------- 4. 标注所有数据点（点上40字符，无重叠） --------------------------
def add_annotation(x_data, y_data, color, param_prefix):
    """
    标注逻辑：点上40字符（新数据UL范围小，1字符≈0.0001，40字符≈0.004）
    param_prefix: 参数前缀（如'α'表示平衡参数）
    """
    char_height = 0.0001  # y轴1字符高度（适配0.8-0.882的小范围）
    y_offset = 20 * char_height  # 点上40字符的y偏移（≈0.004）

    for i in range(len(x_data)):
        x = x_data[i]
        y = y_data[i]
        # 格式化文本：参数保留2位小数，UL保留6位小数（适配max GC的精度）
        text = f'{param_prefix}={x:.2f}\nGCD={y:.6f}'

        # 横向错开（偶数索引右移，奇数左移），避免11个点重叠
        x_offset = 0.02 if i % 2 == 0 else -0.02

        ax.text(
            x + x_offset, y + y_offset,
            text,
            ha='center', va='bottom',
            fontsize=7,  # 小字体减少占用
            color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none')
        )


# 标注min UL（平衡参数前缀'α'）
add_annotation(gamma_params, min_ul_data, '#2E86AB', 'l')
# 标注max GC（y偏移稍大，避免与min UL标注重叠）
add_annotation(gamma_params, max_gc_data, '#A23B72', 'l')

# -------------------------- 5. 坐标轴与图例设置 --------------------------
# X轴（伽马形状参数，英文缩写）
ax.set_xlabel('Balancing Param. (l)', fontsize=12, fontweight='bold')
ax.set_xticks(gamma_params)
ax.set_xticklabels([f'{g:.2f}' for g in gamma_params], rotation=45)  # 标签旋转避免重叠

# Y轴（损失效用UL）：适配新数据范围+标注空间
y_min = 0.915  # 下扩至0.798，避免min UL贴边
y_max = 0.950  # 上扩至0.884，容纳max GC的标注
ax.set_ylabel('Group Consensus Degree (GCD)', fontsize=12, fontweight='bold')
ax.set_ylim(y_min, y_max)

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

# 网格线与边框优化（SCI风格）
ax.yaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
ax.set_axisbelow(True)  # 网格线在折线下方
ax.spines['top'].set_visible(False)  # 隐藏顶部边框
ax.spines['right'].set_visible(False)  # 隐藏右侧边框

# -------------------------- 6. 保存图片 --------------------------
save_dir = r"E:\newManucript\manuscript2\image"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 路径不存在则创建
save_path = os.path.join(save_dir, "fig 13(b).png")

# 高分辨率保存（SCI投稿标准）
plt.tight_layout()  # 紧凑布局，避免标注/图例截断
plt.savefig(
    save_path,
    dpi=300,  # 300dpi（SCI最低要求）
    bbox_inches='tight',  # 确保所有元素完整
    facecolor='white',  # 白色背景（避免打印问题）
    edgecolor='none'
)

plt.close(fig)  # 关闭图片，不显示
