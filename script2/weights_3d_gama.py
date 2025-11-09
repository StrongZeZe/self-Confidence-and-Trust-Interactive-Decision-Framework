
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 3D绘图依赖（旧版本兼容）
import os

#绘制最小损失模型的权重随伽马变化

# -------------------------- 1. 数据准备与预处理 --------------------------
# 1.1 基础数据定义
gamma_params = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]  # 伽马形状参数
indicators = ['Resource', 'Environment', 'External Support', 'Risk', 'Economy', 'Geology']  # 指标
# 权重数据（与gamma_params顺序对应，每个gamma对应6个指标的权重）
weights_data = [
    [0.19584951500575304, 0.21776987767707723, 0.20844194854153628, 0.15717478047406874, 0.11839032445193581, 0.10237355384962903],
    [0.195843996843899, 0.21777209794650865, 0.2084409202329875, 0.15717682154164, 0.11839226021042483, 0.10237390322453993],
    [0.19583311546094273, 0.21777647433972033, 0.20843888845523875, 0.15718084769564386, 0.11839608132512812, 0.10237459272332625],
    [0.19580166924057052, 0.2177891080767108, 0.20843298520709908, 0.15719249346068062, 0.1184071543601249, 0.10237658965481404],
    [0.1957529882166098, 0.21780862507900123, 0.2084237447129651, 0.1572105567802345, 0.11842439040590104, 0.10237969480528823],
    [0.1956678292967253, 0.2178426434432637, 0.20840722364603662, 0.1572422828639546, 0.11845484765818434, 0.10238517309183544],
    [0.1954828820257164, 0.21791599165938152, 0.20836905680053933, 0.15731207149137036, 0.11852265624586497, 0.10239734177712737],
    [0.19530484195389902, 0.21798608018370824, 0.20832699790325823, 0.15738148175190592, 0.11859099223332129, 0.10240960597390746]
]

# 1.2 x轴等间距处理（关键：用索引作为x轴坐标，标签保留原始参数值）
x_equal = np.arange(len(gamma_params))  # x轴等间距坐标（0,1,2,...,7）
y = np.arange(len(indicators))          # y轴坐标（0,1,2,3,4,5）
X, Y = np.meshgrid(x_equal, y)          # 生成网格坐标（适配3D面绘制）

# 1.3 权重数据转网格格式（与X/Y维度匹配）
Z = np.array(weights_data).T  # 转置：行=指标，列=伽马参数（匹配X/Y的网格结构）


# -------------------------- 2. 绘图参数设置（SCI格式+旧版本兼容） --------------------------
plt.rcParams['font.sans-serif'] = ['Arial']  # SCI标准英文字体（避免中文乱码）
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
plt.rcParams['font.size'] = 10               # 基础字体大小
plt.rcParams['axes.labelsize'] = 12          # 坐标轴标签字体大小
plt.rcParams['axes.labelweight'] = 'bold'    # 坐标轴标签加粗
plt.rcParams['figure.titlesize'] = 14        # 图标题字体大小
plt.rcParams['figure.titleweight'] = 'bold'  # 图标题加粗

# 创建3D图（旧版本需显式指定projection='3d'）
fig = plt.figure(figsize=(12, 8))  # 图尺寸（SCI常用宽高比，确保指标名称不拥挤）
ax = fig.add_subplot(111, projection='3d')


# -------------------------- 3. 绘制3D平面图 --------------------------
# 绘制表面图（cmap用SCI常用配色，alpha=0.8平衡清晰度与层次感）
surf = ax.plot_surface(
    X, Y, Z,
    cmap='viridis',          # 科学配色（区分度高，适合权重差异展示）
    alpha=0.8,               # 半透明效果
    edgecolor='black',       # 网格边缘线（增强轮廓，避免面重叠模糊）
    linewidth=0.3            # 边缘线宽度（细线条不干扰视觉）
)

# 添加颜色条（标注权重数值范围，SCI图表必备）
cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.6,              # 缩小颜色条尺寸，避免遮挡
    aspect=20,               # 颜色条长宽比
    pad=0.1                  # 与图的间距
)
cbar.set_label('Weight', fontsize=12, fontweight='bold')  # 颜色条标签


# -------------------------- 4. 坐标轴与标题设置（关键：x轴等间距+标签对应） --------------------------
# X轴（伽马形状参数）：坐标等间距，标签显示原始参数值
ax.set_xticks(x_equal)
ax.set_xticklabels([f'{g}' for g in gamma_params], rotation=45)  # 标签旋转避免重叠
ax.set_xlabel('Gamma Shape Param. (γ)', fontsize=12, fontweight='bold', labelpad=15)  # 英文缩写+标签间距

# Y轴（指标）：坐标对应指标索引，标签显示指标名称
ax.set_yticks(y)
ax.set_yticklabels(indicators)
ax.set_ylabel('Index', fontsize=12, fontweight='bold', labelpad=15)

# Z轴（权重）：设置合理范围，确保数据分布清晰
z_min, z_max = np.min(Z), np.max(Z)
ax.set_zlim(z_min - 0.005, z_max + 0.005)  # 预留上下空间，避免数据贴边
ax.set_zlabel('Weight', fontsize=12, fontweight='bold', labelpad=15)

# 图标题
ax.set_title('The impact of shape parameter variations on weights (min UL-Loss)',
             y=1.02,  # 标题向上偏移，避免遮挡
             fontsize=14,
             fontweight='bold')

# 3D视角调整（优化观察角度，确保所有坐标轴标签和表面清晰可见）
ax.view_init(elev=20, azim=45)  # elev=仰角，azim=方位角（多次测试的最优视角）


# -------------------------- 5. 保存图片（确保路径存在+高分辨率） --------------------------
# 确保保存目录存在（不存在则创建）
save_dir = r"E:\newManucript\manuscript2\image"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "逆伽马形状参数敏感性分析（min_UL权重）.png")

# 保存：SCI投稿标准（dpi=300，bbox_inches='tight'避免标签截断）
plt.tight_layout()
plt.savefig(
    save_path,
    dpi=300,                  # 高分辨率（SCI最低要求300dpi）
    bbox_inches='tight',      # 紧凑布局，防止标签被截断
    facecolor='white',        # 白色背景（避免透明背景打印问题）
    edgecolor='none'          # 无边缘色
)

# 关闭图片，不显示（符合需求）
plt.close(fig)

#伽马群共识最优

# -------------------------- 1. 数据准备与预处理 --------------------------
# 伽马形状参数（x轴）
gamma_params = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
# 指标名称（y轴）
indicators = ['Resource', 'Environment', 'External Support', 'Risk', 'Economy', 'Geology']
# 新权重数据（按gamma参数顺序，每个gamma对应6个指标权重）
weights_data = [
    [0.19576912210809108, 0.2151628926590241, 0.2006883939288368, 0.1610827676242435, 0.1239090560779581, 0.10338776760184641],
    [0.1957633792361643, 0.21516511641675845, 0.20068742968315004, 0.16108495712399806, 0.12391103753809395, 0.10338808000183525],
    [0.19575205478256366, 0.21516949951209483, 0.20068552419182817, 0.16108927599697623, 0.12391494900544613, 0.10338869651109102],
    [0.1957193284554168, 0.21518215121810222, 0.20067998557377323, 0.16110176787396072, 0.12392628496622564, 0.10339048191252137],
    [0.19566866687461476, 0.21520169142415888, 0.200671308447923, 0.16112114165555394, 0.12394393384230072, 0.10339325775544868],
    [0.1955800478199744, 0.21523573360705117, 0.2006557687109724, 0.16115516261796617, 0.12397513321794412, 0.10339815402609176],
    [0.19538762028030232, 0.21530901891585996, 0.2006197029917541, 0.16122995527172307, 0.12404467604017244, 0.10340902650018809],
    [0.19520247194321813, 0.2153787636905487, 0.20057959008488027, 0.16130424172166966, 0.12411494922886919, 0.10341998333081402]
]

# x轴等间距处理（用索引作为坐标，保证间距一致）
x_equal = np.arange(len(gamma_params))  # 等间距坐标：0,1,2,...,7
y = np.arange(len(indicators))          # y轴坐标：0-5
X, Y = np.meshgrid(x_equal, y)          # 生成网格坐标

# 权重数据转网格格式（与X/Y匹配）
Z = np.array(weights_data).T  # 转置后：行=指标，列=伽马参数


# -------------------------- 2. 绘图参数设置 --------------------------
plt.rcParams['font.sans-serif'] = ['Arial']  # SCI标准英文字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
plt.rcParams['font.size'] = 10               # 基础字体大小
plt.rcParams['axes.labelsize'] = 12          # 坐标轴标签大小
plt.rcParams['axes.labelweight'] = 'bold'    # 坐标轴标签加粗
plt.rcParams['figure.titlesize'] = 14        # 图标题大小
plt.rcParams['figure.titleweight'] = 'bold'  # 图标题加粗

# 创建3D图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


# -------------------------- 3. 绘制3D表面图 --------------------------
surf = ax.plot_surface(
    X, Y, Z,
    cmap='viridis',          # 科学配色（区分度高）
    alpha=0.8,               # 半透明效果
    edgecolor='black',       # 网格边缘线增强轮廓
    linewidth=0.3            # 细边缘线避免干扰
)

# 添加颜色条（标注权重范围）
cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.6,
    aspect=20,
    pad=0.1
)
cbar.set_label('Weight', fontsize=12, fontweight='bold')


# -------------------------- 4. 坐标轴与标题设置 --------------------------
# X轴（伽马参数）：等间距坐标+原始参数标签
ax.set_xticks(x_equal)
ax.set_xticklabels([f'{g}' for g in gamma_params], rotation=45)
ax.set_xlabel('Gamma Shape Param. (γ)', fontsize=12, fontweight='bold', labelpad=15)

# Y轴（指标）
ax.set_yticks(y)
ax.set_yticklabels(indicators)
ax.set_ylabel('Index', fontsize=12, fontweight='bold', labelpad=15)

# Z轴（权重）
z_min, z_max = np.min(Z), np.max(Z)
ax.set_zlim(z_min - 0.005, z_max + 0.005)
ax.set_zlabel('Weight', fontsize=12, fontweight='bold', labelpad=15)

# 图标题（更新为新名称）
ax.set_title('The impact of shape parameter variations on weights (max GCD)',
             y=1.02,  # 标题上移避免遮挡
             fontsize=14,
             fontweight='bold')

# 优化3D视角
ax.view_init(elev=20, azim=45)  # 兼顾清晰度和数据可读性


# -------------------------- 5. 保存图片 --------------------------
save_dir = r"E:\newManucript\manuscript2\image"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "逆伽马形状参数敏感性分析（max_GCD权重）.png")

# 高分辨率保存（符合SCI要求）
plt.tight_layout()
plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

plt.close(fig)  # 不显示图片

#平衡参数效用最优           犹豫模糊熵中的平衡参数对结果的影响分析（min UL）（max GC）weight
'''
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 兼容旧版本3D绘图
import os

# -------------------------- 1. 数据准备与预处理 --------------------------
# 平衡参数（x轴，按顺序排列）
balance_params = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
# 指标名称（y轴）
indicators = ['Resource', 'Environment', 'External Support', 'Risk', 'Economy', 'Geology']
# 指标权重数据（按平衡参数顺序，每个参数对应6个指标权重）
weights_data = [
    [0.1958252025661872, 0.2177678623124257, 0.20834603695645748, 0.15725302409068223, 0.11842475861722772, 0.1023831154570195],
    [0.19587833833957463, 0.2177577541522793, 0.20839625683103885, 0.15720147797907863, 0.11839069479959968, 0.10237547789842888],
    [0.19586712214829935, 0.2177629507724306, 0.20841961547526838, 0.15718709963804364, 0.11838895916035629, 0.10237425280560176],
    [0.1958439229135969, 0.2177718074972107, 0.20842877713421357, 0.15718604963550745, 0.11839464494069085, 0.10237479787878052],
    [0.19582137940109187, 0.21778081266555976, 0.2084321763947922, 0.1571887667194135, 0.11840117280995321, 0.10237569200918943],
    [0.19580166924057052, 0.2177891080767108, 0.20843298520709908, 0.15719249346068062, 0.1184071543601249, 0.10237658965481404],
    [0.19578482719656856, 0.2177965284382536, 0.20843256018037704, 0.15719632495140912, 0.11841235461921047, 0.10237740461418114],
    [0.19577045521561662, 0.21780311023894838, 0.20843153863807434, 0.15719995318380886, 0.11841681997690666, 0.10237812274664508],
    [0.19575812377500312, 0.2178089448947707, 0.20843023986778011, 0.1572032844263641, 0.11842065642738389, 0.10237875060869815],
    [0.19574746280992777, 0.21781413105174607, 0.20842883065158851, 0.1572063060178725, 0.11842396967824315, 0.10237929979062202],
    [0.19573904707270645, 0.21781831925436532, 0.208427542520821, 0.15720877507049597, 0.1184265796156975, 0.10237973646591396]
]

# x轴等间距处理（关键：用索引作坐标，避免小参数重合）
x_equal = np.arange(len(balance_params))  # 等间距坐标（0-10）
y = np.arange(len(indicators))            # y轴指标索引（0-5）
X, Y = np.meshgrid(x_equal, y)            # 生成3D绘图网格

# 权重数据转网格格式（与X/Y维度匹配：行=指标，列=平衡参数）
Z = np.array(weights_data).T


# -------------------------- 2. 绘图参数设置（SCI格式+旧版本兼容） --------------------------
plt.rcParams['font.sans-serif'] = ['Arial']  # SCI标准英文字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
plt.rcParams['font.size'] = 10               # 基础字体大小
plt.rcParams['axes.labelsize'] = 12          # 坐标轴标签大小
plt.rcParams['axes.labelweight'] = 'bold'    # 坐标轴标签加粗
plt.rcParams['figure.titlesize'] = 14        # 图标题大小
plt.rcParams['figure.titleweight'] = 'bold'  # 图标题加粗
plt.rcParams['figure.figsize'] = (14, 8)     # 加宽图幅，适配11个x轴点+指标名称


# -------------------------- 3. 创建3D平面图 --------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 旧版本显式指定3D投影

# 绘制3D表面图（SCI常用配色，增强可读性）
surf = ax.plot_surface(
    X, Y, Z,
    cmap='viridis',          # 色盲友好配色，区分权重差异
    alpha=0.8,               # 半透明效果，避免面重叠模糊
    edgecolor='black',       # 黑色细边框，增强网格轮廓
    linewidth=0.3            # 边框线宽，不干扰视觉
)

# 添加颜色条（标注权重范围，SCI图表必备）
cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.6,              # 缩小颜色条，避免遮挡
    aspect=20,               # 颜色条长宽比
    pad=0.1                  # 与图间距
)
cbar.set_label('Weight', fontsize=12, fontweight='bold')  # 颜色条标签（加粗）


# -------------------------- 4. 坐标轴与标题设置 --------------------------
# X轴（平衡参数）：等间距坐标+原始参数标签
ax.set_xticks(x_equal)
# 标签旋转45度避免重叠，0.01/0.99显示2位小数，其余1位
xtick_labels = [f'{p:.2f}' if p in [0.01, 0.99] else f'{p:.1f}' for p in balance_params]
ax.set_xticklabels(xtick_labels, rotation=45)
ax.set_xlabel('Balance Param. (α)', fontsize=12, fontweight='bold', labelpad=15)  # 英文缩写+间距

# Y轴（指标）：索引坐标+指标名称
ax.set_yticks(y)
ax.set_yticklabels(indicators)
ax.set_ylabel('Index', fontsize=12, fontweight='bold', labelpad=15)

# Z轴（权重）：合理范围，容纳所有数据
z_min, z_max = np.min(Z), np.max(Z)
ax.set_zlim(z_min - 0.002, z_max + 0.002)  # 预留上下空间，避免数据贴边
ax.set_zlabel('Weight', fontsize=12, fontweight='bold', labelpad=15)

# 图标题（按需求设置）
ax.set_title('Impact of Balance Parameter Variation on Weights (min UL-Loss)',
             y=1.02,  # 标题上移，避免遮挡
             fontsize=14,
             fontweight='bold')

# 3D视角优化（兼顾所有轴标签和表面清晰度）
ax.view_init(elev=25, azim=45)  # elev=仰角，azim=方位角


# -------------------------- 5. 保存图片（确保路径存在+高分辨率） --------------------------
save_dir = r"E:\newManucript\manuscript2\image"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 路径不存在则创建
save_path = os.path.join(save_dir, "平衡参数敏感性分析（min_UL权重）.png")

# 保存：SCI投稿标准（300dpi，无截断）
plt.tight_layout()
plt.savefig(
    save_path,
    dpi=300,                  # 高分辨率（SCI最低要求）
    bbox_inches='tight',      # 紧凑布局，防止标签/图例截断
    facecolor='white',        # 白色背景，避免打印问题
    edgecolor='none'
)

plt.close(fig)  # 关闭图片，不显示


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 兼容旧版本3D绘图
import os

# -------------------------- 1. 数据准备与预处理 --------------------------
# 伽马形状参数（x轴）
gamma_params = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
# 指标名称（y轴）
indicators = ['Resource', 'Environment', 'External Support', 'Risk', 'Economy', 'Geology']
# 指标权重数据（按伽马参数顺序，每个参数对应6个指标权重）
weights_data = [
    [0.19576912210809108, 0.2151628926590241, 0.2006883939288368, 0.1610827676242435, 0.1239090560779581, 0.10338776760184641],
    [0.1957633792361643, 0.21516511641675845, 0.20068742968315004, 0.16108495712399806, 0.12391103753809395, 0.10338808000183525],
    [0.19575205478256366, 0.21516949951209483, 0.20068552419182817, 0.16108927599697623, 0.12391494900544613, 0.10338869651109102],
    [0.1957193284554168, 0.21518215121810222, 0.20067998557377323, 0.16110176787396072, 0.12392628496622564, 0.10339048191252137],
    [0.19566866687461476, 0.21520169142415888, 0.200671308447923, 0.16112114165555394, 0.12394393384230072, 0.10339325775544868],
    [0.1955800478199744, 0.21523573360705117, 0.2006557687109724, 0.16115516261796617, 0.12397513321794412, 0.10339815402609176],
    [0.19538762028030232, 0.21530901891585996, 0.2006197029917541, 0.16122995527172307, 0.12404467604017244, 0.10340902650018809],
    [0.19520247194321813, 0.2153787636905487, 0.20057959008488027, 0.16130424172166966, 0.12411494922886919, 0.10341998333081402]
]

# x轴等间距处理（用索引作坐标，避免0.05/0.1/0.2重合）
x_equal = np.arange(len(gamma_params))  # 等间距坐标（0-7）
y = np.arange(len(indicators))          # y轴指标索引（0-5）
X, Y = np.meshgrid(x_equal, y)          # 生成3D绘图网格

# 权重数据转网格格式（与X/Y维度匹配：行=指标，列=伽马参数）
Z = np.array(weights_data).T


# -------------------------- 2. 绘图参数设置（SCI格式+旧版本兼容） --------------------------
plt.rcParams['font.sans-serif'] = ['Arial']  # SCI标准英文字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
plt.rcParams['font.size'] = 10               # 基础字体大小
plt.rcParams['axes.labelsize'] = 12          # 坐标轴标签大小
plt.rcParams['axes.labelweight'] = 'bold'    # 坐标轴标签加粗
plt.rcParams['figure.titlesize'] = 14        # 图标题大小
plt.rcParams['figure.titleweight'] = 'bold'  # 图标题加粗
plt.rcParams['figure.figsize'] = (12, 8)     # 适配8个x轴点+6个指标的图幅


# -------------------------- 3. 创建3D平面图 --------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 旧版本显式创建3D轴

# 绘制3D表面图（SCI常用配色，增强权重差异区分）
surf = ax.plot_surface(
    X, Y, Z,
    cmap='viridis',          # 色盲友好配色，权重梯度清晰
    alpha=0.8,               # 半透明避免面重叠模糊
    edgecolor='black',       # 黑色细边框增强网格轮廓
    linewidth=0.3            # 边框线宽，不干扰视觉
)

# 添加颜色条（标注权重范围，SCI图表必备）
cbar = fig.colorbar(
    surf, 
    ax=ax, 
    shrink=0.6,              # 缩小颜色条，避免遮挡数据
    aspect=20,               # 颜色条长宽比
    pad=0.1                  # 与图的间距
)
cbar.set_label('Weight', fontsize=12, fontweight='bold')  # 颜色条标签（加粗）


# -------------------------- 4. 坐标轴与标题设置 --------------------------
# X轴（伽马形状参数）：等间距坐标+原始参数标签
ax.set_xticks(x_equal)
# 刻度标签格式化（0.05/0.1/0.2显2位小数，其余显1位）
xtick_labels = [f'{g:.2f}' if g < 0.5 else f'{g:.1f}' for g in gamma_params]
ax.set_xticklabels(xtick_labels, rotation=45)  # 旋转避免标签重叠
ax.set_xlabel('Gamma Shape Param. (γ)', fontsize=12, fontweight='bold', labelpad=15)  # 英文缩写

# Y轴（指标）：索引坐标+指标名称
ax.set_yticks(y)
ax.set_yticklabels(indicators)
ax.set_ylabel('Index', fontsize=12, fontweight='bold', labelpad=15)

# Z轴（权重）：合理范围，容纳所有数据
z_min, z_max = np.min(Z), np.max(Z)
ax.set_zlim(z_min - 0.002, z_max + 0.002)  # 预留微小空间，避免数据贴边
ax.set_zlabel('Weight', fontsize=12, fontweight='bold', labelpad=15)

# 图标题（默认按“伽马参数-权重”主题设置，可按需修改）
ax.set_title('Impact of Gamma Shape Param. Variation on Weights', 
             y=1.02,  # 标题上移，避免遮挡
             fontsize=14, 
             fontweight='bold')

# 3D视角优化（兼顾所有轴标签和表面清晰度）
ax.view_init(elev=25, azim=45)  # elev=仰角，azim=方位角，确保无遮挡


# -------------------------- 5. 保存图片（确保路径存在+高分辨率） --------------------------
save_dir = r"E:\newManucript\manuscript2\image"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 路径不存在则自动创建
save_path = os.path.join(save_dir, "伽马形状参数-指标权重三维分析图.png")

# 保存：SCI投稿标准（300dpi，无标签截断）
plt.tight_layout()
plt.savefig(
    save_path,
    dpi=300,                  # 高分辨率（SCI最低要求）
    bbox_inches='tight',      # 紧凑布局，防止标签/颜色条截断
    facecolor='white',        # 白色背景，避免打印问题
    edgecolor='none'
)

plt.close(fig)  # 关闭图片，不显示
'''
'''
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 兼容旧版本3D绘图
import os

# -------------------------- 1. 数据准备与预处理 --------------------------
# 1.1 基础数据定义
balance_params = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]  # x轴平衡参数
indicators = ['Resource', 'Environment', 'External Support', 'Risk', 'Economy', 'Geology']  # y轴指标
# 权重数据（与balance_params顺序对应，每个参数对应6个指标权重）
weights_data = [
    [0.1958252025661872, 0.2177678623124257, 0.20834603695645748, 0.15725302409068223, 0.11842475861722772, 0.1023831154570195],
    [0.19587833833957463, 0.2177577541522793, 0.20839625683103885, 0.15720147797907863, 0.11839069479959968, 0.10237547789842888],
    [0.19586712214829935, 0.2177629507724306, 0.20841961547526838, 0.15718709963804364, 0.11838895916035629, 0.10237425280560176],
    [0.1958439229135969, 0.2177718074972107, 0.20842877713421357, 0.15718604963550745, 0.11839464494069085, 0.10237479787878052],
    [0.19582137940109187, 0.21778081266555976, 0.2084321763947922, 0.1571887667194135, 0.11840117280995321, 0.10237569200918943],
    [0.19580166924057052, 0.2177891080767108, 0.20843298520709908, 0.15719249346068062, 0.1184071543601249, 0.10237658965481404],
    [0.19578482719656856, 0.2177965284382536, 0.20843256018037704, 0.15719632495140912, 0.11841235461921047, 0.10237740461418114],
    [0.19577045521561662, 0.21780311023894838, 0.20843153863807434, 0.15719995318380886, 0.11841681997690666, 0.10237812274664508],
    [0.19575812377500312, 0.2178089448947707, 0.20843023986778011, 0.1572032844263641, 0.11842065642738389, 0.10237875060869815],
    [0.19574746280992777, 0.21781413105174607, 0.20842883065158851, 0.1572063060178725, 0.11842396967824315, 0.10237929979062202],
    [0.19573904707270645, 0.21781831925436532, 0.208427542520821, 0.15720877507049597, 0.1184265796156975, 0.10237973646591396]
]

# 1.2 x轴等间距处理（关键：用索引作为x轴坐标，标签保留原始参数值）
x_equal = np.arange(len(balance_params))  # x轴等间距坐标（0,1,2,...,10）
y = np.arange(len(indicators))            # y轴坐标（0,1,2,3,4,5）
X, Y = np.meshgrid(x_equal, y)            # 生成网格坐标（适配3D面绘制）

# 1.3 权重数据转网格格式（与X/Y维度匹配）
Z = np.array(weights_data).T  # 转置：行=指标，列=平衡参数（匹配X/Y的网格结构）


# -------------------------- 2. 绘图参数设置（SCI格式+旧版本兼容） --------------------------
plt.rcParams['font.sans-serif'] = ['Arial']  # SCI标准英文字体（避免中文乱码）
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
plt.rcParams['font.size'] = 10               # 基础字体大小
plt.rcParams['axes.labelsize'] = 12          # 坐标轴标签字体大小
plt.rcParams['axes.labelweight'] = 'bold'    # 坐标轴标签加粗
plt.rcParams['figure.titlesize'] = 14        # 图标题字体大小
plt.rcParams['figure.titleweight'] = 'bold'  # 图标题加粗
plt.rcParams['figure.figsize'] = (14, 8)     # 图尺寸（适配11个x轴点，确保指标名称不拥挤）

# 创建3D图（旧版本需显式指定projection='3d'）
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# -------------------------- 3. 绘制3D平面图 --------------------------
# 绘制表面图（cmap用SCI常用配色，alpha=0.8平衡清晰度与层次感）
surf = ax.plot_surface(
    X, Y, Z,
    cmap='viridis',          # 科学配色（区分度高，适合权重差异展示）
    alpha=0.8,               # 半透明效果
    edgecolor='black',       # 网格边缘线（增强轮廓，避免面重叠模糊）
    linewidth=0.3            # 边缘线宽度（细线条不干扰视觉）
)

# 添加颜色条（标注权重数值范围，SCI图表必备）
cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.6,              # 缩小颜色条尺寸，避免遮挡
    aspect=20,               # 颜色条长宽比
    pad=0.1                  # 与图的间距
)
cbar.set_label('Weight', fontsize=12, fontweight='bold')  # 颜色条标签


# -------------------------- 4. 坐标轴与标题设置（关键：x轴等间距+标签对应） --------------------------
# X轴（平衡参数）：坐标等间距，标签显示原始参数值
ax.set_xticks(x_equal)
# 标签格式化：0.01/0.99显2位小数，其余显1位，旋转45度避免重叠
ax.set_xticklabels(
    [f'{p:.2f}' if p in [0.01, 0.99] else f'{p:.1f}' for p in balance_params],
    rotation=45
)
ax.set_xlabel('Balance Param. (α)', fontsize=12, fontweight='bold', labelpad=15)  # 英文缩写+标签间距

# Y轴（指标）：坐标对应指标索引，标签显示指标名称
ax.set_yticks(y)
ax.set_yticklabels(indicators)
ax.set_ylabel('Index', fontsize=12, fontweight='bold', labelpad=15)

# Z轴（权重）：设置合理范围，确保数据分布清晰
z_min, z_max = np.min(Z), np.max(Z)
ax.set_zlim(z_min - 0.002, z_max + 0.002)  # 预留上下空间，避免数据贴边
ax.set_zlabel('Weight', fontsize=12, fontweight='bold', labelpad=15)

# 图标题（按需求设置）
ax.set_title('Impact of Balance Parameter Variation on Weights (max GCD)',
             y=1.02,  # 标题向上偏移，避免遮挡
             fontsize=14,
             fontweight='bold')

# 3D视角调整（优化观察角度，确保所有坐标轴标签和表面清晰可见）
ax.view_init(elev=25, azim=45)  # elev=仰角，azim=方位角（适配11个x轴点的最优视角）


# -------------------------- 5. 保存图片（确保路径存在+高分辨率） --------------------------
# 确保保存目录存在（不存在则创建）
save_dir = r"E:\newManucript\manuscript2\image"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "平衡参数敏感性分析（max_GC权重）.png")

# 保存：SCI投稿标准（dpi=300，bbox_inches='tight'避免标签截断）
plt.tight_layout()
plt.savefig(
    save_path,
    dpi=300,                  # 高分辨率（SCI最低要求300dpi）
    bbox_inches='tight',      # 紧凑布局，防止标签被截断
    facecolor='white',        # 白色背景（避免透明背景打印问题）
    edgecolor='none'          # 无边缘色
)

# 关闭图片，不显示（符合需求）
plt.close(fig)
'''