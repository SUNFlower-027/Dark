"""Draw an annotated DarkIR network architecture diagram (Chinese labels)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm

for f in ['Microsoft YaHei', 'SimHei', 'SimSun']:
    if any(f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [f]
        break
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 10.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10.5)
ax.axis('off')

def box(x, y, w, h, text, fc, ec='#333', fs=10, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       linewidth=1.5, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal', color='#111')

def arrow(x1, y1, x2, y2, color='#333', style='->', lw=1.6):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# ===== Main pipeline =====
cx = 4.0
y = 9.7
box(cx-1.3, y, 2.6, 0.55, '输入低光图 y\n(HxWx3)', '#E8F0FE', fs=10)
y -= 0.9
box(cx-1.3, y, 2.6, 0.45, 'intro  3x3 conv (3->C)', '#FFF4E6', fs=9); arrow(cx, y+0.45, cx, y+0.55)

enc_colors = ['#D5E8D4', '#C4E0C0', '#B2D6AD']
enc_specs = [
    ('EBlock x1   +   down', 'C=32,  1/2'),
    ('EBlock x2   +   down', 'C=64,  1/4'),
    ('EBlock x3   +   down', 'C=128, 1/8'),
]
for i, (t, sub) in enumerate(enc_specs):
    y -= 0.85
    box(cx-1.6, y, 3.2, 0.65, t + '\n' + sub, enc_colors[i], fs=9)
    arrow(cx, y+0.65, cx, y+0.75)
    ax.text(cx+1.75, y+0.33, '↑ 跳跃连接', fontsize=8, color='#2a7a2a', style='italic')

y -= 0.85
box(cx-1.8, y, 3.6, 0.7, '瓶颈 Bottleneck', '#F8CECC', fs=10, bold=True); arrow(cx, y+0.7, cx, y+0.8)
ax.text(cx, y-0.18, r'middle_enc(EBlockx2) -> side_out -> $\hat{x}_{\downarrow 8}$ (辅助监督 $L_{lol}$) -> middle_dec(DBlockx2) + $x_{light}$',
        ha='center', fontsize=7.5, color='#7a1f1f')

dec_colors = ['#DAE8FC', '#C6DCFA', '#B3D0F7']
dec_specs = [
    ('up  +  skip  +  DBlock x3', 'C=128, 1/4'),
    ('up  +  skip  +  DBlock x1', 'C=64,  1/2'),
    ('up  +  skip  +  DBlock x1', 'C=32,  1/1'),
]
for i, (t, sub) in enumerate(dec_specs):
    y -= 0.95
    box(cx-1.6, y, 3.2, 0.65, t + '\n' + sub, dec_colors[i], fs=9)
    arrow(cx, y+0.65, cx, y+0.75)

y -= 0.9
box(cx-1.6, y, 3.2, 0.5, 'ending 3x3 conv  +  输入残差 (x = x + y)', '#FFF4E6', fs=9); arrow(cx, y+0.5, cx, y+0.6)
y -= 0.9
box(cx-1.3, y, 2.6, 0.55, '输出恢复图 $\hat{x}$\n(HxWx3)', '#E8F0FE', fs=10, bold=True); arrow(cx, y+0.55, cx, y+0.65)

for ey, dy in [(7.35, 4.6), (6.5, 3.65), (5.65, 2.7)]:
    arrow(cx+1.6, ey, cx+1.6, dy, color='#2a7a2a', style='->', lw=1.2)

# ===== Right panel: EBlock =====
bx, by = 9.2, 6.1
box(bx, by, 6.6, 3.7, '', '#FFFFFF', ec='#888')
ax.text(bx+3.3, by+3.45, 'EBlock（编码器块 · 低光增强）', ha='center', fontsize=11, fontweight='bold', color='#1a5e1a')

def erow(ypos, t1, t2, fc):
    box(bx+0.2, ypos, 6.2, 0.62, t1 + '\n' + t2, fc, fs=8)

erow(by+2.65, '① norm -> DW3x3 + 1x1升维(2C) -> Σ膨胀分支(d=1,4,9) -> SimpleGate -> SCA -> 1x1',
     'SpAM：反残差空间注意力（SCA = GAP + 1x1 卷积），残差缩放 β', '#D5E8D4')
erow(by+1.85, '② norm -> FreMLP：rfft2 取幅度/相位 -> 仅增强幅度 -> 原相位重建 -> irfft2',
     '频域光照增强：x = y · x_freq（频域门控）+ 残差 γ', '#B2D6AD')
ax.text(bx+3.3, by+1.25, '核心：只改频谱幅度、不动相位 → 全局且跨尺度一致地提亮',
        ha='center', fontsize=8.5, color='#1a5e1a', style='italic')

# ===== Right panel: DBlock =====
bx2, by2 = 9.2, 1.9
box(bx2, by2, 6.6, 3.7, '', '#FFFFFF', ec='#888')
ax.text(bx2+3.3, by2+3.45, 'DBlock（解码器块 · 去模糊提锐度）', ha='center', fontsize=11, fontweight='bold', color='#1a3a7a')

def drow(ypos, t1, t2, fc):
    box(bx2+0.2, ypos, 6.2, 0.62, t1 + '\n' + t2, fc, fs=8)

drow(by2+2.65, '① norm -> 1x1升维(2C) + DW3x3 -> Σ膨胀分支(d=1,4,9) -> SimpleGate -> SCA -> 1x1',
     'Di-SpAM：多膨胀率 DW 卷积并联 → 大感受野空间注意力，残差缩放 β', '#DAE8FC')
drow(by2+1.85, '② norm -> 1x1升维(2C) -> SimpleGate(2C->C) -> 1x1降维',
     'Gated-FFN：空间域门控前馈（SimpleGate 替代激活），残差缩放 γ', '#B3D0F7')
ax.text(bx2+3.3, by2+1.25, '核心：大感受野建模运动模糊 → 去模糊并提锐度',
        ha='center', fontsize=8.5, color='#1a3a7a', style='italic')

ax.text(8, 10.25, 'DarkIR 网络结构流程图（非对称编码器—解码器）', ha='center', fontsize=14, fontweight='bold')
ax.text(8, 0.35, '说明：绿色=编码器(EBlock，频域低光增强)   蓝色=解码器(DBlock，空间域去模糊)   红色=瓶颈   橙色=首尾卷积   C=width(32/64)',
        ha='center', fontsize=8.5, color='#555')

out = r'd:\Dark\DarkIR\figures\network_architecture.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print('saved', out)
