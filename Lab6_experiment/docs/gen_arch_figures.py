# -*- coding: utf-8 -*-
"""
Publication-quality regeneration of Fig 1-8 (architecture / schematic figures)
Backend: Python / matplotlib  |  Style: nature-figure NMI standards
All panel titles in English.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ── Publication rcParams ───────────────────────────────────────────────────
plt.rcParams['font.family']         = 'sans-serif'
plt.rcParams['font.sans-serif']     = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype']        = 'none'
plt.rcParams['axes.spines.right']   = False
plt.rcParams['axes.spines.top']     = False
plt.rcParams['legend.frameon']      = False
plt.rcParams['axes.linewidth']      = 1.4
plt.rcParams['xtick.major.width']   = 1.4
plt.rcParams['ytick.major.width']   = 1.4
plt.rcParams['font.size']           = 11

# ── NMI palette ───────────────────────────────────────────────────────────
C_BLUE_MAIN = "#0F4D92"
C_BLUE_SEC  = "#3775BA"
C_BLUE_SOFT = "#8AAFD4"
C_TEAL      = "#42949E"
C_GREEN     = "#2E9E44"
C_RED       = "#B64342"
C_NEUTRAL   = "#767676"
C_LIGHT     = "#D8D8D8"
C_DARK      = "#272727"
C_BG        = "#FFFFFF"

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

def add_panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='left', color=C_DARK)

def box(ax, cx, cy, w, h, text, fc, ec='white', fs=9, tc='white', style="round,pad=0.08"):
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                        boxstyle=style, facecolor=fc, edgecolor=ec, lw=1.5)
    ax.add_patch(r)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight='bold')

def arrow(ax, x1, x2, y, color=C_DARK, lw=1.8):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def varrow(ax, x, y1, y2, color=C_DARK, lw=1.8):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


# ══════════════════════════════════════════════════════════════════════════
# Fig 1 — ResNet-18 Architecture
# ══════════════════════════════════════════════════════════════════════════
def fig1():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.5); ax.axis('off')

    layers = [
        ("Input\n3×32×32",    0.7,  C_LIGHT,     C_NEUTRAL),
        ("Conv1\n64ch 3×3",   2.0,  C_BLUE_SOFT, 'white'),
        ("Layer1\n64ch ×2",   3.5,  C_BLUE_SEC,  'white'),
        ("Layer2\n128ch ×2",  5.2,  C_BLUE_MAIN, 'white'),
        ("Layer3\n256ch ×2",  6.9,  C_BLUE_MAIN, 'white'),
        ("Layer4\n512ch ×2",  8.6,  "#0A3666",   'white'),
        ("AvgPool\n4×4",      10.2, C_TEAL,      'white'),
        ("FC\n512→10",        11.7, C_GREEN,     'white'),
        ("Softmax\nOutput",   13.2, C_RED,       'white'),
    ]
    sizes = ["3×32×32","64×32×32","64×32×32","128×16×16",
             "256×8×8","512×4×4","512×1×1","10","10"]

    bw, bh = 1.1, 1.8
    for (label, cx, fc, tc), sz in zip(layers, sizes):
        box(ax, cx, 2.5, bw, bh, label, fc, 'white', fs=8.5, tc=tc)
        ax.text(cx, 1.3, sz, ha='center', va='center', fontsize=7, color=C_NEUTRAL)

    xs = [l[1] for l in layers]
    for i in range(len(xs)-1):
        arrow(ax, xs[i]+bw/2, xs[i+1]-bw/2, 2.5, color=C_NEUTRAL)

    ax.set_title("Figure 1 — ResNet-18 Architecture (CIFAR-10 Input: 3×32×32)",
                 fontsize=12, fontweight='bold', pad=10, color=C_DARK)
    fig.tight_layout(pad=1.5)
    fig.savefig(f'{OUT}/fig1_resnet18.png', dpi=180, bbox_inches='tight')
    plt.close(fig); print("fig1 done")


# ══════════════════════════════════════════════════════════════════════════
# Fig 2 — Residual Block
# ══════════════════════════════════════════════════════════════════════════
def fig2():
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.set_xlim(0, 7); ax.set_ylim(0, 9); ax.axis('off')

    steps = [
        (3.0, 8.0, "Input  x",              C_LIGHT,     C_NEUTRAL),
        (3.0, 6.8, "Conv2d 3×3\n(stride)",   C_BLUE_SEC,  'white'),
        (3.0, 5.6, "BatchNorm + ReLU",        C_BLUE_SOFT, 'white'),
        (3.0, 4.4, "Conv2d 3×3\n(stride=1)", C_BLUE_SEC,  'white'),
        (3.0, 3.2, "BatchNorm",               C_BLUE_SOFT, 'white'),
        (3.0, 1.9, "Add  (+)",                C_TEAL,      'white'),
        (3.0, 0.8, "ReLU → Output",           C_GREEN,     'white'),
    ]
    for cx, cy, lbl, fc, tc in steps:
        h = 0.65 if '\n' not in lbl else 0.75
        box(ax, cx, cy, 2.6, h, lbl, fc, 'white', fs=9, tc=tc)

    for i in range(len(steps)-2):
        varrow(ax, 3.0, steps[i][1]-0.38, steps[i+1][1]+0.38)
    varrow(ax, 3.0, steps[-2][1]-0.38, steps[-1][1]+0.25)

    # Shortcut connection
    sx = 5.2
    ax.annotate('', xy=(sx, 8.0), xytext=(3.0+1.3, 8.0),
                arrowprops=dict(arrowstyle='-', color=C_RED, lw=2.2))
    ax.plot([sx, sx], [8.0, 2.23], color=C_RED, lw=2.2)
    ax.annotate('', xy=(3.0+1.3, 2.23), xytext=(sx, 2.23),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=2.2))
    ax.text(5.65, 5.0, "Shortcut\nconnection", ha='center', fontsize=8.5,
            color=C_RED, style='italic')
    # Optional 1×1
    box(ax, 5.2, 5.0, 1.5, 0.6, "1×1 Conv\n(optional)", C_RED, 'white', fs=7.5, tc='white')

    ax.set_title("Figure 2 — Residual Block Structure",
                 fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout(pad=1.5)
    fig.savefig(f'{OUT}/fig2_residual_block.png', dpi=180, bbox_inches='tight')
    plt.close(fig); print("fig2 done")


# ══════════════════════════════════════════════════════════════════════════
# Fig 3 — Implicit GEMM (Weight × im2col = Output)
# ══════════════════════════════════════════════════════════════════════════
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    titles  = ["Weight  A\n[K × C·R·S]",
               "Input (im2col)  B\n[C·R·S × N·Oh·Ow]",
               "Output  C\n[K × N·Oh·Ow]"]
    shapes  = [(6, 9), (9, 8), (6, 8)]
    cmaps   = ['Blues', 'Oranges', 'Greens']

    for ax, title, (rows, cols), cmap in zip(axes, titles, shapes, cmaps):
        data = np.random.rand(rows, cols) * 0.5 + 0.3
        ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel(f'{cols} columns', fontsize=9)
        ax.set_ylabel(f'{rows} rows', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(C_NEUTRAL); sp.set_linewidth(1.2)

    fig.text(0.37, 0.50, '×', fontsize=32, ha='center', va='center',
             color=C_DARK, fontweight='bold')
    fig.text(0.63, 0.50, '=', fontsize=32, ha='center', va='center',
             color=C_DARK, fontweight='bold')
    fig.suptitle("Figure 3 — Implicit GEMM: Convolution as Matrix Multiplication\n"
                 "(no explicit im2col memory expansion required)",
                 fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout(pad=2)
    fig.savefig(f'{OUT}/fig3_implicit_gemm.png', dpi=180, bbox_inches='tight')
    plt.close(fig); print("fig3 done")


# ══════════════════════════════════════════════════════════════════════════
# Fig 4 — Thread Block Tile Assignment
# ══════════════════════════════════════════════════════════════════════════
def fig4():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-0.5, 10); ax.set_ylim(-0.5, 7.5); ax.axis('off')

    palette = [C_BLUE_SOFT, C_BLUE_SEC, C_BLUE_MAIN, "#0A3666",
               "#A8D8A8", C_TEAL, C_GREEN, "#1A6B3A"]
    tw, th = 1.8, 1.4
    idx = 0
    for row in range(4):
        for col in range(4):
            x = col * tw + 0.2
            y = (3 - row) * th + 1.5
            r = FancyBboxPatch((x, y), tw-0.15, th-0.1,
                               boxstyle="round,pad=0.06",
                               facecolor=palette[idx % 8], edgecolor='white', lw=1.5)
            ax.add_patch(r)
            ax.text(x+(tw-0.15)/2, y+(th-0.1)/2,
                    f'Block({row},{col})', ha='center', va='center',
                    fontsize=7.5, color='white', fontweight='bold')
            idx += 1

    ax.text(3.8, 7.1, 'N · Oh · Ow direction  (blockIdx.x)',
            ha='center', fontsize=10, color=C_DARK, fontweight='bold')
    ax.text(-0.1, 4.2, 'K direction\n(blockIdx.y)',
            ha='center', fontsize=10, color=C_DARK, fontweight='bold', rotation=90)
    ax.annotate('', xy=(7.5, 6.8), xytext=(0.2, 6.8),
                arrowprops=dict(arrowstyle='->', color=C_DARK, lw=2))
    ax.annotate('', xy=(0.0, 1.3), xytext=(0.0, 6.8),
                arrowprops=dict(arrowstyle='->', color=C_DARK, lw=2))

    ax.text(8.5, 6.5, 'Each Block:\n• 256 threads (8 warps)\n• 128×128 output tile\n• 64 outputs / thread',
            ha='left', va='top', fontsize=9, color=C_DARK,
            bbox=dict(facecolor='#EBF5FB', edgecolor=C_BLUE_SEC, boxstyle='round,pad=0.4'))
    ax.text(3.8, 0.8, 'blockIdx.z = N  (Batch dimension)',
            ha='center', fontsize=10, color=C_TEAL, fontweight='bold',
            bbox=dict(facecolor='#E0F4F4', edgecolor=C_TEAL, boxstyle='round,pad=0.3'))

    ax.set_title("Figure 4 — Implicit GEMM Thread Block Tile Assignment Strategy",
                 fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout(pad=1.5)
    fig.savefig(f'{OUT}/fig4_tile_layout.png', dpi=180, bbox_inches='tight')
    plt.close(fig); print("fig4 done")


# ══════════════════════════════════════════════════════════════════════════
# Fig 5 — Shared Memory Layout
# ══════════════════════════════════════════════════════════════════════════
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # smemweight
    ax = axes[0]
    w_data = np.random.rand(8, 128)*0.5+0.25
    pad_data = np.full((8, 4), 0.08)
    full_w = np.hstack([w_data, pad_data])
    ax.imshow(full_w, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.axvline(127.5, color=C_RED, lw=2, ls='--', label='+4 padding (bank conflict)')
    ax.set_xlabel('K direction  (128 valid + 4 padding = stride 132)', fontsize=9)
    ax.set_ylabel('CRS rows (8)', fontsize=9)
    ax.set_title('smemweight[8 × 132]\nWeight tile in shared memory', fontsize=10, fontweight='bold')
    ax.set_yticks(range(8)); ax.set_yticklabels([f'CRS_{i}' for i in range(8)], fontsize=8)
    ax.set_xticks([])
    ax.legend(loc='upper right', fontsize=8)
    ax.text(64, 8.8, 'STS: weight_sts_addr = (tx%8)*132 + (tx/8)*4',
            ha='center', fontsize=8, color=C_RED, style='italic')

    # smeminput
    ax = axes[1]
    i_data = np.random.rand(8, 128)*0.5+0.15
    ax.imshow(i_data, cmap='Oranges', vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel('Oh·Ow direction  (128 positions)', fontsize=9)
    ax.set_ylabel('CRS rows (8)', fontsize=9)
    ax.set_title('smeminput[8 × 128]\nInput (im2col) tile in shared memory', fontsize=10, fontweight='bold')
    ax.set_yticks(range(8)); ax.set_yticklabels([f'CRS_{i}' for i in range(8)], fontsize=8)
    ax.set_xticks([])
    ax.text(64, 8.8, 'STS: input_sts_addr = (tx/32)*128 + (tx%32),  +i*32',
            ha='center', fontsize=8, color='#C07010', style='italic')

    fig.suptitle("Figure 5 — Shared Memory Data Layout (one CRS iteration, 8 rows × 128 cols)",
                 fontsize=11, fontweight='bold')
    fig.tight_layout(pad=2)
    fig.savefig(f'{OUT}/fig5_shared_memory.png', dpi=180, bbox_inches='tight')
    plt.close(fig); print("fig5 done")


# ══════════════════════════════════════════════════════════════════════════
# Fig 6 — FP32 vs FP16 Bit Structure & Performance
# ══════════════════════════════════════════════════════════════════════════
def fig6():
    fig, (ax_bits, ax_perf) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Bit structure diagram
    ax_bits.axis('off'); ax_bits.set_xlim(0, 1.05); ax_bits.set_ylim(0, 1.0)

    def draw_bits(ax, y, cfg, lbl, tc):
        ax.text(-0.02, y+0.11, lbl, ha='right', va='center',
                fontsize=10, fontweight='bold', color=tc)
        x = 0.05
        for name, cnt, c in cfg:
            w = cnt * 0.028
            r = mpatches.Rectangle((x, y), w, 0.2,
                                    facecolor=c, edgecolor='white', lw=1.5)
            ax.add_patch(r)
            if cnt >= 3:
                ax.text(x+w/2, y+0.10, f'{name}\n{cnt}b',
                        ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
            x += w

    draw_bits(ax_bits, 0.60,
              [('Sign', 1, C_RED), ('Exponent', 8, '#E67E22'), ('Mantissa', 23, C_GREEN)],
              'FP32\n32 bit', C_DARK)
    draw_bits(ax_bits, 0.25,
              [('S', 1, C_RED), ('Exp', 5, '#E67E22'), ('Mantissa', 10, C_GREEN)],
              'FP16\n16 bit', C_DARK)

    for name, c in [('Sign', C_RED), ('Exponent', '#E67E22'), ('Mantissa', C_GREEN)]:
        ax_bits.bar(0, 0, color=c, label=name)
    ax_bits.legend(loc='lower right', fontsize=9)
    ax_bits.set_title('FP32 vs FP16 Bit Structure', fontsize=11, fontweight='bold')

    # Performance bar
    categories = ['Memory\nfootprint', 'Bandwidth\nefficiency',
                  'CUDA Core\nthroughput', 'Tensor Core\nthroughput']
    fp32_vals = [100, 100, 100, 0]
    fp16_vals = [50,  200, 200, 800]
    x = np.arange(len(categories)); w = 0.35
    b1 = ax_perf.bar(x-w/2, fp32_vals, w, label='FP32', color=C_BLUE_SOFT, edgecolor='white')
    b2 = ax_perf.bar(x+w/2, fp16_vals, w, label='FP16', color=C_BLUE_MAIN, edgecolor='white')
    ax_perf.set_xticks(x); ax_perf.set_xticklabels(categories, fontsize=9)
    ax_perf.set_ylabel('Relative value  (FP32 = 100)', fontsize=10)
    ax_perf.set_title('FP32 vs FP16 Performance Metrics', fontsize=11, fontweight='bold')
    ax_perf.legend(fontsize=10); ax_perf.grid(axis='y', alpha=0.3, lw=0.8)
    for bar, v in list(zip(b1, fp32_vals)) + list(zip(b2, fp16_vals)):
        if v > 0:
            ax_perf.text(bar.get_x()+bar.get_width()/2, v+8, f'{v}%',
                         ha='center', fontsize=8, fontweight='bold', color=C_DARK)

    fig.suptitle("Figure 6 — FP32 vs FP16 Data Type Comparison",
                 fontsize=12, fontweight='bold')
    fig.tight_layout(pad=2)
    fig.savefig(f'{OUT}/fig6_fp16_fp32.png', dpi=180, bbox_inches='tight')
    plt.close(fig); print("fig6 done")


# ══════════════════════════════════════════════════════════════════════════
# Fig 7 — WMMA API Flow
# ══════════════════════════════════════════════════════════════════════════
def fig7():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.5); ax.axis('off')

    steps = [
        (1.2,  3.0, "Declare\nfragments\na/b/c_frag", C_BLUE_SOFT),
        (3.2,  3.0, "fill_fragment\n(c_frag, 0.0f)",  C_BLUE_SEC),
        (5.6,  3.0, "load_matrix_sync\n(a_frag, ptr_a, K)\nload b_frag", C_BLUE_MAIN),
        (8.2,  3.0, "mma_sync\n(c,a,b,c)\nTensor Core FMA", C_TEAL),
        (10.8, 3.0, "store_matrix_sync\n(ptr_c, c_frag, N)", C_GREEN),
    ]
    for cx, cy, txt, fc in steps:
        box(ax, cx, cy, 1.95, 1.55, txt, fc, 'white', fs=8.5)

    xs = [s[0] for s in steps]
    for i in range(len(xs)-1):
        arrow(ax, xs[i]+0.975, xs[i+1]-0.975, 3.0)

    # Loop arc
    ax.annotate('', xy=(4.63, 2.22), xytext=(4.63, 1.3),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.8))
    ax.plot([4.63, 9.17], [1.3, 1.3], color=C_RED, lw=1.8)
    ax.annotate('', xy=(9.17, 2.22), xytext=(9.17, 1.3),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.8))
    ax.text(6.9, 0.85, 'for ki in range(0, K, 16)   —   iterate K/16 times',
            ha='center', fontsize=9, color=C_RED, style='italic')

    # Fragment size annotation
    for label, cx, color in [('A 16×16', 5.1, C_BLUE_SEC), ('B 16×16', 6.1, '#E67E22')]:
        r = mpatches.Rectangle((cx-0.38, 4.2), 0.75, 0.85,
                                facecolor=color, edgecolor='white', alpha=0.85)
        ax.add_patch(r)
        ax.text(cx, 4.62, label, ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold')
    ax.text(5.6, 5.25, '1 Warp (32 threads) cooperatively loads two 16×16 fragments',
            ha='center', fontsize=8.5, color=C_DARK, style='italic')

    # Tensor Core note
    ax.text(8.2, 4.8, '1 Warp, 1 clock:\n16×16×16 = 4096 FMAs',
            ha='center', fontsize=9, color=C_TEAL,
            bbox=dict(facecolor='#E0F4F4', edgecolor=C_TEAL, boxstyle='round,pad=0.35'))

    ax.set_title("Figure 7 — Tensor Core WMMA API Invocation Flow",
                 fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout(pad=1.5)
    fig.savefig(f'{OUT}/fig7_wmma.png', dpi=180, bbox_inches='tight')
    plt.close(fig); print("fig7 done")


# ══════════════════════════════════════════════════════════════════════════
# Fig 8 — Three-Layer Operator Architecture
# ══════════════════════════════════════════════════════════════════════════
def fig8():
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 8.5); ax.axis('off')

    layers = [
        (5.0, 7.0, 9.0, 1.1,
         "Python Inference Layer  (inference.py)",
         "model(images)  →  custom Conv2d module.forward()",
         C_BLUE_MAIN, "#DDEEFF"),
        (5.0, 5.4, 9.0, 1.1,
         "Registration Layer  (pytorch/setup.py)",
         "CUDAExtension → compile C++/CUDA to .so\n"
         "conv2d_optim_fp32  |  conv2d_optim_fp16  |  conv2d_optim_wmma",
         C_TEAL, "#E0F4F4"),
        (5.0, 3.7, 9.0, 1.1,
         "Dispatch Layer  (cpp/conv2d_optim_fp32.cpp)",
         "CHECK_CUDA + pack param_t struct + pybind11 interface",
         "#6E2F8E", "#F0E8F8"),
        (5.0, 2.0, 9.0, 1.1,
         "Kernel Layer  (cuda/conv2d_optim_kernel_fp32.cu)",
         "implgemm<<<grid, block>>>  —  Implicit GEMM 128×128 tile",
         C_RED, "#FDECEA"),
    ]

    for cx, cy, w, h, title, sub, dark, light in layers:
        r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                           boxstyle="round,pad=0.1",
                           facecolor=light, edgecolor=dark, lw=2)
        ax.add_patch(r)
        ax.text(cx, cy+0.18, title, ha='center', va='center',
                fontsize=11, color=dark, fontweight='bold')
        ax.text(cx, cy-0.22, sub,   ha='center', va='center',
                fontsize=8.5, color='#444', style='italic')

    ys = [l[1] for l in layers]
    for i in range(len(ys)-1):
        y1 = ys[i]   - layers[i][3]/2
        y2 = ys[i+1] + layers[i+1][3]/2
        ax.annotate('', xy=(5.8, y2), xytext=(5.8, y1),
                    arrowprops=dict(arrowstyle='->', color=C_DARK, lw=2))
        ax.text(6.2, (y1+y2)/2, 'call ↓', fontsize=8.5, color=C_NEUTRAL, style='italic')
        ax.annotate('', xy=(4.2, y1), xytext=(4.2, y2),
                    arrowprops=dict(arrowstyle='->', color=C_LIGHT, lw=1.5))
        ax.text(3.5, (y1+y2)/2, '↑ return', fontsize=8, color=C_LIGHT, style='italic')

    ax.set_title("Figure 8 — Three-Layer Operator Architecture Design",
                 fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout(pad=1.5)
    fig.savefig(f'{OUT}/fig8_arch.png', dpi=180, bbox_inches='tight')
    plt.close(fig); print("fig8 done")


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    print(f"\nFig 1-8 (publication quality, English titles) saved to {OUT}/")
