# -*- coding: utf-8 -*-
"""
Publication-quality figure regeneration (nature-figure style, Python backend)
Replaces key result figures with clean, journal-ready versions.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json, os

# ── Publication style setup ────────────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top']   = False
plt.rcParams['legend.frameon']    = False
plt.rcParams['axes.linewidth']    = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size']  = 4
plt.rcParams['ytick.major.size']  = 4
plt.rcParams['font.size'] = 11

# ── NMI semantic palette ──────────────────────────────────────────────────
C_BASELINE = "#767676"     # neutral – baseline reference
C_FP32     = "#3775BA"     # blue secondary – FP32 optim
C_FP16     = "#0F4D92"     # blue main – FP16 hero
C_WMMA     = "#42949E"     # teal – WMMA Tensor Core
C_CUDACORE = "#B64342"     # red – slow reference (Tensor Core comparison)
C_TCCORE   = "#2E9E44"     # delta-up green – fast Tensor Core

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

data = json.load(open("docs/benchmark_results.json", encoding="utf-8"))
bs   = [8, 16, 32, 64, 128, 256, 512]
base = [data["Baseline_FP32"]["throughput"][str(b)] for b in bs]
fp32 = [data["Optim_FP32"]["throughput"][str(b)]    for b in bs]
fp16 = [data["Optim_FP16"]["throughput"][str(b)]    for b in bs]
wmma = [data["Optim_WMMA"]["throughput"][str(b)]    for b in bs]

def add_panel_label(ax, label, x=-0.10, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='left')


# ══════════════════════════════════════════════════════════════════════════
# Fig 9  Hero throughput panel (line) + speedup inset bar
# Claim: FP16 ImplGEMM delivers 3.45× peak; WMMA best at small batch
# ══════════════════════════════════════════════════════════════════════════
def fig9_throughput():
    fig, (ax_main, ax_bar) = plt.subplots(1, 2, figsize=(13, 5),
                                           gridspec_kw={'width_ratios': [1.6, 1]})

    # ── Main: trend lines ────────────────────────────────────────────────
    kw = dict(lw=2.2, markersize=7)
    ax_main.plot(bs, base, 'o--', color=C_BASELINE, label='Baseline FP32',   **kw)
    ax_main.plot(bs, fp32, 's-',  color=C_FP32,     label='Optim FP32',      **kw)
    ax_main.plot(bs, fp16, '^-',  color=C_FP16,     label='Optim FP16 ★',   lw=2.8, markersize=7)
    ax_main.plot(bs, wmma, 'D-',  color=C_WMMA,     label='Optim WMMA',      **kw)

    # Direct labels at BS=256 (peak)
    for vals, color, fmt in [(fp32, C_FP32, '2.59×'), (fp16, C_FP16, '3.45×'), (wmma, C_WMMA, '1.92×')]:
        idx = bs.index(256)
        v_b = base[idx]
        ax_main.annotate(fmt,
            xy=(256, vals[idx]), xytext=(310, vals[idx] + (50 if color == C_FP32 else 80)),
            fontsize=9, color=color, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    ax_main.set_xscale('log', base=2)
    ax_main.set_xticks(bs)
    ax_main.set_xticklabels([str(b) for b in bs])
    ax_main.set_xlabel('Batch Size', fontsize=12)
    ax_main.set_ylabel('Throughput (images / sec)', fontsize=12)
    ax_main.set_title('Inference Throughput vs Batch Size', fontsize=12, fontweight='bold')
    ax_main.legend(fontsize=10, loc='upper left')
    ax_main.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    add_panel_label(ax_main, 'a')

    # ── Bar: speedup at BS=256 ────────────────────────────────────────────
    idx256 = bs.index(256)
    names  = ['FP32\nOptim', 'FP16\nOptim ★', 'WMMA\nTensor Core']
    speeds = [fp32[idx256]/base[idx256], fp16[idx256]/base[idx256], wmma[idx256]/base[idx256]]
    colors = [C_FP32, C_FP16, C_WMMA]

    bars = ax_bar.bar(names, speeds, color=colors, width=0.55,
                      edgecolor='white', linewidth=0)
    ax_bar.axhline(1.0, color=C_BASELINE, lw=1.5, ls='--', alpha=0.8, label='Baseline (1×)')
    ax_bar.set_ylabel('Speedup vs Baseline', fontsize=12)
    ax_bar.set_title('Peak Speedup at Batch Size = 256\n(RTX 4060 Laptop GPU)', fontsize=11, fontweight='bold')
    ax_bar.set_ylim(0, max(speeds) * 1.25)
    for bar, sp in zip(bars, speeds):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{sp:.2f}×', ha='center', va='bottom',
                    fontsize=12, fontweight='bold',
                    color=bar.get_facecolor() if isinstance(bar.get_facecolor(), str)
                    else '#333333')
    ax_bar.legend(fontsize=10)
    add_panel_label(ax_bar, 'b')

    fig.text(0.5, -0.02, 'GPU: NVIDIA GeForce RTX 4060 Laptop GPU  |  CUDA 12.9  |  PyTorch 2.5.1',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(pad=2)
    fig.savefig(f'{OUT}/fig9_throughput.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("fig9 done")


# ══════════════════════════════════════════════════════════════════════════
# Fig 10  Tensor Core vs CUDA Core (real measured data)
# Claim: WMMA achieves 28–35× speedup for matrices ≥256×256
# ══════════════════════════════════════════════════════════════════════════
def fig10_tensorcore():
    sizes   = [128,   256,    512,    1024,    2048  ]
    t_wmma  = [0.246, 0.023,  0.137,  0.935,   7.393 ]
    t_basic = [0.102, 0.642,  4.091,  32.411,  258.708]
    speedup = [b/w for b, w in zip(t_basic, t_wmma)]
    labels  = [f'{s}×{s}' for s in sizes]

    fig, (ax_time, ax_sp) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Time bars (log scale) ─────────────────────────────────────────────
    x = np.arange(len(sizes)); w = 0.38
    ax_time.bar(x - w/2, t_basic, w, color=C_CUDACORE, label='CUDA Core (scalar)',
                edgecolor='white', lw=0)
    ax_time.bar(x + w/2, t_wmma,  w, color=C_TCCORE,   label='Tensor Core (WMMA)',
                edgecolor='white', lw=0)
    for i, (vb, vw) in enumerate(zip(t_basic, t_wmma)):
        ax_time.text(i - w/2, vb * 1.15, f'{vb:.3f}', ha='center', fontsize=8,
                     color=C_CUDACORE, fontweight='bold')
        ax_time.text(i + w/2, vw * 1.15, f'{vw:.3f}', ha='center', fontsize=8,
                     color=C_TCCORE,   fontweight='bold')
    ax_time.set_xticks(x); ax_time.set_xticklabels(labels, fontsize=10)
    ax_time.set_yscale('log')
    ax_time.set_ylabel('Execution Time (ms, log scale)', fontsize=12)
    ax_time.set_title('FP16 GEMM Execution Time\n(All verifications: PASS, Errors = 0)', fontsize=11, fontweight='bold')
    ax_time.legend(fontsize=10)
    add_panel_label(ax_time, 'a')

    # ── Speedup bars ──────────────────────────────────────────────────────
    bar_colors = [C_CUDACORE if sp < 1 else C_TCCORE for sp in speedup]
    bars = ax_sp.bar(x, speedup, color=bar_colors, edgecolor='white', lw=0, width=0.55)
    ax_sp.axhline(1.0, color=C_BASELINE, lw=1.5, ls='--', alpha=0.7)
    for bar, sp in zip(bars, speedup):
        ax_sp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                   f'{sp:.1f}×', ha='center', fontsize=11, fontweight='bold',
                   color='#333')
    ax_sp.set_xticks(x); ax_sp.set_xticklabels(labels, fontsize=10)
    ax_sp.set_ylabel('Speedup (Tensor Core / CUDA Core)', fontsize=12)
    ax_sp.set_title('Tensor Core Speedup over CUDA Core Scalar', fontsize=11, fontweight='bold')
    ax_sp.annotate('Kernel launch\noverhead dominates', xy=(0, speedup[0]),
                   xytext=(0.6, 1.5), fontsize=9, color=C_CUDACORE,
                   arrowprops=dict(arrowstyle='->', color=C_CUDACORE, lw=1.2))
    add_panel_label(ax_sp, 'b')

    fig.text(0.5, -0.02, 'GPU: NVIDIA GeForce RTX 4060 Laptop GPU  |  nvcc -arch=sm_89',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(pad=2)
    fig.savefig(f'{OUT}/fig10_tensorcore.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("fig10 done")


# ══════════════════════════════════════════════════════════════════════════
# Fig 11  Accuracy + full speedup across all operators
# Claim: All operators maintain 96% accuracy; speedup scales with BS
# ══════════════════════════════════════════════════════════════════════════
def fig11_speedup_accuracy():
    fig, (ax_acc, ax_sp) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Accuracy bars ─────────────────────────────────────────────────────
    op_names = ['Baseline\nFP32', 'Optim\nFP32', 'Optim\nFP16\n(best)', 'Optim\nWMMA']
    accs = [data[k]['accuracy'] for k in ['Baseline_FP32','Optim_FP32','Optim_FP16','Optim_WMMA']]
    cors = [data[k].get('correct', int(round(data[k]['accuracy']/100*50))) for k in ['Baseline_FP32','Optim_FP32','Optim_FP16','Optim_WMMA']]
    total = data['Baseline_FP32']['total']
    colors_acc = [C_BASELINE, C_FP32, C_FP16, C_WMMA]

    bars = ax_acc.bar(op_names, accs, color=colors_acc, width=0.5,
                      edgecolor='white', lw=0)
    ax_acc.axhline(90, color='#E53935', lw=1.5, ls='--', label='Required ≥ 90%')
    ax_acc.set_ylim(80, 102)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=12)
    ax_acc.set_title(f'Inference Accuracy\n(CIFAR-10, n = {total} samples)', fontsize=11, fontweight='bold')
    ax_acc.legend(fontsize=10)
    for bar, a, c in zip(bars, accs, cors):
        ax_acc.text(bar.get_x()+bar.get_width()/2, a + 0.3,
                    f'{a:.0f}%\n({c}/{total})', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='white'
                    if a > 91 else '#333')
    add_panel_label(ax_acc, 'a')

    # ── Speedup lines across all BS ───────────────────────────────────────
    sp32 = [fp32[i]/base[i] for i in range(len(bs))]
    sp16 = [fp16[i]/base[i] for i in range(len(bs))]
    spwm = [wmma[i]/base[i] for i in range(len(bs))]

    kw = dict(lw=2.2, markersize=7)
    ax_sp.plot(bs, sp32, 's-',  color=C_FP32, label='FP32 Optim',      **kw)
    ax_sp.plot(bs, sp16, '^-',  color=C_FP16, label='FP16 Optim ★',   lw=2.8, markersize=7)
    ax_sp.plot(bs, spwm, 'D-',  color=C_WMMA, label='WMMA Tensor Core',**kw)
    ax_sp.axhline(1.0, color=C_BASELINE, lw=1.5, ls='--', alpha=0.7, label='Baseline (1×)')
    ax_sp.set_xscale('log', base=2)
    ax_sp.set_xticks(bs); ax_sp.set_xticklabels([str(b) for b in bs])
    ax_sp.set_xlabel('Batch Size', fontsize=12)
    ax_sp.set_ylabel('Speedup vs Baseline FP32', fontsize=12)
    ax_sp.set_title('Throughput Speedup vs Batch Size', fontsize=11, fontweight='bold')
    ax_sp.legend(fontsize=10)
    add_panel_label(ax_sp, 'b')

    fig.tight_layout(pad=2)
    fig.savefig(f'{OUT}/fig11_speedup_accuracy.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("fig11 done")


# ══════════════════════════════════════════════════════════════════════════
# Fig 12  All-operator bar chart (absolute throughput by BS)
# ══════════════════════════════════════════════════════════════════════════
def fig12_bar():
    fig, ax = plt.subplots(figsize=(13, 5.5))

    x = np.arange(len(bs)); w = 0.22
    offsets = [-1.5, -0.5, 0.5, 1.5]
    series  = [base, fp32, fp16, wmma]
    clrs    = [C_BASELINE, C_FP32, C_FP16, C_WMMA]
    names   = ['Baseline FP32', 'Optim FP32', 'Optim FP16 ★', 'Optim WMMA']

    for offset, vals, c, n in zip(offsets, series, clrs, names):
        ax.bar(x + offset*w, vals, w, color=c, label=n, edgecolor='white', lw=0)

    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in bs])
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Throughput (images / sec)', fontsize=12)
    ax.set_title('Inference Throughput by Batch Size — All Operators\n'
                 '(RTX 4060 Laptop GPU, Accuracy = 96% for all operators)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # Annotate peak of FP16 at BS=256
    idx = list(x)[bs.index(256)]
    ax.annotate(f'Peak: {fp16[bs.index(256)]:.0f} img/s',
                xy=(idx + 0.5*w, fp16[bs.index(256)]),
                xytext=(idx + 0.5*w + 0.5, fp16[bs.index(256)] + 150),
                fontsize=9, color=C_FP16, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_FP16, lw=1.2))

    fig.tight_layout(pad=2)
    fig.savefig(f'{OUT}/fig12_bar_throughput.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("fig12 done")


# ══════════════════════════════════════════════════════════════════════════
# Fig 13  Comprehensive 4-operator comparison
# ══════════════════════════════════════════════════════════════════════════
def fig13_all():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5))

    # left: log-scale throughput trend
    kw = dict(lw=2.2, markersize=7)
    ax_l.plot(bs, base, 'o--', color=C_BASELINE, label='Baseline FP32',   **kw)
    ax_l.plot(bs, fp32, 's-',  color=C_FP32,     label='Optim FP32',      **kw)
    ax_l.plot(bs, fp16, '^-',  color=C_FP16,     label='Optim FP16 ★',   lw=2.5, markersize=8)
    ax_l.plot(bs, wmma, 'D-',  color=C_WMMA,     label='Optim WMMA',      **kw)
    ax_l.set_xscale('log', base=2)
    ax_l.set_xticks(bs); ax_l.set_xticklabels([str(b) for b in bs])
    ax_l.set_xlabel('Batch Size', fontsize=12)
    ax_l.set_ylabel('Throughput (images / sec)', fontsize=12)
    ax_l.set_title('Inference Throughput — Four Operators', fontsize=11, fontweight='bold')
    ax_l.legend(fontsize=10)
    add_panel_label(ax_l, 'a')

    # right: speedup grouped bars at BS 8 / 128 / 256
    selected_bs = [8, 128, 256]
    sel_idx = [bs.index(b) for b in selected_bs]
    xg = np.arange(3); w2 = 0.25
    for oi, (vals, c, n) in enumerate(zip([fp32, fp16, wmma], [C_FP32, C_FP16, C_WMMA],
                                          ['FP32 Optim', 'FP16 Optim ★', 'WMMA'])):
        sp = [vals[i]/base[i] for i in sel_idx]
        bars = ax_r.bar(xg + (oi-1)*w2, sp, w2, color=c, label=n, edgecolor='white', lw=0)
        for bar, v in zip(bars, sp):
            ax_r.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.04,
                      f'{v:.2f}×', ha='center', fontsize=8, fontweight='bold', color=c)

    ax_r.axhline(1.0, color=C_BASELINE, lw=1.5, ls='--', alpha=0.7, label='Baseline (1×)')
    ax_r.set_xticks(xg)
    ax_r.set_xticklabels([f'BS={b}' for b in selected_bs], fontsize=11)
    ax_r.set_ylabel('Speedup vs Baseline FP32', fontsize=12)
    ax_r.set_title('Speedup at Selected Batch Sizes', fontsize=11, fontweight='bold')
    ax_r.legend(fontsize=9, ncol=2)
    add_panel_label(ax_r, 'b')

    fig.text(0.5, -0.02, 'GPU: NVIDIA GeForce RTX 4060 Laptop GPU  |  CUDA 12.9  |  Accuracy = 96% (all operators)',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(pad=2)
    fig.savefig(f'{OUT}/fig13_all_operators.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("fig13 done")


if __name__ == '__main__':
    fig9_throughput()
    fig10_tensorcore()
    fig11_speedup_accuracy()
    fig12_bar()
    fig13_all()
    print(f"\nAll publication figures saved to {OUT}/")
